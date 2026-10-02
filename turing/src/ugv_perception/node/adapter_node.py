"""ROS 2 perception node: subscribe Image+CameraInfo, compose_tick, publish port."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float64MultiArray

from ugv_perception.backend.device import can_overlap_gpu
from ugv_perception.compose.load import load_compose_configs
from ugv_perception.ingest.msgs import CameraInfoView, ImageView
from ugv_perception.ingest.ros_bridge import camera_info_msg_to_view, image_msg_to_view
from ugv_perception.node.cycle import perception_cycle
from ugv_perception.node.metrics import PerceptionMetrics
from ugv_perception.node.wire import wire_compose_out

_ROOT = Path(__file__).resolve().parents[3]
_NS = 1_000_000_000


def _image_qos() -> QoSProfile:
    """KEEP_LAST depth 1. Image streams are live, not latched."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def camera_info_qos() -> QoSProfile:
    """KEEP_LAST depth 1. TRANSIENT_LOCAL receives a latched calibration."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


class _CountingAdapter:
    def __init__(self, inner: object, metrics: PerceptionMetrics) -> None:
        self._inner = inner
        self._metrics = metrics

    def infer(self, frame: object) -> object:
        self._metrics.infer_calls += 1
        return self._inner.infer(frame)


class PerceptionAdapterNode(Node):
    def __init__(
        self,
        *,
        adapter: object,
        remap_path: Path | None = None,
        gates_path: Path | None = None,
        freshness_path: Path | None = None,
        now_ns_fn: object | None = None,
        queue_depth: object | None = None,
        adapter_id: str | None = None,
        depth: object | None = None,
    ) -> None:
        if queue_depth is not None and (type(queue_depth) is not int or queue_depth != 1):
            raise TypeError("queue_depth must be Python int == 1")
        from rclpy.parameter import Parameter

        overrides = []
        if adapter_id is not None:
            if type(adapter_id) is not str or adapter_id == "":
                raise TypeError("adapter_id must be a non-empty str")
            overrides.append(Parameter("adapter", Parameter.Type.STRING, adapter_id))
        super().__init__("ugv_perception", parameter_overrides=overrides)
        self.declare_parameter("adapter", "rugd")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("queue_depth", 1)
        adapter_name = self.get_parameter("adapter").get_parameter_value().string_value
        if adapter_name not in ("rugd", "yoloe", "onnx"):
            raise ValueError("adapter must be rugd, yoloe, or onnx; live default is rugd")
        raw_depth = (
            queue_depth
            if queue_depth is not None
            else self.get_parameter("queue_depth").value
        )
        if type(raw_depth) is not int or raw_depth != 1:
            raise TypeError("queue_depth must be Python int == 1")
        remap_path = remap_path or _ROOT / "config" / "ontologies" / f"{adapter_name}.yaml"
        gates_path = gates_path or _ROOT / "config" / "perception" / f"{adapter_name}.yaml"
        freshness_path = freshness_path or _ROOT / "config" / "perception" / "port.yaml"
        table, gates, fresh = load_compose_configs(
            remap_path=remap_path,
            gates_path=gates_path,
            freshness_path=freshness_path,
        )
        self.metrics = PerceptionMetrics()
        self._adapter = _CountingAdapter(adapter, self.metrics)
        self._table = table
        self._gates = gates
        self._fresh = fresh
        self._now_ns_fn = now_ns_fn or time.time_ns
        self._last_image: ImageView | None = None
        self._last_info: CameraInfoView | None = None
        self._last_camera_info_msg: CameraInfo | None = None
        self._stamp_lock = threading.Lock()
        self._last_image_stamp: int | None = None
        self._inferred_stamp: int | None = None
        self._stop = threading.Event()
        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self._sub_image = self.create_subscription(
            Image, image_topic, self._on_image, _image_qos()
        )
        self._sub_info = self.create_subscription(
            CameraInfo, info_topic, self._on_info, camera_info_qos()
        )
        self._pub_degraded = self.create_publisher(Bool, "/ugv/perception_degraded", 10)
        self._pub_mask = self.create_publisher(Image, "/segmentation/mask", 10)
        self._pub_conf = self.create_publisher(Image, "/segmentation/confidence", 10)
        self._pub_meta = self.create_publisher(Float64MultiArray, "/segmentation/port_meta", 10)
        self._pub_cinfo = self.create_publisher(CameraInfo, "/segmentation/camera_info", 10)
        self._depth = depth
        self._depth_pool: ThreadPoolExecutor | None = None
        self._pub_cloud = None
        self._pub_depth = None
        if depth is not None:
            from sensor_msgs.msg import PointCloud2

            self._pub_cloud = self.create_publisher(PointCloud2, "/perception/depth_cloud", 10)
            self._pub_depth = self.create_publisher(Image, "/perception/depth/image", 10)
        period_s = float(self._fresh.perception_max_age) / 2.0
        self._watchdog = threading.Thread(
            target=self._watchdog_loop,
            args=(period_s,),
            name="ugv_perception_watchdog",
            daemon=True,
        )
        self._watchdog.start()

    def destroy_node(self) -> None:
        self._stop.set()
        pool = getattr(self, "_depth_pool", None)
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
            self._depth_pool = None
        wd = getattr(self, "_watchdog", None)
        if wd is not None and wd.is_alive():
            wd.join(timeout=2.0)
        super().destroy_node()

    def _on_image(self, msg: Image) -> None:
        view = image_msg_to_view(msg)
        with self._stamp_lock:
            self._last_image_stamp = view.stamp_ns
        self.metrics.frames_in += 1
        self._last_image = view
        self._tick()

    def _on_info(self, msg: CameraInfo) -> None:
        self._last_info = camera_info_msg_to_view(msg)
        self._last_camera_info_msg = msg
        with self._stamp_lock:
            stamp = self._last_image_stamp
        if stamp is not None and stamp != self._inferred_stamp:
            self._tick()

    def _tick(self) -> None:
        now_ns = self._now_ns_fn()
        if type(now_ns) is not int or now_ns <= 0:
            raise TypeError("now_ns must be a Python int > 0")
        image = self._last_image
        info = self._last_info
        had_pair = image is not None and info is not None
        depth_fut = None
        if had_pair and can_overlap_gpu(self._adapter, self._depth):
            if self._depth_pool is None:
                self._depth_pool = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="ugv_da3"
                )
            depth_fut = self._depth_pool.submit(self._depth_maps, image, info)
        out = perception_cycle(
            image=image,
            camera_info=info,
            now_ns=now_ns,
            adapter=self._adapter,
            remap_table=self._table,
            gate_profile=self._gates,
            freshness_profile=self._fresh,
        )
        wired = wire_compose_out(out)
        self._pub_degraded.publish(wired.degraded)
        if wired.degraded.data is True:
            self.metrics.degraded_true += 1
        else:
            self.metrics.degraded_false += 1
        if wired.mask is not None:
            stamp = (
                int(wired.mask.header.stamp.sec) * _NS
                + int(wired.mask.header.stamp.nanosec)
            )
            self.metrics.latencies_ns.append(now_ns - stamp)
            self.metrics.masks_published += 1
            self._pub_mask.publish(wired.mask)
            self._pub_conf.publish(wired.confidence)
            self._pub_meta.publish(wired.port_meta)
            if self._last_camera_info_msg is not None:
                self._pub_cinfo.publish(self._last_camera_info_msg)
            if depth_fut is not None:
                self._emit_depth(depth_fut)
            else:
                self._publish_depth()
        elif depth_fut is not None:
            try:
                depth_fut.result()
            except Exception:
                pass
        if had_pair and image is not None:
            self._inferred_stamp = image.stamp_ns

    def _depth_maps(
        self, image: ImageView, info: CameraInfoView
    ) -> tuple[tuple[object, object], int, str]:
        from ugv_perception.ingest.decode import decode_frame

        frame = decode_frame(image, info)
        return self._depth.maps(frame.rgb, info.k), frame.stamp_ns, frame.frame_id

    def _emit_depth(self, depth_fut: object) -> None:
        """Wait for overlapped DA3. Failure publishes no cloud and does not touch degraded."""
        try:
            (depth_m, points), stamp_ns, frame_id = depth_fut.result()
        except Exception:
            return
        try:
            from ugv_perception.node.cloud import depth_to_image, points_to_cloud

            if self._pub_depth is not None:
                self._pub_depth.publish(depth_to_image(depth_m, stamp_ns, frame_id))
            if points is not None and self._pub_cloud is not None:
                self._pub_cloud.publish(points_to_cloud(points, stamp_ns, frame_id))
        except Exception:
            return

    def _publish_depth(self) -> None:
        """Sequential DA3 after the mask. Failure publishes no cloud and does not touch degraded."""
        if self._depth is None or self._last_image is None or self._last_info is None:
            return
        try:
            from ugv_perception.ingest.decode import decode_frame
            from ugv_perception.node.cloud import depth_to_image, points_to_cloud

            frame = decode_frame(self._last_image, self._last_info)
            depth_m, points = self._depth.maps(frame.rgb, self._last_info.k)
            if self._pub_depth is not None:
                self._pub_depth.publish(
                    depth_to_image(depth_m, frame.stamp_ns, frame.frame_id)
                )
            if points is not None and self._pub_cloud is not None:
                self._pub_cloud.publish(
                    points_to_cloud(points, frame.stamp_ns, frame.frame_id)
                )
        except Exception:
            return

    def _watchdog_loop(self, period_s: float) -> None:
        max_age = float(self._fresh.perception_max_age)
        while not self._stop.wait(timeout=period_s):
            now_ns = self._now_ns_fn()
            if type(now_ns) is not int or now_ns <= 0:
                continue
            with self._stamp_lock:
                stamp = self._last_image_stamp
            if stamp is None or (now_ns - stamp) / _NS > max_age:
                try:
                    flag = Bool()
                    flag.data = True
                    self._pub_degraded.publish(flag)
                    self.metrics.degraded_true += 1
                except Exception:
                    break


def main() -> None:
    from ugv_perception.backend.depth_live import build_depth_channel
    from ugv_perception.backend.rugd_live import build_live_adapter

    rclpy.init()
    boot = rclpy.create_node("ugv_perception_boot")
    boot.declare_parameter("adapter", "rugd")
    selected = boot.get_parameter("adapter").get_parameter_value().string_value
    boot.destroy_node()
    if selected == "onnx":
        from ugv_perception.backend.onnx_live import build_onnx_adapter

        adapter = build_onnx_adapter(_ROOT)
    elif selected == "yoloe":
        from ugv_perception.adapter.prompts import load_prompts
        from ugv_perception.adapter.yoloe import YoloeAdapter, load_adapter_config
        from ugv_perception.backend.factory import build_backend

        cfg = load_adapter_config(_ROOT / "config" / "adapters" / "yoloe.yaml")
        prompts = load_prompts(
            _ROOT / "config" / "perception" / "yoloe_prompts.yaml",
            _ROOT / "config" / "ontologies" / "yoloe.yaml",
        )
        backend = build_backend(cfg["backend"], str(_ROOT / cfg["weights"]), prompts)
        adapter = YoloeAdapter(backend, prompts)
    else:
        adapter = build_live_adapter(_ROOT)
    depth = build_depth_channel(_ROOT)
    node = PerceptionAdapterNode(adapter=adapter, depth=depth, adapter_id=selected)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
