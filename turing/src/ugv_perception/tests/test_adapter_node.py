"""Live rclpy spin of PerceptionAdapterNode with fixture Image+CameraInfo."""

from __future__ import annotations

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("sensor_msgs")
pytest.importorskip("std_msgs")
pytest.importorskip("builtin_interfaces")

from builtin_interfaces.msg import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Header

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from ugv_perception.node.adapter_node import PerceptionAdapterNode, camera_info_qos
from ugv_perception.tests.test_node_cycle import SpyAdapter

_STAMP = 2_000_000_000
_K = (400.0, 0.0, 1.0, 0.0, 400.0, 1.0, 0.0, 0.0, 1.0)


def _msgs() -> tuple[Image, CameraInfo]:
    t = Time()
    t.sec = _STAMP // 1_000_000_000
    t.nanosec = _STAMP % 1_000_000_000
    img = Image()
    img.header = Header(stamp=t, frame_id="camera_optical")
    img.height = 2
    img.width = 2
    img.encoding = "rgb8"
    img.step = 6
    img.data = bytes([10, 20, 30, 40, 50, 60, 70, 80, 90, 1, 2, 3])
    info = CameraInfo()
    info.header = Header(stamp=t, frame_id="camera_optical")
    info.height = 2
    info.width = 2
    info.k = list(_K)
    return img, info


def test_adapter_node_spin_fixture_topics() -> None:
    rclpy.init()
    spy = SpyAdapter()
    node = PerceptionAdapterNode(
        adapter=spy,
        adapter_id="yoloe",
        now_ns_fn=lambda: _STAMP + 100_000_000,
    )
    helper = Node("test_cam_pub")
    pub_i = helper.create_publisher(Image, "/camera/image_raw", 10)
    pub_c = helper.create_publisher(CameraInfo, "/camera/camera_info", camera_info_qos())
    masks: list[Image] = []
    flags: list[bool] = []
    helper.create_subscription(Image, "/segmentation/mask", masks.append, 10)
    helper.create_subscription(Bool, "/ugv/perception_degraded", lambda m: flags.append(m.data), 10)
    img, info = _msgs()
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(helper)
    try:
        for _ in range(80):
            pub_i.publish(img)
            pub_c.publish(info)
            ex.spin_once(timeout_sec=0.05)
            if spy.calls >= 1 and masks and flags:
                break
        assert spy.calls >= 1
        assert masks, "expected /segmentation/mask from adapter_node"
        assert masks[0].header.frame_id == "camera_optical"
        assert masks[0].header.stamp.sec == _STAMP // 1_000_000_000
        assert False in flags or True in flags
    finally:
        ex.remove_node(node)
        ex.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_gpu_overlap_starts_depth_during_mask() -> None:
    import time

    import numpy as np
    from sensor_msgs.msg import PointCloud2

    from ugv_perception.adapter.frame import ImageFrame
    from ugv_perception.adapter.output import ADAPTER_ID, UNLABELED_NAME, RawSemOutput

    events: list[tuple[str, float]] = []

    class _Backend:
        device = "GPU"

    class _GpuAdapter:
        _backend = _Backend()

        def infer(self, frame: ImageFrame) -> RawSemOutput:
            events.append(("mask_start", time.perf_counter()))
            time.sleep(0.08)
            events.append(("mask_end", time.perf_counter()))
            hw = frame.rgb.shape[:2]
            return RawSemOutput(
                adapter_id=ADAPTER_ID,
                label_ids=np.ones(hw, dtype=np.int32),
                raw_scores=np.full(hw, 0.9, dtype=np.float32),
                id_to_name={0: UNLABELED_NAME, 1: "dirt_path"},
                stamp_ns=frame.stamp_ns,
                frame_id=frame.frame_id,
                hw=hw,
            )

    class _GpuDepth:
        _backend = _Backend()

        def maps(self, rgb, k):
            events.append(("depth_start", time.perf_counter()))
            time.sleep(0.08)
            events.append(("depth_end", time.perf_counter()))
            h, w = int(rgb.shape[0]), int(rgb.shape[1])
            return np.ones((h, w), dtype=np.float32), np.zeros((h * w, 3), dtype=np.float32)

    rclpy.init()
    node = PerceptionAdapterNode(
        adapter=_GpuAdapter(),
        depth=_GpuDepth(),
        adapter_id="yoloe",
        now_ns_fn=lambda: _STAMP + 100_000_000,
    )
    helper = Node("test_overlap_pub")
    pub_i = helper.create_publisher(Image, "/camera/image_raw", 10)
    pub_c = helper.create_publisher(CameraInfo, "/camera/camera_info", camera_info_qos())
    clouds: list[PointCloud2] = []
    masks: list[Image] = []
    helper.create_subscription(PointCloud2, "/perception/depth_cloud", clouds.append, 10)
    helper.create_subscription(Image, "/segmentation/mask", masks.append, 10)
    img, info = _msgs()
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(helper)
    try:
        for _ in range(80):
            pub_i.publish(img)
            pub_c.publish(info)
            ex.spin_once(timeout_sec=0.05)
            names = [name for name, _t in events]
            if "mask_end" in names and "depth_end" in names:
                for _extra in range(10):
                    ex.spin_once(timeout_sec=0.05)
                break
        names = [name for name, _t in events]
        assert "mask_start" in names and "depth_start" in names
        starts = {name: t for name, t in events if name.endswith("_start")}
        ends = {name: t for name, t in events if name.endswith("_end")}
        assert starts["depth_start"] < ends["mask_end"]
        assert clouds
        assert masks
        assert clouds[0].header.stamp.sec == masks[0].header.stamp.sec
        assert clouds[0].header.stamp.nanosec == masks[0].header.stamp.nanosec
        assert clouds[0].header.frame_id == masks[0].header.frame_id
    finally:
        ex.remove_node(node)
        ex.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_cpu_backends_stay_sequential() -> None:
    import time

    import numpy as np

    from ugv_perception.adapter.frame import ImageFrame
    from ugv_perception.adapter.output import ADAPTER_ID, UNLABELED_NAME, RawSemOutput

    events: list[str] = []

    class _Backend:
        device = "CPU"

    class _CpuAdapter:
        _backend = _Backend()

        def infer(self, frame: ImageFrame) -> RawSemOutput:
            events.append("mask_start")
            time.sleep(0.02)
            events.append("mask_end")
            hw = frame.rgb.shape[:2]
            return RawSemOutput(
                adapter_id=ADAPTER_ID,
                label_ids=np.ones(hw, dtype=np.int32),
                raw_scores=np.full(hw, 0.9, dtype=np.float32),
                id_to_name={0: UNLABELED_NAME, 1: "dirt_path"},
                stamp_ns=frame.stamp_ns,
                frame_id=frame.frame_id,
                hw=hw,
            )

    class _CpuDepth:
        _backend = _Backend()

        def maps(self, rgb, k):
            events.append("depth_start")
            h, w = int(rgb.shape[0]), int(rgb.shape[1])
            events.append("depth_end")
            return np.ones((h, w), dtype=np.float32), np.zeros((h * w, 3), dtype=np.float32)

    rclpy.init()
    node = PerceptionAdapterNode(
        adapter=_CpuAdapter(),
        depth=_CpuDepth(),
        adapter_id="yoloe",
        now_ns_fn=lambda: _STAMP + 100_000_000,
    )
    helper = Node("test_cpu_seq_pub")
    pub_i = helper.create_publisher(Image, "/camera/image_raw", 10)
    pub_c = helper.create_publisher(CameraInfo, "/camera/camera_info", camera_info_qos())
    img, info = _msgs()
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(helper)
    try:
        for _ in range(80):
            pub_i.publish(img)
            pub_c.publish(info)
            ex.spin_once(timeout_sec=0.05)
            if "depth_end" in events:
                break
        assert events[:4] == ["mask_start", "mask_end", "depth_start", "depth_end"]
    finally:
        ex.remove_node(node)
        ex.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
