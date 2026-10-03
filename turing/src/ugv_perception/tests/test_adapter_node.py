"""Live rclpy spin of PerceptionAdapterNode with fixture Image+CameraInfo."""

from __future__ import annotations

import time

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


def test_one_decode_per_frame_feeds_segmentation_and_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cycle decodes the Image once; depth gets that same ImageFrame instead of decoding it again."""
    import numpy as np

    from ugv_perception.node import cycle

    decodes: list[object] = []
    real_decode = cycle.decode_frame

    def counting_decode(image, info):
        frame = real_decode(image, info)
        decodes.append(frame)
        return frame

    monkeypatch.setattr(cycle, "decode_frame", counting_decode)

    class _Depth:
        def __init__(self) -> None:
            self.rgbs: list[object] = []

        def maps(self, rgb, k):
            self.rgbs.append(rgb)
            return np.full(rgb.shape[:2], 2.0, dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    rclpy.init()
    spy, depth = SpyAdapter(), _Depth()
    node = PerceptionAdapterNode(
        adapter=spy, adapter_id="yoloe", now_ns_fn=lambda: _STAMP + 100_000_000, depth=depth
    )
    try:
        img, info = _msgs()
        node._on_info(info)
        node._on_image(img)
        assert spy.calls == 1 and len(decodes) == 1, "one decode for the whole tick"
        assert len(depth.rgbs) == 1 and depth.rgbs[0] is decodes[0].rgb, "depth reuses the cycle's frame"
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


@pytest.mark.parametrize("device", ["GPU", "cuda"])
def test_gpu_overlap_runs_maps_during_compose(device: str) -> None:
    """DA3 maps() starts while RUGD infer is still running. Same stamp. CPU stays sequential.

    OpenVINO GPU and CUDA both use this pool. CUDA backends keep a private stream each so a
    host copy waits only for that net.
    """
    import numpy as np

    events: list[tuple[str, float]] = []

    class _Gpu:
        def __init__(self) -> None:
            self.device = device

    class _Depth:
        _backend = _Gpu()

        def maps(self, rgb, k):
            events.append(("depth_start", time.perf_counter()))
            time.sleep(0.06)
            events.append(("depth_end", time.perf_counter()))
            return np.ones(rgb.shape[:2], dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    class _Adapter(SpyAdapter):
        def __init__(self) -> None:
            super().__init__()
            self._backend = _Gpu()

        def infer(self, frame):
            events.append(("mask_start", time.perf_counter()))
            time.sleep(0.06)
            events.append(("mask_end", time.perf_counter()))
            return super().infer(frame)

    rclpy.init()
    node = PerceptionAdapterNode(
        adapter=_Adapter(), adapter_id="yoloe", now_ns_fn=lambda: _STAMP + 100_000_000, depth=_Depth()
    )
    try:
        img, info = _msgs()
        node._on_info(info)
        node._on_image(img)
        names = [name for name, _t in events]
        assert "mask_start" in names and "depth_start" in names
        starts = {name: t for name, t in events if name.endswith("_start")}
        ends = {name: t for name, t in events if name.endswith("_end")}
        assert starts["depth_start"] < ends["mask_end"]
        assert node.metrics.masks_published == 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_cpu_pair_stays_sequential() -> None:
    import numpy as np

    events: list[str] = []

    class _Cpu:
        device = "CPU"

    class _Depth:
        _backend = _Cpu()

        def begin_maps(self, rgb, k) -> None:
            events.append("da3_start")

        def maps(self, rgb, k):
            events.append("sequential")
            return np.ones(rgb.shape[:2], dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    class _Adapter(SpyAdapter):
        def __init__(self) -> None:
            super().__init__()
            self._backend = _Cpu()

    rclpy.init()
    node = PerceptionAdapterNode(
        adapter=_Adapter(), adapter_id="yoloe", now_ns_fn=lambda: _STAMP + 100_000_000, depth=_Depth()
    )
    try:
        img, info = _msgs()
        node._on_info(info)
        node._on_image(img)
        assert events == ["sequential"]
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_depth_failure_is_counted_and_warned_and_leaves_the_mask_alone() -> None:
    """A DA3 failure publishes no depth but is not silent: counted, logged once, every 100th after.

    perception_degraded stays the segmentation port's flag (§8.4); the hold comes from Dev 2's depth_stale."""

    from ugv_perception.node.cycle import decode_cycle_frame

    class _BrokenDepth:
        def maps(self, rgb, k):
            raise RuntimeError("DA3 exploded")

    rclpy.init()
    spy = SpyAdapter()
    node = PerceptionAdapterNode(
        adapter=spy, adapter_id="yoloe", now_ns_fn=lambda: _STAMP + 100_000_000, depth=_BrokenDepth()
    )
    warnings: list[str] = []
    node.get_logger().warning = lambda msg, **kw: warnings.append(msg)  # type: ignore[method-assign]
    try:
        img, info = _msgs()
        node._on_info(info)
        node._on_image(img)
        assert node.metrics.masks_published == 1, "the mask still goes out"
        assert node.metrics.depth_errors == 1
        assert node.metrics.degraded_true == 0, "depth failure does not touch perception_degraded"
        assert len(warnings) == 1 and "DA3 exploded" in warnings[0]
        frame = decode_cycle_frame(node._last_image, node._last_info)
        for _ in range(99):
            node._publish_depth(frame)
        assert node.metrics.depth_errors == 100
        assert len(warnings) == 2, "first failure, then every 100th"
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
