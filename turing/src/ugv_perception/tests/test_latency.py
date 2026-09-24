"""T11 backpressure / starve watchdog. Fixture Image, not outdoor."""

from __future__ import annotations

import threading
import time
from pathlib import Path

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
from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy

from ugv_perception.node.adapter_node import PerceptionAdapterNode, camera_info_qos
from ugv_perception.tests.fixtures import FixtureAdapter

_NS = 1_000_000_000
_STAMP = 2_000_000_000
_K = (400.0, 0.0, 1.0, 0.0, 400.0, 1.0, 0.0, 0.0, 1.0)


def _stamp(ns: int) -> Time:
    t = Time()
    t.sec = ns // _NS
    t.nanosec = ns % _NS
    return t


def _msgs(stamp_ns: int = _STAMP) -> tuple[Image, CameraInfo]:
    t = _stamp(stamp_ns)
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


def _stamp_of(msg: Image) -> int:
    return int(msg.header.stamp.sec) * _NS + int(msg.header.stamp.nanosec)


class BlockingAdapter(FixtureAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def infer(self, frame):
        self.entered.set()
        self.release.wait(timeout=8.0)
        return super().infer(frame)


def test_l1_image_and_camera_info_keep_last_depth_1() -> None:
    rclpy.init()
    node = PerceptionAdapterNode(
        adapter=FixtureAdapter(),
        adapter_id="yoloe",
        now_ns_fn=lambda: _STAMP + 10_000_000,
    )
    try:
        img_qos = node._sub_image.qos_profile
        info_qos = node._sub_info.qos_profile
        for qos in (img_qos, info_qos):
            assert qos.history == HistoryPolicy.KEEP_LAST
            assert qos.depth == 1
            assert qos.reliability == ReliabilityPolicy.RELIABLE
        assert img_qos.durability == DurabilityPolicy.VOLATILE
        assert info_qos.durability == DurabilityPolicy.TRANSIENT_LOCAL
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


@pytest.mark.parametrize("bad", [10, 0, True, 1.0])
def test_l2_queue_depth_must_be_python_int_one(bad: object) -> None:
    with pytest.raises(TypeError):
        PerceptionAdapterNode(adapter=FixtureAdapter(), queue_depth=bad)


def test_l3_metrics_do_not_write_stamp() -> None:
    text = Path(__file__).resolve().parents[1].joinpath("node/adapter_node.py").read_text(
        encoding="utf-8"
    )
    assert "header.stamp =" not in text
    metrics = Path(__file__).resolve().parents[1].joinpath("node/metrics.py").read_text(
        encoding="utf-8"
    )
    assert "header.stamp" not in metrics


def test_l4_l5_slow_infer_skips_ahead_original_stamps() -> None:
    rclpy.init()
    adapter = BlockingAdapter()
    clock = {"n": _STAMP + 10_000_000}

    def now_ns() -> int:
        return clock["n"]

    node = PerceptionAdapterNode(adapter=adapter, now_ns_fn=now_ns, adapter_id="yoloe")
    helper = Node("t11_burst")
    pub_i = helper.create_publisher(Image, "/camera/image_raw", 10)
    pub_c = helper.create_publisher(CameraInfo, "/camera/camera_info", camera_info_qos())
    masks: list[Image] = []
    helper.create_subscription(Image, "/segmentation/mask", masks.append, 10)
    img0, info = _msgs(_STAMP)
    ex_node = SingleThreadedExecutor()
    ex_obs = SingleThreadedExecutor()
    ex_node.add_node(node)
    ex_obs.add_node(helper)
    running = True

    def spin_node() -> None:
        while running:
            ex_node.spin_once(timeout_sec=0.05)

    def spin_obs() -> None:
        while running:
            ex_obs.spin_once(timeout_sec=0.05)

    t_node = threading.Thread(target=spin_node, daemon=True)
    t_obs = threading.Thread(target=spin_obs, daemon=True)
    t_node.start()
    t_obs.start()
    try:
        pub_c.publish(info)
        time.sleep(0.3)
        pub_i.publish(img0)
        assert adapter.entered.wait(timeout=3.0)
        last_ns = _STAMP
        for i in range(1, 12):
            last_ns = _STAMP + i * 1_000_000
            img, _ = _msgs(last_ns)
            pub_i.publish(img)
        adapter.release.set()
        deadline = time.time() + 3.0
        while time.time() < deadline and not masks:
            time.sleep(0.05)
        clock["n"] = last_ns + 10_000_000
        pub_i.publish(_msgs(last_ns)[0])
        deadline = time.time() + 4.0
        while time.time() < deadline:
            stamps = [_stamp_of(m) for m in masks]
            if last_ns in stamps:
                break
            time.sleep(0.05)
        assert adapter.calls >= 1
        assert adapter.calls < 12
        assert node.metrics.frames_in < 12
        assert masks
        stamps = [_stamp_of(m) for m in masks]
        assert last_ns in stamps
        assert stamps[-1] == last_ns
        for s in stamps:
            assert s != clock["n"]
    finally:
        running = False
        adapter.release.set()
        t_node.join(timeout=2.0)
        t_obs.join(timeout=2.0)
        ex_node.remove_node(node)
        ex_obs.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_l6_stale_tick_no_mask() -> None:
    rclpy.init()
    adapter = FixtureAdapter()
    node = PerceptionAdapterNode(
        adapter=adapter,
        now_ns_fn=lambda: _STAMP + 600_000_000,
        adapter_id="yoloe",
    )
    helper = Node("t11_stale")
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
        for _ in range(40):
            pub_c.publish(info)
            pub_i.publish(img)
            ex.spin_once(timeout_sec=0.05)
            if True in flags:
                break
        assert True in flags
        assert masks == []
        assert adapter.calls == 0
    finally:
        ex.remove_node(node)
        ex.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_l7_watchdog_degrades_while_infer_blocked() -> None:
    rclpy.init()
    adapter = BlockingAdapter()
    clock = {"n": _STAMP + 10_000_000}

    def now_ns() -> int:
        return clock["n"]

    node = PerceptionAdapterNode(adapter=adapter, now_ns_fn=now_ns, adapter_id="yoloe")
    helper = Node("t11_watch")
    pub_i = helper.create_publisher(Image, "/camera/image_raw", 10)
    pub_c = helper.create_publisher(CameraInfo, "/camera/camera_info", camera_info_qos())
    flags: list[bool] = []
    helper.create_subscription(Bool, "/ugv/perception_degraded", lambda m: flags.append(m.data), 10)
    img, info = _msgs()
    ex_node = SingleThreadedExecutor()
    ex_obs = SingleThreadedExecutor()
    ex_node.add_node(node)
    ex_obs.add_node(helper)
    running = True

    def spin_node() -> None:
        while running:
            ex_node.spin_once(timeout_sec=0.05)

    def spin_obs() -> None:
        while running:
            ex_obs.spin_once(timeout_sec=0.05)

    t_node = threading.Thread(target=spin_node, daemon=True)
    t_obs = threading.Thread(target=spin_obs, daemon=True)
    t_node.start()
    t_obs.start()
    try:
        pub_c.publish(info)
        for _ in range(8):
            time.sleep(0.02)
        pub_i.publish(img)
        assert adapter.entered.wait(timeout=3.0)
        assert adapter.release.is_set() is False
        clock["n"] = _STAMP + 600_000_000
        deadline = time.time() + 2.0
        while time.time() < deadline and True not in flags:
            time.sleep(0.05)
        assert True in flags
        assert adapter.release.is_set() is False
    finally:
        running = False
        adapter.release.set()
        t_node.join(timeout=2.0)
        t_obs.join(timeout=2.0)
        ex_node.remove_node(node)
        ex_obs.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_l9_no_cmd_vel() -> None:
    rclpy.init()
    node = PerceptionAdapterNode(
        adapter=FixtureAdapter(),
        adapter_id="yoloe",
        now_ns_fn=lambda: _STAMP + 10_000_000,
    )
    try:
        names = [p.topic_name for p in node.publishers]
        assert "/cmd_vel" not in names
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_l10_no_dummy_source_in_production() -> None:
    root = Path(__file__).resolve().parents[1]
    for folder in ("adapter", "ingest", "compose", "node"):
        for py in (root / folder).glob("*.py"):
            text = py.read_text(encoding="utf-8")
            assert "DummySource" not in text
            assert "LiveCameraSource" not in text


def test_l12_no_t08() -> None:
    here = Path(__file__).resolve().parent
    for py in (here / "test_latency.py", here / "test_latency_live.py"):
        text = py.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                assert "DepthFrame" not in stripped
                assert "depth_anything" not in stripped
