"""T10 optional ROS contract. Skip if rclpy/sensor_msgs missing — not a T10 fail."""

from __future__ import annotations

import numpy as np
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
from ugv_perception.port.ids import CANONICAL, CONF_ENCODING, MASK_ENCODING
from ugv_perception.tests.fixtures import FixtureAdapter

_NS = 1_000_000_000
_STAMP = 2_000_000_000
_K = (400.0, 0.0, 1.0, 0.0, 400.0, 1.0, 0.0, 0.0, 1.0)


def _msgs() -> tuple[Image, CameraInfo]:
    t = Time()
    t.sec = _STAMP // _NS
    t.nanosec = _STAMP % _NS
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


def test_wr1_wr2_wr4_wr5_happy_topics() -> None:
    rclpy.init()
    adapter = FixtureAdapter()
    node = PerceptionAdapterNode(
        adapter=adapter,
        adapter_id="yoloe",
        now_ns_fn=lambda: _STAMP + 10_000_000,
    )
    helper = Node("t10_cam_pub")
    pub_i = helper.create_publisher(Image, "/camera/image_raw", 10)
    pub_c = helper.create_publisher(CameraInfo, "/camera/camera_info", camera_info_qos())
    masks: list[Image] = []
    confs: list[Image] = []
    cinfos: list[CameraInfo] = []
    flags: list[bool] = []
    helper.create_subscription(Image, "/segmentation/mask", masks.append, 10)
    helper.create_subscription(Image, "/segmentation/confidence", confs.append, 10)
    helper.create_subscription(CameraInfo, "/segmentation/camera_info", cinfos.append, 10)
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
            if adapter.calls >= 1 and masks and confs and cinfos and False in flags:
                break
        assert adapter.calls >= 1
        assert masks, "expected /segmentation/mask"
        assert masks[0].encoding == MASK_ENCODING
        assert masks[0].header.frame_id == "camera_optical"
        assert int(masks[0].header.stamp.sec) * _NS + int(masks[0].header.stamp.nanosec) == _STAMP
        pix = np.frombuffer(bytes(masks[0].data), dtype=np.uint8)
        assert set(int(x) for x in np.unique(pix).tolist()) <= CANONICAL
        assert confs, "confidence present iff mask"
        assert confs[0].encoding == CONF_ENCODING
        assert confs[0].header.frame_id == masks[0].header.frame_id
        conf = np.frombuffer(bytes(confs[0].data), dtype=np.float32)
        assert bool(np.isfinite(conf).all())
        assert float(conf.min()) >= 0.0
        assert float(conf.max()) <= 1.0
        assert cinfos, "expected /segmentation/camera_info"
        assert cinfos[0].header.frame_id == masks[0].header.frame_id
        assert False in flags
    finally:
        ex.remove_node(node)
        ex.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_wr3_stale_degrades_no_mask() -> None:
    rclpy.init()
    adapter = FixtureAdapter()
    node = PerceptionAdapterNode(
        adapter=adapter,
        adapter_id="yoloe",
        now_ns_fn=lambda: _STAMP + 600_000_000,
    )
    helper = Node("t10_cam_stale")
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
