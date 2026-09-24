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
