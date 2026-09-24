"""One eval of the intended Dev 1 path: Image+CameraInfo → RUGD SegFormer GPU → port.

Not outdoor. Not live_cam. Not DummySource.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ugv_perception.adapter.frame import ImageFrame
from ugv_perception.backend.rugd_live import build_live_adapter
from ugv_perception.compose import compose_tick, load_compose_configs
from ugv_perception.node.wire import wire_compose_out
from ugv_perception.port.ids import CANONICAL, CONF_ENCODING, MASK_ENCODING

_ROOT = Path(__file__).resolve().parents[3]
_IR = _ROOT / "weights" / "rugd-segformer.xml"
_PHOTO = _ROOT / "thetestimage1.jpg"
_NS = 1_000_000_000
_STAMP = 2_000_000_000
_NOW = _STAMP + 10_000_000


def _load_rgb() -> np.ndarray:
    from PIL import Image as PILImage

    if not _PHOTO.is_file():
        pytest.skip("thetestimage1.jpg missing")
    im = PILImage.open(_PHOTO).convert("RGB")
    rgb = np.asarray(im, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise AssertionError("test image must be HxWx3")
    return rgb


def _k_for(hw: tuple[int, int]) -> tuple[float, ...]:
    h, w = hw
    return (400.0, 0.0, w / 2.0, 0.0, 400.0, h / 2.0, 0.0, 0.0, 1.0)


def _stamp_of(msg) -> int:
    return int(msg.header.stamp.sec) * _NS + int(msg.header.stamp.nanosec)


def _pixels(msg) -> np.ndarray:
    return np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(
        int(msg.height), int(msg.width)
    )


def _conf(msg) -> np.ndarray:
    return np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(
        int(msg.height), int(msg.width)
    )


def _assert_wire(wired) -> None:
    assert wired.degraded is not None
    assert wired.degraded.data is False
    assert wired.mask is not None
    assert wired.mask.encoding == MASK_ENCODING
    assert _stamp_of(wired.mask) == _STAMP
    assert wired.mask.header.frame_id == "camera_optical"
    pix = _pixels(wired.mask)
    assert set(int(x) for x in np.unique(pix).tolist()) <= CANONICAL
    assert wired.confidence is not None
    assert wired.confidence.encoding == CONF_ENCODING
    assert _stamp_of(wired.confidence) == _STAMP
    conf = _conf(wired.confidence)
    assert bool(np.isfinite(conf).all())
    assert float(conf.min()) >= 0.0
    assert float(conf.max()) <= 1.0
    assert wired.port_meta is not None
    assert len(wired.port_meta.data) == 3
    assert wired.port_meta.data[0] == 1.0
    assert wired.port_meta.data[2] == 1.0


def test_v1_intended_path_at_once() -> None:
    if not _IR.is_file():
        pytest.skip("RUGD SegFormer IR missing")
    try:
        adapter = build_live_adapter(_ROOT)
    except Exception as exc:
        pytest.skip(f"OpenVINO GPU compile unavailable: {exc}")
    table, gates, fresh = load_compose_configs(
        remap_path=_ROOT / "config" / "ontologies" / "rugd.yaml",
        gates_path=_ROOT / "config" / "perception" / "rugd.yaml",
        freshness_path=_ROOT / "config" / "perception" / "port.yaml",
    )
    rgb = _load_rgb()
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    frame = ImageFrame(rgb=rgb, stamp_ns=_STAMP, frame_id="camera_optical")
    out = compose_tick(
        frame=frame,
        now_ns=_NOW,
        adapter=adapter,
        remap_table=table,
        gate_profile=gates,
        freshness_profile=fresh,
    )
    wired = wire_compose_out(out)
    _assert_wire(wired)

    pytest.importorskip("rclpy")
    pytest.importorskip("sensor_msgs")
    from builtin_interfaces.msg import Time
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import Bool, Header

    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node

    from ugv_perception.node.adapter_node import PerceptionAdapterNode, camera_info_qos

    t = Time()
    t.sec = _STAMP // _NS
    t.nanosec = _STAMP % _NS
    img = Image()
    img.header = Header(stamp=t, frame_id="camera_optical")
    img.height = h
    img.width = w
    img.encoding = "rgb8"
    img.step = w * 3
    img.data = bytes(rgb.tobytes())
    info = CameraInfo()
    info.header = Header(stamp=t, frame_id="camera_optical")
    info.height = h
    info.width = w
    info.k = list(_k_for((h, w)))

    rclpy.init()
    node = PerceptionAdapterNode(adapter=adapter, now_ns_fn=lambda: _NOW)
    helper = Node("v1_once_pub")
    pub_i = helper.create_publisher(Image, "/camera/image_raw", 10)
    pub_c = helper.create_publisher(CameraInfo, "/camera/camera_info", camera_info_qos())
    masks: list[Image] = []
    confs: list[Image] = []
    flags: list[bool] = []
    helper.create_subscription(Image, "/segmentation/mask", masks.append, 10)
    helper.create_subscription(Image, "/segmentation/confidence", confs.append, 10)
    helper.create_subscription(Bool, "/ugv/perception_degraded", lambda m: flags.append(m.data), 10)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(helper)
    try:
        for _ in range(80):
            pub_c.publish(info)
            pub_i.publish(img)
            ex.spin_once(timeout_sec=0.1)
            if masks and confs and False in flags:
                break
        assert masks, "node must publish the fresh mask"
        assert masks[0].encoding == MASK_ENCODING
        assert _stamp_of(masks[0]) == _STAMP
        assert masks[0].header.frame_id == "camera_optical"
        pix = np.frombuffer(bytes(masks[0].data), dtype=np.uint8)
        assert set(int(x) for x in np.unique(pix).tolist()) <= CANONICAL
        assert confs
        assert confs[0].encoding == CONF_ENCODING
        assert False in flags
    finally:
        ex.remove_node(node)
        ex.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
