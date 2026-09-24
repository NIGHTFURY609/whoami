"""Eval 2: live PerceptionAdapterNode + Image/CameraInfo publisher.

Stand-in for Dev 5 topics. Not outdoor. Not live_cam. Not DummySource.
RUGD SegFormer-B5 OpenVINO GPU is the product adapter.
"""

from __future__ import annotations

from pathlib import Path

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

from ugv_perception.backend.rugd_live import build_live_adapter
from ugv_perception.node.adapter_node import PerceptionAdapterNode, camera_info_qos
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


@pytest.fixture(scope="module")
def rugd_adapter():
    if not _IR.is_file():
        pytest.skip("RUGD SegFormer IR missing")
    try:
        return build_live_adapter(_ROOT)
    except Exception as exc:
        pytest.skip(f"OpenVINO GPU compile unavailable: {exc}")


def _stamp_msg(ns: int) -> Time:
    t = Time()
    t.sec = ns // _NS
    t.nanosec = ns % _NS
    return t


def _stamp_of(msg: Image) -> int:
    return int(msg.header.stamp.sec) * _NS + int(msg.header.stamp.nanosec)


def _image(rgb: np.ndarray, stamp_ns: int = _STAMP, frame_id: str = "camera_optical") -> Image:
    img = Image()
    img.header = Header(stamp=_stamp_msg(stamp_ns), frame_id=frame_id)
    img.height, img.width = int(rgb.shape[0]), int(rgb.shape[1])
    img.encoding = "rgb8"
    img.step = img.width * 3
    img.data = bytes(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
    return img


def _info(
    stamp_ns: int = _STAMP,
    frame_id: str = "camera_optical",
    k: tuple[float, ...] | None = None,
    hw: tuple[int, int] | None = None,
) -> CameraInfo:
    if hw is None:
        rgb = _load_rgb()
        hw = (int(rgb.shape[0]), int(rgb.shape[1]))
    if k is None:
        k = _k_for(hw)
    info = CameraInfo()
    info.header = Header(stamp=_stamp_msg(stamp_ns), frame_id=frame_id)
    info.height, info.width = hw
    info.k = list(k)
    return info


def _run(adapter, img, info, now_ns: int = _NOW, loops: int = 80):
    rclpy.init()
    node = PerceptionAdapterNode(adapter=adapter, now_ns_fn=lambda: now_ns)
    helper = Node("eval2_pub")
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
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(helper)
    try:
        for _ in range(loops):
            pub_c.publish(info)
            pub_i.publish(img)
            ex.spin_once(timeout_sec=0.05)
            # One callback per spin_once. A current sample is mask + conf +
            # CameraInfo (§8.5) and degraded false (§8.4). Do not stop on the
            # watchdog's earlier degraded=true.
            if masks and confs and cinfos and False in flags:
                break
        return node, masks, confs, cinfos, flags
    finally:
        ex.remove_node(node)
        ex.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _assert_mask_contract(mask: Image, conf: Image | None) -> None:
    assert mask.encoding == MASK_ENCODING
    assert _stamp_of(mask) == _STAMP
    assert mask.header.frame_id == "camera_optical"
    pix = np.frombuffer(bytes(mask.data), dtype=np.uint8)
    assert set(int(x) for x in np.unique(pix).tolist()) <= CANONICAL
    assert conf is not None
    assert conf.encoding == CONF_ENCODING
    assert _stamp_of(conf) == _STAMP
    assert conf.header.frame_id == mask.header.frame_id
    c = np.frombuffer(bytes(conf.data), dtype=np.float32)
    assert bool(np.isfinite(c).all())
    assert float(c.min()) >= 0.0
    assert float(c.max()) <= 1.0


def test_eval2_fresh_photo_to_port(rugd_adapter) -> None:
    rgb = _load_rgb()
    hw = (int(rgb.shape[0]), int(rgb.shape[1]))
    node, masks, confs, cinfos, flags = _run(
        rugd_adapter, _image(rgb), _info(hw=hw)
    )
    assert masks, "fresh frame must publish a {0,1,2} mask"
    _assert_mask_contract(masks[0], confs[0] if confs else None)
    assert masks[0].height == hw[0]
    assert masks[0].width == hw[1]
    assert cinfos
    assert cinfos[0].header.frame_id == masks[0].header.frame_id
    assert False in flags


def test_eval2_fresh_photo_classes_are_canonical(rugd_adapter) -> None:
    rgb = _load_rgb()
    hw = (int(rgb.shape[0]), int(rgb.shape[1]))
    _node, masks, confs, _cinfos, flags = _run(
        rugd_adapter, _image(rgb), _info(hw=hw)
    )
    assert masks, "fresh frame must publish a {0,1,2} mask"
    _assert_mask_contract(masks[0], confs[0] if confs else None)
    pix = np.frombuffer(bytes(masks[0].data), dtype=np.uint8)
    assert pix.size == hw[0] * hw[1]
    assert set(int(x) for x in np.unique(pix).tolist()) <= CANONICAL


def test_eval2_stale_stamp_no_mask(rugd_adapter) -> None:
    rgb = _load_rgb()
    hw = (int(rgb.shape[0]), int(rgb.shape[1]))
    _node, masks, confs, _cinfos, flags = _run(
        rugd_adapter,
        _image(rgb),
        _info(hw=hw),
        now_ns=_STAMP + 600_000_000,
    )
    assert True in flags
    assert masks == []
    assert confs == []


def test_eval2_identity_k_degrades(rugd_adapter) -> None:
    rgb = _load_rgb()
    hw = (int(rgb.shape[0]), int(rgb.shape[1]))
    identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    _node, masks, confs, _cinfos, flags = _run(
        rugd_adapter, _image(rgb), _info(k=identity, hw=hw)
    )
    assert True in flags
    assert masks == []
    assert confs == []


def test_eval2_frame_id_mismatch_degrades(rugd_adapter) -> None:
    rgb = _load_rgb()
    hw = (int(rgb.shape[0]), int(rgb.shape[1]))
    _node, masks, confs, _cinfos, flags = _run(
        rugd_adapter,
        _image(rgb, frame_id="camera_optical"),
        _info(frame_id="other_optical", hw=hw),
    )
    assert True in flags
    assert masks == []
    assert confs == []


def test_eval2_starve_watchdog_without_images(rugd_adapter) -> None:
    rclpy.init()
    node = PerceptionAdapterNode(
        adapter=rugd_adapter, now_ns_fn=lambda: _NOW
    )
    helper = Node("eval2_starve")
    flags: list[bool] = []
    helper.create_subscription(
        Bool, "/ugv/perception_degraded", lambda m: flags.append(m.data), 10
    )
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(helper)
    try:
        import time

        deadline = time.time() + 2.0
        while time.time() < deadline and True not in flags:
            ex.spin_once(timeout_sec=0.05)
        assert True in flags
        assert node.metrics.frames_in == 0
        assert node.metrics.masks_published == 0
    finally:
        ex.remove_node(node)
        ex.remove_node(helper)
        node.destroy_node()
        helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_eval2_confidence_present_iff_mask(rugd_adapter) -> None:
    rgb = _load_rgb()
    hw = (int(rgb.shape[0]), int(rgb.shape[1]))
    _node, masks, confs, _cinfos, flags = _run(
        rugd_adapter, _image(rgb), _info(hw=hw)
    )
    assert masks, "fresh frame must publish a mask and its confidence"
    assert confs
    _assert_mask_contract(masks[0], confs[0])
    assert False in flags
