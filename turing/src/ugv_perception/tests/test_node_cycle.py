"""T07 ROS cycle + wire — fixture Image/CameraInfo views, SpyAdapter, no dummy camera."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sensor_msgs")
pytest.importorskip("std_msgs")

from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from ugv_perception.adapter.frame import ImageFrame
from ugv_perception.adapter.output import ADAPTER_ID, UNLABELED_NAME, RawSemOutput
from ugv_perception.compose.load import load_compose_configs
from ugv_perception.ingest.msgs import CameraInfoView, ImageView
from ugv_perception.node.cycle import cycle_on_frame, decode_cycle_frame, perception_cycle
from ugv_perception.node.wire import wire_compose_out
from ugv_perception.port.ids import MASK_ENCODING

_ROOT = Path(__file__).resolve().parents[3]
_HW = (2, 2)
_STAMP = 2_000_000_000
_NOW = _STAMP + 100_000_000
_K = (400.0, 0.0, 1.0, 0.0, 400.0, 1.0, 0.0, 0.0, 1.0)


def _views() -> tuple[ImageView, CameraInfoView]:
    data = bytes([10, 20, 30, 40, 50, 60, 70, 80, 90, 1, 2, 3])
    image = ImageView(
        stamp_ns=_STAMP,
        frame_id="camera_optical",
        height=2,
        width=2,
        encoding="rgb8",
        step=6,
        data=data,
    )
    info = CameraInfoView(
        stamp_ns=_STAMP,
        frame_id="camera_optical",
        height=2,
        width=2,
        k=_K,
    )
    return image, info


class SpyAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, frame: ImageFrame) -> RawSemOutput:
        self.calls += 1
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


def _kernels():
    return load_compose_configs(
        remap_path=_ROOT / "config" / "ontologies" / "yoloe.yaml",
        gates_path=_ROOT / "config" / "perception" / "yoloe.yaml",
        freshness_path=_ROOT / "config" / "perception" / "port.yaml",
    )


def test_cycle_missing_image_degraded_no_infer() -> None:
    spy = SpyAdapter()
    table, gates, fresh = _kernels()
    _, info = _views()
    out = perception_cycle(
        image=None,
        camera_info=info,
        now_ns=_NOW,
        adapter=spy,
        remap_table=table,
        gate_profile=gates,
        freshness_profile=fresh,
    )
    assert spy.calls == 0
    assert out.mask is None
    assert out.decision.degraded is True
    wired = wire_compose_out(out)
    assert isinstance(wired.degraded, Bool)
    assert wired.degraded.data is True
    assert wired.mask is None


def test_cycle_happy_path_sensor_stamp_on_mask() -> None:
    spy = SpyAdapter()
    table, gates, fresh = _kernels()
    image, info = _views()
    out = perception_cycle(
        image=image,
        camera_info=info,
        now_ns=_NOW,
        adapter=spy,
        remap_table=table,
        gate_profile=gates,
        freshness_profile=fresh,
    )
    assert spy.calls == 1
    assert out.mask is not None
    assert out.mask.header.stamp_ns == _STAMP
    wired = wire_compose_out(out)
    assert isinstance(wired.mask, Image)
    assert wired.mask.encoding == MASK_ENCODING
    assert wired.mask.header.frame_id == "camera_optical"
    assert wired.mask.header.stamp.sec == _STAMP // 1_000_000_000
    assert wired.degraded.data is False
    assert wired.port_meta is not None
    assert wired.port_meta.data[0] == 1.0
    assert wired.port_meta.data[2] == 1.0


def test_cycle_bad_k_is_frame_none_degraded() -> None:
    spy = SpyAdapter()
    table, gates, fresh = _kernels()
    image, info = _views()
    bad = CameraInfoView(
        stamp_ns=info.stamp_ns,
        frame_id=info.frame_id,
        height=info.height,
        width=info.width,
        k=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    out = perception_cycle(
        image=image,
        camera_info=bad,
        now_ns=_NOW,
        adapter=spy,
        remap_table=table,
        gate_profile=gates,
        freshness_profile=fresh,
    )
    assert spy.calls == 0
    assert out.decision.degraded is True
    assert out.mask is None


def test_decode_cycle_frame_is_none_for_missing_or_invalid_input() -> None:
    image, info = _views()
    assert decode_cycle_frame(None, info) is None
    assert decode_cycle_frame(image, None) is None
    bad = CameraInfoView(
        stamp_ns=info.stamp_ns,
        frame_id=info.frame_id,
        height=info.height,
        width=info.width,
        k=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    assert decode_cycle_frame(image, bad) is None
    frame = decode_cycle_frame(image, info)
    assert frame is not None
    assert frame.stamp_ns == _STAMP


def test_cycle_on_frame_decodes_nothing_and_matches_perception_cycle() -> None:
    table, gates, fresh = _kernels()
    image, info = _views()
    whole = perception_cycle(
        image=image,
        camera_info=info,
        now_ns=_NOW,
        adapter=SpyAdapter(),
        remap_table=table,
        gate_profile=gates,
        freshness_profile=fresh,
    )
    frame = decode_cycle_frame(image, info)
    spy = SpyAdapter()
    split = cycle_on_frame(
        frame=frame,
        now_ns=_NOW,
        adapter=spy,
        remap_table=table,
        gate_profile=gates,
        freshness_profile=fresh,
    )
    assert spy.calls == 1
    assert split.decision == whole.decision
    assert np.array_equal(split.mask.classes, whole.mask.classes)
    assert split.mask.header.stamp_ns == whole.mask.header.stamp_ns
    # no frame: degraded, no inference
    nothing = cycle_on_frame(
        frame=None,
        now_ns=_NOW,
        adapter=spy,
        remap_table=table,
        gate_profile=gates,
        freshness_profile=fresh,
    )
    assert spy.calls == 1
    assert nothing.mask is None
    assert nothing.decision.degraded is True


def test_node_sources_have_no_cmd_vel() -> None:
    root = Path(__file__).resolve().parents[1] / "node"
    for py in root.glob("*.py"):
        text = py.read_text(encoding="utf-8")
        assert "/cmd_vel" not in text
        assert "openvino" not in text or py.name == "adapter_node.py"
