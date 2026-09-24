"""T13 GA-Nav decode. Scripted logits. No weights, no GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ugv_perception.adapter.frame import ImageFrame
from ugv_perception.adapter.ganav import (
    GROUP_NAMES,
    GanavAdapter,
    decode_ganav_logits,
    preprocess_rgb,
)
from ugv_perception.adapter.output import AdapterError
from ugv_perception.compose import compose_tick, load_compose_configs

_ROOT = Path(__file__).resolve().parents[3]


def _frame(h: int = 4, w: int = 4) -> ImageFrame:
    return ImageFrame(
        rgb=np.zeros((h, w, 3), dtype=np.uint8),
        stamp_ns=2_000_000_000,
        frame_id="camera_optical",
    )


def test_decode_argmax_softmax_on_camera_hw() -> None:
    frame = _frame()
    logits = np.full((1, 6, 4, 4), -20.0, dtype=np.float32)
    logits[0, 2] = 8.0
    raw = decode_ganav_logits(logits, frame, align_corners=False)
    assert raw.adapter_id == "ganav"
    assert raw.stamp_ns == frame.stamp_ns
    assert raw.frame_id == frame.frame_id
    assert raw.hw == (4, 4)
    assert set(int(x) for x in np.unique(raw.label_ids).tolist()) == {2}
    assert raw.id_to_name[2] == "rough"
    assert float(raw.raw_scores.min()) > 0.9


def test_decode_upsamples_logits_before_softmax() -> None:
    frame = _frame(4, 4)
    logits = np.zeros((6, 2, 2), dtype=np.float32)
    logits[5] = 5.0
    raw = decode_ganav_logits(logits, frame, align_corners=False)
    assert raw.label_ids.shape == (4, 4)
    assert set(int(x) for x in np.unique(raw.label_ids).tolist()) == {5}
    assert raw.id_to_name[5] == GROUP_NAMES[5]


def test_decode_rejects_bad_layout_and_nan() -> None:
    frame = _frame()
    with pytest.raises(AdapterError):
        decode_ganav_logits(np.zeros((1, 4, 4, 4), dtype=np.float32), frame)
    bad = np.zeros((6, 4, 4), dtype=np.float32)
    bad[0, 0, 0] = np.nan
    with pytest.raises(AdapterError):
        decode_ganav_logits(bad, frame)


def test_preprocess_pads_to_config_canvas() -> None:
    rgb = np.zeros((100, 80, 3), dtype=np.uint8)
    out = preprocess_rgb(
        rgb,
        img_scale_wh=(300, 375),
        pad_wh=(300, 375),
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
    )
    assert out.shape == (3, 375, 300)
    assert out.dtype == np.float32


def test_compose_tick_uses_ganav_ontology() -> None:
    frame = _frame()
    logits = np.full((6, 4, 4), -20.0, dtype=np.float32)
    logits[5, :, :2] = 6.0
    logits[2, :, 2:] = 6.0
    adapter = GanavAdapter(lambda _rgb: logits)
    table, gates, fresh = load_compose_configs(
        remap_path=_ROOT / "config" / "ontologies" / "ganav.yaml",
        gates_path=_ROOT / "config" / "perception" / "ganav.yaml",
        freshness_path=_ROOT / "config" / "perception" / "port.yaml",
    )
    out = compose_tick(
        frame=frame,
        now_ns=frame.stamp_ns + 10_000_000,
        adapter=adapter,
        remap_table=table,
        gate_profile=gates,
        freshness_profile=fresh,
    )
    assert out.mask is not None
    assert out.decision.degraded is False
    left = set(int(x) for x in np.unique(out.mask.classes[:, :2]).tolist())
    right = set(int(x) for x in np.unique(out.mask.classes[:, 2:]).tolist())
    assert left == {2}
    assert right == {1}
