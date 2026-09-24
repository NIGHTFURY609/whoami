"""RUGD SegFormer decode. Scripted logits. No weights, no GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ugv_perception.adapter.frame import ImageFrame
from ugv_perception.adapter.output import AdapterError
from ugv_perception.adapter.rugd import CLASS_NAMES, decode_rugd_logits, preprocess_rgb
from ugv_perception.compose import compose_tick, load_compose_configs
from ugv_perception.remap.load import load_remap

_ROOT = Path(__file__).resolve().parents[3]


def _frame(h: int = 4, w: int = 6) -> ImageFrame:
    return ImageFrame(
        rgb=np.zeros((h, w, 3), dtype=np.uint8),
        stamp_ns=2_000_000_000,
        frame_id="camera_optical",
    )


def test_decode_tree_argmax_on_camera_hw() -> None:
    frame = _frame()
    logits = np.full((1, 25, 2, 2), -20.0, dtype=np.float32)
    logits[0, 4] = 8.0
    raw = decode_rugd_logits(logits, frame)
    assert raw.adapter_id == "rugd"
    assert raw.hw == (4, 6)
    assert set(int(x) for x in np.unique(raw.label_ids).tolist()) == {4}
    assert raw.id_to_name[4] == "tree"
    assert float(raw.raw_scores.min()) > 0.5


def test_decode_rejects_nan() -> None:
    logits = np.zeros((25, 2, 2), dtype=np.float32)
    logits[0, 0, 0] = np.nan
    try:
        decode_rugd_logits(logits, _frame())
    except AdapterError:
        return
    raise AssertionError("NaN logits must raise")


def test_preprocess_nchw_shape() -> None:
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    blob = preprocess_rgb(rgb, input_hw=(8, 8), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    assert blob.shape == (1, 3, 8, 8)
    assert blob.dtype == np.float32


def test_ontology_maps_dirt_and_tree() -> None:
    table = load_remap(_ROOT / "config" / "ontologies" / "rugd.yaml")
    assert table.adapter_id == "rugd"
    assert table.name_to_class["dirt"] == 1
    assert table.name_to_class["gravel"] == 1
    assert table.name_to_class["tree"] == 2
    assert table.name_to_class["sky"] == 0
    assert set(CLASS_NAMES) == set(table.name_to_class)


def test_compose_tick_uses_rugd_table() -> None:
    class _Scripted:
        def infer(self, frame: ImageFrame):
            logits = np.full((1, 25, 4, 4), -20.0, dtype=np.float32)
            logits[0, 1, :, :2] = 6.0
            logits[0, 4, :, 2:] = 6.0
            return decode_rugd_logits(logits, frame)

    table, gates, fresh = load_compose_configs(
        remap_path=_ROOT / "config" / "ontologies" / "rugd.yaml",
        gates_path=_ROOT / "config" / "perception" / "rugd.yaml",
        freshness_path=_ROOT / "config" / "perception" / "port.yaml",
    )
    frame = _frame()
    out = compose_tick(
        frame=frame,
        now_ns=frame.stamp_ns + 1_000_000,
        adapter=_Scripted(),
        remap_table=table,
        gate_profile=gates,
        freshness_profile=fresh,
    )
    assert out.mask is not None
    pix = out.mask.classes
    assert int(pix[0, 0]) == 1
    assert int(pix[0, -1]) == 2
