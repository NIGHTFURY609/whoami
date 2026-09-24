"""GA-Nav dense logits → RawSemOutput. No OpenVINO, no YOLO pack."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray

from ugv_perception.adapter.frame import ImageFrame
from ugv_perception.adapter.output import AdapterError, RawSemOutput

GROUP_NAMES: tuple[str, ...] = (
    "background",
    "smooth",
    "rough",
    "bumpy",
    "water",
    "obstacle",
)
N_GROUPS = 6


def load_ganav_config(path: str | Path) -> dict[str, object]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"adapter YAML missing: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if type(data) is not dict or data.get("adapter_id") != "ganav":
        raise ValueError("adapter_id must be ganav")
    return data


def preprocess_rgb(
    rgb: NDArray[np.uint8],
    *,
    img_scale_wh: tuple[int, int],
    pad_wh: tuple[int, int],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> NDArray[np.float32]:
    """keep_ratio resize into (W, H), pad bottom-right, ImageNet norm, NCHW."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise TypeError("rgb must be uint8 HWC")
    max_w, max_h = img_scale_wh
    pad_w, pad_h = pad_wh
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    scale = min(max_w / w, max_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = _resize_hwc(rgb.astype(np.float32), new_h, new_w)
    canvas = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
    canvas[:new_h, :new_w] = resized
    mean_a = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_a = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    norm = (canvas - mean_a) / std_a
    return np.transpose(norm, (2, 0, 1))


def decode_ganav_logits(
    logits: np.ndarray,
    frame: ImageFrame,
    *,
    align_corners: bool = False,
) -> RawSemOutput:
    """(1, 6, h, w) or (6, h, w) logits → labels + argmax softmax on camera HW."""
    a = np.asarray(logits, dtype=np.float64)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 3 or a.shape[0] != N_GROUPS:
        raise AdapterError(f"GA-Nav logits must be (6, h, w), got {tuple(a.shape)}")
    if np.any(~np.isfinite(a)):
        raise AdapterError("GA-Nav logits are not finite")
    rh, rw = int(frame.rgb.shape[0]), int(frame.rgb.shape[1])
    if (a.shape[1], a.shape[2]) != (rh, rw):
        a = np.stack(
            [_resize_map(a[c], rh, rw, align_corners=align_corners) for c in range(N_GROUPS)],
            axis=0,
        )
    shifted = a - a.max(axis=0, keepdims=True)
    exp = np.exp(np.clip(shifted, -80.0, 80.0))
    prob = exp / exp.sum(axis=0, keepdims=True)
    labels = prob.argmax(axis=0).astype(np.int32)
    scores = np.take_along_axis(prob, labels[None, ...], axis=0)[0].astype(np.float32)
    if np.any(~np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise AdapterError("GA-Nav scores are not finite and in [0,1]")
    return RawSemOutput(
        adapter_id="ganav",
        label_ids=labels,
        raw_scores=scores,
        id_to_name={i: GROUP_NAMES[i] for i in range(N_GROUPS)},
        stamp_ns=frame.stamp_ns,
        frame_id=frame.frame_id,
        hw=(rh, rw),
    )


class GanavAdapter:
    """infer() from a logits callable. OpenVINO wiring waits on an IR."""

    def __init__(self, logits_fn: object, *, align_corners: bool = False) -> None:
        if not callable(logits_fn):
            raise TypeError("logits_fn must be callable")
        self._logits_fn = logits_fn
        self._align_corners = align_corners

    def infer(self, frame: ImageFrame) -> RawSemOutput:
        try:
            logits = self._logits_fn(frame.rgb)
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError("GA-Nav backend failed") from exc
        return decode_ganav_logits(logits, frame, align_corners=self._align_corners)


def _resize_hwc(img: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.stack(
        [_resize_map(img[:, :, c], h, w, align_corners=False) for c in range(img.shape[2])],
        axis=2,
    ).astype(np.float32)


def _resize_map(src: np.ndarray, h: int, w: int, *, align_corners: bool) -> np.ndarray:
    mh, mw = src.shape
    if (mh, mw) == (h, w):
        return src.astype(np.float64, copy=False)
    src_f = src.astype(np.float64, copy=False)
    if align_corners:
        ys = np.linspace(0.0, mh - 1, h) if h > 1 else np.zeros(h)
        xs = np.linspace(0.0, mw - 1, w) if w > 1 else np.zeros(w)
    else:
        ys = (np.arange(h) + 0.5) * mh / h - 0.5
        xs = (np.arange(w) + 0.5) * mw / w - 0.5
        ys = np.clip(ys, 0.0, mh - 1)
        xs = np.clip(xs, 0.0, mw - 1)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    y0 = np.floor(yy).astype(np.intp)
    x0 = np.floor(xx).astype(np.intp)
    y1 = np.minimum(y0 + 1, mh - 1)
    x1 = np.minimum(x0 + 1, mw - 1)
    wy = yy - y0
    wx = xx - x0
    return (
        src_f[y0, x0] * (1.0 - wy) * (1.0 - wx)
        + src_f[y0, x1] * (1.0 - wy) * wx
        + src_f[y1, x0] * wy * (1.0 - wx)
        + src_f[y1, x1] * wy * wx
    )
