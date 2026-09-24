"""OpenVINO 2026.4.0 GPU backend. Import openvino only inside load/run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ugv_perception.adapter.output import AdapterError, Instance
from ugv_perception.backend.instances import instances_from_engine

_DEVICE = "GPU"
# Engine NMS (Ultralytics YOLO-seg defaults). Not T04 τ.
_CONF_THRES = 0.25
_IOU_THRES = 0.70
_MAX_DET = 300


class OpenVinoGpuBackend:
    id = "openvino_gpu"

    def __init__(self) -> None:
        self._compiled = None
        self._prompts: tuple[str, ...] | None = None

    def load(self, weights_path: str, **engine_args: object) -> None:
        prompts = engine_args.get("prompts")
        if type(prompts) is not tuple or len(prompts) == 0:
            raise TypeError("load(..., prompts=tuple[str, ...]) is required")
        path = Path(weights_path)
        if not path.is_file():
            raise FileNotFoundError(f"OpenVINO IR missing: {path}")
        try:
            import openvino as ov
        except ImportError as exc:
            raise AdapterError("openvino is not installed") from exc
        core = ov.Core()
        devices = list(core.available_devices)
        if not any(str(d).startswith(_DEVICE) for d in devices):
            raise AdapterError(f"OpenVINO {_DEVICE} not available; devices={devices}")
        try:
            model = core.read_model(str(path))
            self._compiled = core.compile_model(model, _DEVICE)
        except Exception as exc:
            raise AdapterError(f"OpenVINO compile on {_DEVICE} failed") from exc
        self._prompts = prompts

    def run(self, rgb: NDArray[np.uint8]) -> tuple[Instance, ...]:
        if self._compiled is None or self._prompts is None:
            raise AdapterError("OpenVinoGpuBackend.load() was not called")
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3:
            raise TypeError("rgb must be uint8 HWC")
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        try:
            compiled = self._compiled
            inp = compiled.inputs[0]
            shape = list(inp.shape)
            blob = _rgb_to_input(rgb, shape)
            result = compiled([blob])
            model_hw = _model_hw(shape, rgb_hw=(h, w))
            class_indices, scores, masks = _parse_ov_result(
                result, n_classes=len(self._prompts), model_hw=model_hw
            )
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError("OpenVINO GPU run failed") from exc
        return instances_from_engine(
            rgb_hw=(h, w),
            prompts=self._prompts,
            class_indices=class_indices,
            scores=scores,
            masks=masks,
        )


def _rgb_to_input(rgb: np.ndarray, shape: list[object]) -> np.ndarray:
    """Stretch to model H/W if the static shape exposes them; NCHW float32 [0,1]."""
    img = rgb.astype(np.float32) / 255.0
    dims = [int(x) if str(x).isdigit() or isinstance(x, (int, np.integer)) else -1 for x in shape]
    if len(dims) == 4 and dims[1] in (1, 3):
        mh, mw = dims[2], dims[3]
        if mh > 0 and mw > 0 and (mh, mw) != (img.shape[0], img.shape[1]):
            img = _stretch_hwc(img, mh, mw)
        return np.transpose(img, (2, 0, 1))[np.newaxis, ...]
    if len(dims) == 4 and dims[-1] in (1, 3):
        mh, mw = dims[1], dims[2]
        if mh > 0 and mw > 0 and (mh, mw) != (img.shape[0], img.shape[1]):
            img = _stretch_hwc(img, mh, mw)
        return img[np.newaxis, ...]
    return np.transpose(img, (2, 0, 1))[np.newaxis, ...]


def _model_hw(shape: list[object], rgb_hw: tuple[int, int]) -> tuple[int, int]:
    dims = [int(x) if str(x).isdigit() or isinstance(x, (int, np.integer)) else -1 for x in shape]
    if len(dims) == 4 and dims[1] in (1, 3) and dims[2] > 0 and dims[3] > 0:
        return dims[2], dims[3]
    if len(dims) == 4 and dims[-1] in (1, 3) and dims[1] > 0 and dims[2] > 0:
        return dims[1], dims[2]
    return rgb_hw


def _stretch_hwc(img: np.ndarray, h: int, w: int) -> np.ndarray:
    from ugv_perception.backend.instances import _bilinear_float

    chans = [_bilinear_float(img[:, :, c], h, w).astype(np.float32) for c in range(img.shape[2])]
    return np.stack(chans, axis=2)


def _result_arrays(result: object) -> list[np.ndarray]:
    if hasattr(result, "values"):
        tensors = list(result.values())
    elif isinstance(result, (list, tuple)):
        tensors = list(result)
    else:
        raise AdapterError("unrecognized OpenVINO result type")
    return [np.asarray(t) for t in tensors]


def decode_yolo_seg(
    pred: np.ndarray,
    proto: np.ndarray,
    n_classes: int,
    model_hw: tuple[int, int],
    *,
    conf_thres: float = _CONF_THRES,
    iou_thres: float = _IOU_THRES,
    max_det: int = _MAX_DET,
) -> tuple[list[int], list[float], list[np.ndarray]]:
    """YOLO-seg IR: pred (4+nc+nm, N) + proto (nm, mh, mw) → instances.

    Numpy only. Ultralytics is not the runtime. Empty after NMS is legal.
    """
    if type(n_classes) is not int or n_classes <= 0:
        raise ValueError("n_classes must be a positive Python int")
    proto_chw = _squeeze_proto(proto)
    nm, mh, mw = proto_chw.shape
    matrix = _as_pred_matrix(pred, n_classes=n_classes, nm=nm)
    boxes_xywh = matrix[:, :4]
    cls = matrix[:, 4 : 4 + n_classes]
    coeffs = matrix[:, 4 + n_classes :]
    if np.any(~np.isfinite(cls)) or np.any(cls < 0.0) or np.any(cls > 1.0):
        raise AdapterError("YOLO-seg class scores are not finite and in [0,1]; no clip")
    scores = cls.max(axis=1)
    class_indices = cls.argmax(axis=1).astype(np.int64)
    keep = scores >= conf_thres
    if not np.any(keep):
        return [], [], []
    boxes_xywh = boxes_xywh[keep]
    scores = scores[keep]
    class_indices = class_indices[keep]
    coeffs = coeffs[keep]
    xyxy = _xywh_to_xyxy(boxes_xywh)
    nms_idx = _nms_per_class(xyxy, scores, class_indices, iou_thres=iou_thres, max_det=max_det)
    if nms_idx.size == 0:
        return [], [], []
    xyxy = xyxy[nms_idx]
    scores = scores[nms_idx]
    class_indices = class_indices[nms_idx]
    coeffs = coeffs[nms_idx]
    if np.any(~np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise AdapterError("YOLO-seg scores are not finite and in [0,1]; no clip")
    masks = _masks_from_proto(coeffs, proto_chw, xyxy, model_hw)
    return (
        [int(x) for x in class_indices.tolist()],
        [float(x) for x in scores.tolist()],
        [np.asarray(m, dtype=np.float32) for m in masks],
    )


def _parse_ov_result(
    result: object, n_classes: int, model_hw: tuple[int, int]
) -> tuple[list[int], list[float], list[np.ndarray]]:
    arrays = _result_arrays(result)
    if not arrays or all(a.size == 0 for a in arrays):
        return [], [], []
    pred = None
    proto = None
    for a in arrays:
        if a.ndim == 4:
            proto = a
        elif a.ndim == 3:
            pred = a
        elif a.ndim == 2:
            pred = a
    if pred is None or proto is None:
        raise AdapterError("OpenVINO IR outputs do not match YOLO-seg (pred, proto) layout")
    return decode_yolo_seg(pred, proto, n_classes, model_hw)


def _squeeze_proto(proto: np.ndarray) -> np.ndarray:
    a = np.asarray(proto)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 3 or a.shape[0] < 1 or a.shape[1] < 1 or a.shape[2] < 1:
        raise AdapterError("YOLO-seg proto must be (nm, mh, mw)")
    return a


def _as_pred_matrix(pred: np.ndarray, n_classes: int, nm: int) -> np.ndarray:
    a = np.asarray(pred)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise AdapterError("YOLO-seg pred must be 2-D after squeezing batch")
    width = 4 + n_classes + nm
    if a.shape[0] == width:
        a = a.T
    if a.shape[1] != width:
        raise AdapterError(
            f"YOLO-seg pred last dim must be 4+n_classes+nm={width}, got {a.shape}"
        )
    return a.astype(np.float32, copy=False)


def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    xy = xywh[:, :2]
    half = xywh[:, 2:] / 2.0
    return np.concatenate([xy - half, xy + half], axis=1)


def _iou_xyxy(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_b = np.maximum(0.0, others[:, 2] - others[:, 0]) * np.maximum(
        0.0, others[:, 3] - others[:, 1]
    )
    union = area_a + area_b - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0.0)


def _nms_one_class(xyxy: np.ndarray, scores: np.ndarray, iou_thres: float, max_det: int) -> np.ndarray:
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size > 0 and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = _iou_xyxy(xyxy[i], xyxy[rest])
        order = rest[ious <= iou_thres]
    return np.asarray(keep, dtype=np.intp)


def _nms_per_class(
    xyxy: np.ndarray,
    scores: np.ndarray,
    class_indices: np.ndarray,
    *,
    iou_thres: float,
    max_det: int,
) -> np.ndarray:
    kept: list[int] = []
    for cls in np.unique(class_indices):
        idx = np.flatnonzero(class_indices == cls)
        local = _nms_one_class(xyxy[idx], scores[idx], iou_thres=iou_thres, max_det=max_det)
        kept.extend(idx[local].tolist())
    if not kept:
        return np.zeros((0,), dtype=np.intp)
    kept_arr = np.asarray(kept, dtype=np.intp)
    order = np.argsort(scores[kept_arr])[::-1][:max_det]
    return kept_arr[order]


def _masks_from_proto(
    coeffs: np.ndarray,
    proto: np.ndarray,
    xyxy: np.ndarray,
    model_hw: tuple[int, int],
) -> np.ndarray:
    nm, mh, mw = proto.shape
    logits = coeffs @ proto.reshape(nm, -1)
    masks = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    masks = masks.reshape(-1, mh, mw).astype(np.float32, copy=False)
    model_h, model_w = model_hw
    sx = mw / float(model_w)
    sy = mh / float(model_h)
    x1 = xyxy[:, 0] * sx
    y1 = xyxy[:, 1] * sy
    x2 = xyxy[:, 2] * sx
    y2 = xyxy[:, 3] * sy
    cols = np.arange(mw, dtype=np.float32)[None, None, :]
    rows = np.arange(mh, dtype=np.float32)[None, :, None]
    x1b = x1[:, None, None]
    y1b = y1[:, None, None]
    x2b = x2[:, None, None]
    y2b = y2[:, None, None]
    inside = (cols >= x1b) & (cols < x2b) & (rows >= y1b) & (rows < y2b)
    masks = masks * inside.astype(np.float32)
    if np.any(~np.isfinite(masks)) or np.any(masks < 0.0) or np.any(masks > 1.0):
        raise AdapterError("YOLO-seg masks are not finite and in [0,1]; no clip")
    return masks


class OpenVinoGpuTensorBackend:
    """Compile an IR on GPU and return the first output tensor. No YOLO decode."""

    id = "openvino_gpu"

    def __init__(self) -> None:
        self._compiled = None

    def load(self, weights_path: str) -> None:
        path = Path(weights_path)
        if not path.is_file():
            raise FileNotFoundError(f"OpenVINO IR missing: {path}")
        try:
            import openvino as ov
        except ImportError as exc:
            raise AdapterError("openvino is not installed") from exc
        core = ov.Core()
        devices = list(core.available_devices)
        if not any(str(d).startswith(_DEVICE) for d in devices):
            raise AdapterError(f"OpenVINO {_DEVICE} not available; devices={devices}")
        try:
            model = core.read_model(str(path))
            self._compiled = core.compile_model(model, _DEVICE)
        except Exception as exc:
            raise AdapterError(f"OpenVINO compile on {_DEVICE} failed") from exc

    def run(self, blob: NDArray[np.float32]) -> np.ndarray:
        if self._compiled is None:
            raise AdapterError("OpenVinoGpuTensorBackend.load() was not called")
        if not isinstance(blob, np.ndarray) or blob.dtype != np.float32 or blob.ndim != 4:
            raise TypeError("blob must be float32 NCHW")
        try:
            result = self._compiled([blob])
            out = np.asarray(result[self._compiled.output(0)])
        except Exception as exc:
            raise AdapterError("OpenVINO GPU run failed") from exc
        return out
