"""OpenVINO 2026.4.0 GPU backend. Import openvino only inside load/run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ugv_perception.adapter.output import AdapterError, Instance
from ugv_perception.backend.instances import instances_from_engine

_DEVICE = "GPU"
_CPU = "CPU"
_SHARED_CORE = None
# Engine NMS (Ultralytics YOLO-seg defaults). Not T04 τ.
_CONF_THRES = 0.25
_IOU_THRES = 0.70
_MAX_DET = 300


def _shared_core() -> object:
    """One Core for every tensor model so two GPU InferRequests can overlap."""
    global _SHARED_CORE
    import openvino as ov

    if _SHARED_CORE is None:
        _SHARED_CORE = ov.Core()
    return _SHARED_CORE


def _compile_gpu_then_cpu(
    core: object, model: object, config: dict | None = None
) -> tuple[object, str]:
    """Prefer OpenVINO GPU (Intel Arc or NVIDIA plugin). CPU is last. Compile once."""
    devices = [str(d) for d in core.available_devices]
    names: list[str] = []
    for prefix in (_DEVICE, "NVIDIA", "CUDA", _CPU):
        for item in devices:
            if item.startswith(prefix) and item not in names:
                names.append(item)
    if not names:
        raise AdapterError(f"OpenVINO has no GPU or CPU; devices={devices}")
    last: Exception | None = None
    for name in names:
        try:
            if config is None:
                return core.compile_model(model, name), name
            return core.compile_model(model, name, config), name
        except Exception as exc:
            last = exc
            continue
    raise AdapterError("OpenVINO compile failed on GPU and CPU") from last


class OpenVinoGpuBackend:
    id = "openvino_gpu"

    def __init__(self) -> None:
        self._compiled = None
        self._prompts: tuple[str, ...] | None = None
        self.device = _DEVICE

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
        try:
            model = core.read_model(str(path))
            self._compiled, self.device = _compile_gpu_then_cpu(core, model)
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError("OpenVINO compile failed") from exc
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


def _compile_seg_post(
    core: object,
    device: str,
    channels: int,
    in_h: int,
    in_w: int,
    out_h: int,
    out_w: int,
) -> object:
    """Bilinear upsample + softmax + argmax on the same OpenVINO device as the net."""
    from openvino import Model, Type, properties
    from openvino import opset13 as ops

    logits_p = ops.parameter([1, channels, in_h, in_w], Type.f32, name="logits")
    interp = ops.interpolate(
        logits_p,
        np.array([out_h, out_w], dtype=np.int64),
        mode="linear",
        shape_calculation_mode="sizes",
        coordinate_transformation_mode="half_pixel",
        axes=np.array([2, 3], dtype=np.int64),
    )
    prob = ops.softmax(interp, axis=1)
    topk = ops.topk(
        prob, np.int32(1), axis=1, mode="max", sort="none", index_element_type="i32"
    )
    axes = np.array([1], dtype=np.int64)
    labels = ops.squeeze(topk.output(1), axes)
    scores = ops.squeeze(topk.output(0), axes)
    model = Model([labels, scores], [logits_p], "rugd_seg_post")
    try:
        return core.compile_model(
            model, device, {properties.hint.inference_precision: Type.f32}
        )
    except AdapterError:
        raise
    except Exception as exc:
        raise AdapterError("OpenVINO seg-post compile failed") from exc


def _compile_rgb_pre(
    core: object,
    device: str,
    in_h: int,
    in_w: int,
    sizes: tuple[tuple[int, int], ...],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    clip_255: bool,
) -> object:
    """u8 NHWC → f32 NCHW bilinear to each size → ImageNet. Pin f32."""
    from openvino import Model, Type, properties
    from openvino import opset13 as ops

    if not sizes:
        raise ValueError("rgb pre needs at least one target size")
    rgb_p = ops.parameter([1, in_h, in_w, 3], Type.u8, name="rgb")
    x = ops.convert(rgb_p, Type.f32)
    x = ops.transpose(x, np.array([0, 3, 1, 2], dtype=np.int64))
    cur_h, cur_w = int(in_h), int(in_w)
    for oh, ow in sizes:
        th, tw = int(oh), int(ow)
        if (th, tw) != (cur_h, cur_w):
            x = ops.interpolate(
                x,
                np.array([th, tw], dtype=np.int64),
                mode="linear",
                shape_calculation_mode="sizes",
                coordinate_transformation_mode="half_pixel",
                axes=np.array([2, 3], dtype=np.int64),
            )
            if clip_255:
                x = ops.clamp(x, 0.0, 255.0)
            cur_h, cur_w = th, tw
    x = ops.multiply(x, np.array(1.0 / 255.0, dtype=np.float32))
    mean_a = np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std_a = np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1)
    x = ops.subtract(x, mean_a)
    x = ops.divide(x, std_a)
    model = Model([x], [rgb_p], "rgb_pre")
    try:
        return core.compile_model(
            model, device, {properties.hint.inference_precision: Type.f32}
        )
    except AdapterError:
        raise
    except Exception as exc:
        raise AdapterError("OpenVINO rgb-pre compile failed") from exc


class OpenVinoGpuTensorBackend:
    """Compile an IR on GPU and return the first output tensor. No YOLO decode."""

    id = "openvino_gpu"

    def __init__(self) -> None:
        self._compiled = None
        self._core = None
        self._model = None
        self._hw: tuple[int, int] | None = None
        self._post = None
        self._post_key: tuple | None = None
        self._pre = None
        self._pre_key: tuple | None = None
        self._fallback_logits: np.ndarray | None = None
        self._req = None
        self._in_flight = False
        self._in_flight_input: np.ndarray | None = None
        self._primed_blob: np.ndarray | None = None
        self.seg_post_disabled = False
        self.rgb_pre_disabled = False
        self.device = _DEVICE

    def load(self, weights_path: str, input_hw: tuple[int, int] | None = None) -> None:
        path = Path(weights_path)
        if not path.is_file():
            raise FileNotFoundError(f"OpenVINO IR missing: {path}")
        try:
            import openvino as ov
        except ImportError as exc:
            raise AdapterError("openvino is not installed") from exc
        core = _shared_core()
        try:
            model = core.read_model(str(path))
        except Exception as exc:
            raise AdapterError(f"OpenVINO read failed for {path}") from exc
        self._core = core
        self._model = model
        if input_hw is not None:
            self.ensure_hw(input_hw[0], input_hw[1])
        else:
            self._compile()

    def ensure_hw(self, height: int, width: int) -> None:
        """Compile once for this model size. A second size is refused."""
        if self._model is None or self._core is None:
            raise AdapterError("OpenVinoGpuTensorBackend.load() was not called")
        if self._compiled is not None:
            if self._hw != (height, width):
                raise AdapterError(
                    f"DA3 IR already compiled for {self._hw}, not {(height, width)}"
                )
            return
        try:
            self._model.reshape([1, 3, height, width])
            self._compile()
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError("OpenVINO compile failed") from exc
        self._hw = (height, width)

    def _compile(self) -> None:
        try:
            # Same compile as the GPU-pre + GPU-decode build: FP32 IR, plugin-default runtime.
            # Pre and post graphs stay pinned f32. Do not set INFERENCE_PRECISION_HINT=f32 on the net.
            self._compiled, self.device = _compile_gpu_then_cpu(self._core, self._model)
            self._req = None
            self._in_flight = False
            self._in_flight_input = None
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError("OpenVINO compile failed") from exc
        partial = self._model.input(0).get_partial_shape()
        if not partial.is_dynamic:
            dims = [dim.get_length() for dim in partial]
            if len(dims) == 4:
                self._hw = (int(dims[2]), int(dims[3]))

    def run(self, blob: NDArray[np.float32]) -> np.ndarray:
        if self._fallback_logits is not None:
            out = self._fallback_logits
            self._fallback_logits = None
            return out
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

    def run_seg(
        self, blob: NDArray[np.float32], out_hw: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Logits on the compiled device, then interpolate+softmax on that same device."""
        logits = self.run(blob)
        try:
            return self._decode_seg(logits, out_hw)
        except AdapterError as exc:
            if "not finite" in str(exc):
                raise
            self.seg_post_disabled = True
            self._fallback_logits = np.asarray(logits, dtype=np.float32)
            raise

    def _decode_seg(
        self, logits: np.ndarray, out_hw: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._core is None:
            raise AdapterError("OpenVinoGpuTensorBackend.load() was not called")
        a = np.asarray(logits, dtype=np.float32)
        if a.ndim == 3:
            a = a[None, ...]
        if a.ndim != 4 or a.shape[0] != 1:
            raise AdapterError("RUGD logits for GPU decode must be (1, C, h, w)")
        if not np.isfinite(a).all():
            raise AdapterError("RUGD logits are not finite")
        _n, channels, lh, lw = a.shape
        oh, ow = int(out_hw[0]), int(out_hw[1])
        key = (str(self.device), channels, lh, lw, oh, ow)
        if self._post is None or self._post_key != key:
            try:
                self._post = _compile_seg_post(
                    self._core, str(self.device), channels, lh, lw, oh, ow
                )
                self._post_key = key
            except AdapterError:
                self.seg_post_disabled = True
                raise
        try:
            result = self._post([a])
            labels = np.squeeze(np.asarray(result[self._post.output(0)])).astype(np.int32)
            scores = np.squeeze(np.asarray(result[self._post.output(1)])).astype(np.float32)
        except AdapterError:
            self.seg_post_disabled = True
            raise
        except Exception as exc:
            self.seg_post_disabled = True
            raise AdapterError("OpenVINO seg decode failed") from exc
        return labels, scores

    def run_all(self, blob: NDArray[np.float32]) -> list[np.ndarray]:
        if self._compiled is None:
            raise AdapterError("OpenVinoGpuTensorBackend.load() was not called")
        if not isinstance(blob, np.ndarray) or blob.dtype != np.float32 or blob.ndim != 4:
            raise TypeError("blob must be float32 NCHW")
        try:
            result = self._compiled([blob])
            return [np.asarray(result[out]) for out in self._compiled.outputs]
        except Exception as exc:
            raise AdapterError("OpenVINO GPU run failed") from exc

    def begin_run_all(self, blob: NDArray[np.float32]) -> None:
        """Queue this model. wait_run_all() copies the outputs. One request per compiled model."""
        if self._compiled is None:
            raise AdapterError("OpenVinoGpuTensorBackend.load() was not called")
        if not isinstance(blob, np.ndarray) or blob.dtype != np.float32 or blob.ndim != 4:
            raise TypeError("blob must be float32 NCHW")
        if self._in_flight:
            raise AdapterError("OpenVINO request already in flight")
        try:
            if self._req is None:
                self._req = self._compiled.create_infer_request()
            self._in_flight_input = np.ascontiguousarray(blob)
            self._req.start_async([self._in_flight_input])
            self._in_flight = True
        except AdapterError:
            self._in_flight = False
            self._in_flight_input = None
            raise
        except Exception as exc:
            self._in_flight = False
            self._in_flight_input = None
            raise AdapterError("OpenVINO GPU run failed") from exc

    def wait_run_all(self) -> list[np.ndarray]:
        if self._req is None or not self._in_flight:
            raise AdapterError("OpenVINO GPU run was not started")
        try:
            self._req.wait()
            return [
                np.array(self._req.get_output_tensor(i).data, copy=True)
                for i in range(len(self._compiled.outputs))
            ]
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError("OpenVINO GPU run failed") from exc
        finally:
            self._in_flight = False
            self._in_flight_input = None

    def begin_run_all_from_rgb(
        self,
        rgb: NDArray[np.uint8],
        sizes: tuple[tuple[int, int], ...],
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        blob = self._rgb_pre(rgb, sizes, mean, std, clip_255=True)
        self.begin_run_all(blob)

    def prime_seg_rgb(
        self,
        rgb: NDArray[np.uint8],
        input_hw: tuple[int, int],
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        """Fill the RUGD input before either net is started."""
        self._primed_blob = self._rgb_pre(rgb, (input_hw,), mean, std, clip_255=False)

    def run_seg_from_rgb(
        self,
        rgb: NDArray[np.uint8],
        out_hw: tuple[int, int],
        input_hw: tuple[int, int],
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._primed_blob is not None:
            blob = self._primed_blob
            self._primed_blob = None
        else:
            blob = self._rgb_pre(rgb, (input_hw,), mean, std, clip_255=False)
        return self.run_seg(blob, out_hw)

    def run_all_from_rgb(
        self,
        rgb: NDArray[np.uint8],
        sizes: tuple[tuple[int, int], ...],
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> list[np.ndarray]:
        blob = self._rgb_pre(rgb, sizes, mean, std, clip_255=True)
        return self.run_all(blob)

    def _rgb_pre(
        self,
        rgb: np.ndarray,
        sizes: tuple[tuple[int, int], ...],
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        clip_255: bool,
    ) -> np.ndarray:
        if self.rgb_pre_disabled:
            raise AdapterError("OpenVINO rgb pre is disabled")
        if self._core is None:
            raise AdapterError("OpenVinoGpuTensorBackend.load() was not called")
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise TypeError("rgb must be uint8 HWC")
        in_h, in_w = int(rgb.shape[0]), int(rgb.shape[1])
        steps = tuple((int(h), int(w)) for h, w in sizes)
        key = (
            str(self.device),
            in_h,
            in_w,
            steps,
            tuple(float(x) for x in mean),
            tuple(float(x) for x in std),
            bool(clip_255),
        )
        if self._pre is None or self._pre_key != key:
            try:
                self._pre = _compile_rgb_pre(
                    self._core, str(self.device), in_h, in_w, steps, mean, std, clip_255
                )
                self._pre_key = key
            except AdapterError:
                self.rgb_pre_disabled = True
                raise
            except Exception as exc:
                self.rgb_pre_disabled = True
                raise AdapterError("OpenVINO rgb-pre compile failed") from exc
        try:
            batched = np.ascontiguousarray(rgb[None, ...])
            result = self._pre([batched])
            blob = np.asarray(result[self._pre.output(0)], dtype=np.float32)
        except AdapterError:
            self.rgb_pre_disabled = True
            raise
        except Exception as exc:
            self.rgb_pre_disabled = True
            raise AdapterError("OpenVINO rgb pre failed") from exc
        if blob.ndim != 4 or blob.shape[0] != 1 or blob.shape[1] != 3:
            self.rgb_pre_disabled = True
            raise AdapterError("OpenVINO rgb pre must return NCHW")
        return blob
