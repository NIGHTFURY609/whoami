"""Startup load of DA3. Absent IR and safetensors means T08 stays off."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ugv_perception.adapter.output import AdapterError
from ugv_perception.backend.device import pick_tensor_backend
from ugv_perception.depth.geometry import (
    MEAN,
    STD,
    backproject,
    focal_model,
    hole_safe_resize,
    k_model,
    meters_from_raw,
    model_hw,
    preprocess_nchw,
    two_step_hw,
)


class DepthChannel:
    def __init__(self, backend: object) -> None:
        self._backend = backend
        self._metres_disabled = False  # sticky, like the backends' rgb_pre_disabled

    def maps(self, rgb: np.ndarray, k: tuple[float, ...] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Camera-sized meters (NaN holes) and unorganized XYZ. One infer, FP32 on every backend.

        CUDA: `run_depth_metres` does the preprocess and the meters/resize on the device (geometry_gpu). If it
        fails once, this channel stops asking for it and takes the path below for good.
        Every backend with `run_all_from_rgb` (OpenVINO GPU on Arc, and CUDA as the fallback) preprocesses the
        RGB on the device; the numpy preprocess is the last resort.
        """
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise TypeError("rgb must be uint8 HWC")
        k_cam = np.asarray(k, dtype=np.float64).reshape(3, 3)
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        k_m, (mh, mw) = k_model(k_cam, (height, width))
        if model_hw(height, width) != (mh, mw):
            raise AdapterError("model size disagrees with K_model")
        on_device = getattr(self._backend, "run_depth_metres", None)
        if callable(on_device) and not self._metres_disabled:
            try:
                on_camera = on_device(rgb, focal_model(k_m), (mh, mw), (height, width))
                return on_camera, backproject(on_camera, k_cam)
            except AdapterError:
                self._metres_disabled = True
        self._backend.ensure_hw(mh, mw)
        first, second = two_step_hw(height, width)
        if second != (mh, mw):
            raise AdapterError("preprocess size disagrees with K_model")
        run_from_rgb = getattr(self._backend, "run_all_from_rgb", None)
        outputs = None
        if callable(run_from_rgb) and not getattr(self._backend, "rgb_pre_disabled", False):
            try:
                outputs = run_from_rgb(rgb, (first, second), MEAN, STD)
            except AdapterError:
                if not getattr(self._backend, "rgb_pre_disabled", False):
                    raise
        if outputs is None:
            blob, sized = preprocess_nchw(rgb)
            if sized != (mh, mw):
                raise AdapterError("preprocess size disagrees with K_model")
            outputs = self._backend.run_all(blob)
        if len(outputs) < 2:
            raise AdapterError("DA3 must return depth_raw and sky")
        raw = np.squeeze(outputs[0])
        sky = np.squeeze(outputs[1])
        depth_m = meters_from_raw(raw, focal_model(k_m))
        on_camera = hole_safe_resize(depth_m, sky, (height, width)).astype(np.float32)
        return on_camera, backproject(on_camera, k_cam)

    def points(self, rgb: np.ndarray, k: tuple[float, ...] | np.ndarray) -> np.ndarray:
        return self.maps(rgb, k)[1]


def build_depth_channel(root: Path) -> DepthChannel | None:
    xml = root / "weights" / "da3metric-large.xml"
    folder = root / "weights" / "da3metric-large"
    backend = pick_tensor_backend(ir_xml=xml, safetensors_dir=folder, kind="da3")
    if backend is None:
        return None
    return DepthChannel(backend)
