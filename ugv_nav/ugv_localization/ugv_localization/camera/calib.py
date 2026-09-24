"""Camera calibration in ROS camera_info_manager YAML format.

Refuses uncalibrated / fake intrinsics (all-zero K, principal point outside the image).
Mono RTAB-Map with a lying K produces confident wrong geometry, so we fail closed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from ugv_localization.common.yamlio import load_yaml_mapping

# distortion_model -> required coefficient count
DISTORTION_COEFFS: dict[str, int] = {
    "plumb_bob": 5,
    "rational_polynomial": 8,
    "equidistant": 4,
}

_REQUIRED_KEYS = (
    "image_width",
    "image_height",
    "camera_name",
    "camera_matrix",
    "distortion_model",
    "distortion_coefficients",
    "rectification_matrix",
    "projection_matrix",
)

_EPS = 1e-9


class CalibrationError(ValueError):
    """Calibration is missing, malformed, or not a real calibration."""


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    camera_name: str
    width: int
    height: int
    k: tuple[float, ...]  # 9, row-major
    distortion_model: str
    d: tuple[float, ...]
    r: tuple[float, ...]  # 9
    p: tuple[float, ...]  # 12


def _floats(values: object, *, name: str) -> tuple[float, ...]:
    try:
        out = tuple(float(v) for v in values)  # type: ignore[union-attr]
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{name} must be a list of numbers") from exc
    if not all(math.isfinite(v) for v in out):
        raise CalibrationError(f"{name} values must be finite")
    return out


def validate_intrinsics(width: int, height: int, k: tuple[float, ...]) -> None:
    if type(width) is not int or width <= 0:
        raise CalibrationError(f"image width must be int > 0, got {width!r}")
    if type(height) is not int or height <= 0:
        raise CalibrationError(f"image height must be int > 0, got {height!r}")
    k = tuple(k)
    if len(k) != 9:
        raise CalibrationError(f"K must have 9 values, got {len(k)}")
    if not all(math.isfinite(float(v)) for v in k):
        raise CalibrationError("K values must be finite")
    fx, _, cx, k3, fy, cy, k6, k7, k8 = k
    if fx <= 0.0:
        raise CalibrationError(f"K fx must be > 0 (uncalibrated?), got {fx}")
    if fy <= 0.0:
        raise CalibrationError(f"K fy must be > 0, got {fy}")
    if not 0.0 < cx < width:
        raise CalibrationError(f"K cx={cx} outside image width {width}")
    if not 0.0 < cy < height:
        raise CalibrationError(f"K cy={cy} outside image height {height}")
    if abs(k3) > _EPS or abs(k6) > _EPS or abs(k7) > _EPS or abs(k8 - 1.0) > _EPS:
        raise CalibrationError("K bottom row must be [0 0 1] and k[3] must be 0")


def _validate_distortion(model: str, d: tuple[float, ...]) -> None:
    if model not in DISTORTION_COEFFS:
        raise CalibrationError(
            f"distortion_model {model!r} unsupported; expected one of {sorted(DISTORTION_COEFFS)}"
        )
    need = DISTORTION_COEFFS[model]
    if len(d) != need:
        raise CalibrationError(f"{model} needs {need} distortion coefficients, got {len(d)}")


def _validate_rp(r: tuple[float, ...], p: tuple[float, ...]) -> None:
    if len(r) != 9:
        raise CalibrationError(f"rectification matrix must have 9 values, got {len(r)}")
    if len(p) != 12:
        raise CalibrationError(f"projection matrix must have 12 values, got {len(p)}")
    if p[0] <= 0.0 or p[5] <= 0.0 or abs(p[10] - 1.0) > _EPS:
        raise CalibrationError("projection matrix fx'/fy' must be > 0 and P[2,2] must be 1")


def _matrix(data: dict, key: str, rows: int, cols: int) -> tuple[float, ...]:
    block = data[key]
    if not isinstance(block, dict) or set(block) != {"rows", "cols", "data"}:
        raise CalibrationError(f"{key} must be a mapping with rows, cols, data")
    if block["rows"] != rows or (cols >= 0 and block["cols"] != cols):
        raise CalibrationError(f"{key} must be {rows}x{cols}, got {block['rows']}x{block['cols']}")
    values = _floats(block["data"], name=key)
    if len(values) != block["rows"] * block["cols"]:
        raise CalibrationError(f"{key}: data length {len(values)} != rows*cols")
    return values


def calibration_from_camera_info(
    *,
    camera_name: str,
    width: int,
    height: int,
    distortion_model: str,
    d: tuple[float, ...] | list[float],
    k: tuple[float, ...] | list[float],
    r: tuple[float, ...] | list[float],
    p: tuple[float, ...] | list[float],
) -> CameraCalibration:
    """Build from a real CameraInfo message's fields. Empty D (ideal sim camera) → zeros."""
    if not camera_name:
        raise CalibrationError("camera_name must be non-empty")
    k_t = _floats(k, name="K")
    validate_intrinsics(width, height, k_t)
    d_t = _floats(d, name="D")
    if len(d_t) == 0 and distortion_model in DISTORTION_COEFFS:
        d_t = (0.0,) * DISTORTION_COEFFS[distortion_model]
    _validate_distortion(distortion_model, d_t)
    r_t = _floats(r, name="R")
    p_t = _floats(p, name="P")
    _validate_rp(r_t, p_t)
    return CameraCalibration(
        camera_name=camera_name,
        width=width,
        height=height,
        k=k_t,
        distortion_model=distortion_model,
        d=d_t,
        r=r_t,
        p=p_t,
    )


def load_calibration(path: str | Path) -> CameraCalibration:
    data = load_yaml_mapping(path)
    for key in _REQUIRED_KEYS:
        if key not in data:
            raise CalibrationError(f"{path}: missing key {key}")
    model = data["distortion_model"]
    if not isinstance(model, str):
        raise CalibrationError("distortion_model must be a string")
    k = _matrix(data, "camera_matrix", 3, 3)
    d = _matrix(data, "distortion_coefficients", 1, -1)
    r = _matrix(data, "rectification_matrix", 3, 3)
    p = _matrix(data, "projection_matrix", 3, 4)
    validate_intrinsics(data["image_width"], data["image_height"], k)
    _validate_distortion(model, d)
    _validate_rp(r, p)
    return CameraCalibration(
        camera_name=str(data["camera_name"]),
        width=data["image_width"],
        height=data["image_height"],
        k=k,
        distortion_model=model,
        d=d,
        r=r,
        p=p,
    )


def calibration_to_yaml_dict(cal: CameraCalibration) -> dict:
    return {
        "image_width": cal.width,
        "image_height": cal.height,
        "camera_name": cal.camera_name,
        "camera_matrix": {"rows": 3, "cols": 3, "data": list(cal.k)},
        "distortion_model": cal.distortion_model,
        "distortion_coefficients": {"rows": 1, "cols": len(cal.d), "data": list(cal.d)},
        "rectification_matrix": {"rows": 3, "cols": 3, "data": list(cal.r)},
        "projection_matrix": {"rows": 3, "cols": 4, "data": list(cal.p)},
    }
