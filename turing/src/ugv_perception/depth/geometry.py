"""DA3METRIC-LARGE geometry. Pure numpy. No vendor runtime."""

from __future__ import annotations

import numpy as np

PROCESS_RES = 504
PATCH = 14
SKY_THRESHOLD = 0.3
VALID_COVERAGE = 0.5
METRIC_SCALE = 300.0
MAX_AGE_S = 0.50
_NS = 1_000_000_000


def nearest_multiple(value: int, patch: int = PATCH) -> int:
    down = (value // patch) * patch
    up = down + patch
    picked = up if abs(up - value) <= abs(value - down) else down
    return max(1, picked)


def two_step_hw(height: int, width: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Longest side → 504, then each side to the nearest multiple of 14."""
    if height < 1 or width < 1:
        raise ValueError("height and width must be > 0")
    longest = max(height, width)
    if longest == PROCESS_RES:
        first = (height, width)
    else:
        scale = PROCESS_RES / float(longest)
        first = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))
    second = (nearest_multiple(first[0]), nearest_multiple(first[1]))
    return first, second


def model_hw(height: int, width: int) -> tuple[int, int]:
    return two_step_hw(height, width)[1]


def scale_k(k: np.ndarray, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> np.ndarray:
    """Scale fx,cx by width ratio and fy,cy by height ratio."""
    out = np.array(k, dtype=np.float64, copy=True).reshape(3, 3)
    sh, sw = src_hw
    dh, dw = dst_hw
    out[0, :] *= dw / float(sw)
    out[1, :] *= dh / float(sh)
    return out


def k_model(k_camera: np.ndarray, camera_hw: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    """Two resize steps, intrinsics updated after each, matching DA3."""
    height, width = camera_hw
    first, second = two_step_hw(height, width)
    k1 = scale_k(k_camera, (height, width), first)
    k2 = scale_k(k1, first, second)
    return k2, second


def focal_model(k: np.ndarray) -> float:
    mat = np.asarray(k, dtype=np.float64).reshape(3, 3)
    return float((mat[0, 0] + mat[1, 1]) / 2.0)


def meters_from_raw(raw: np.ndarray, focal: float) -> np.ndarray:
    return np.asarray(raw, dtype=np.float64) * (float(focal) / METRIC_SCALE)


def valid_mask(depth_m: np.ndarray, sky: np.ndarray) -> np.ndarray:
    sky_a = np.asarray(sky)
    if sky_a.dtype == np.bool_:
        hole = sky_a
    else:
        hole = np.asarray(sky_a, dtype=np.float64) >= SKY_THRESHOLD
    return np.isfinite(depth_m) & ~hole


def half_pixel_positions(n_src: int, n_dst: int) -> np.ndarray:
    """Source sample position of each of `n_dst` output pixels along one axis: half-pixel centres, edge clamp.

    Float64, in [0, n_src - 1]. The one definition: the GPU twin in geometry_gpu builds its index and
    weight tables from this, so the two resizes cannot drift apart.
    """
    return np.clip((np.arange(n_dst) + 0.5) * n_src / n_dst - 0.5, 0.0, n_src - 1)


def _bilinear(src: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize (half-pixel centres, edge clamp), float64. Separable: rows, then columns."""
    src_f = np.asarray(src, dtype=np.float64)
    mh, mw = src_f.shape
    if (mh, mw) == (out_h, out_w):
        return src_f.copy()
    ys = half_pixel_positions(mh, out_h)
    xs = half_pixel_positions(mw, out_w)
    y0 = np.floor(ys).astype(np.intp)
    x0 = np.floor(xs).astype(np.intp)
    y1 = np.minimum(y0 + 1, mh - 1)
    x1 = np.minimum(x0 + 1, mw - 1)
    wy = (ys - y0)[:, None]
    wx = xs - x0
    rows = src_f[y0, :] * (1.0 - wy) + src_f[y1, :] * wy
    return rows[:, x0] * (1.0 - wx) + rows[:, x1] * wx


def hole_safe_resize(depth_m: np.ndarray, sky: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float64)
    valid = valid_mask(depth, sky).astype(np.float64)
    filled = np.where(valid > 0.0, depth, 0.0)
    weighted = _bilinear(filled * valid, out_hw[0], out_hw[1])
    weight = _bilinear(valid, out_hw[0], out_hw[1])
    out = np.full(out_hw, np.nan, dtype=np.float64)
    keep = weight >= VALID_COVERAGE
    out[keep] = weighted[keep] / weight[keep]
    return out


def backproject(depth_hw: np.ndarray, k_camera: np.ndarray) -> np.ndarray:
    """Unorganized XYZ. Holes omitted. Optical frame, meters."""
    depth = np.asarray(depth_hw, dtype=np.float64)
    mat = np.asarray(k_camera, dtype=np.float64).reshape(3, 3)
    fx, fy = float(mat[0, 0]), float(mat[1, 1])
    cx, cy = float(mat[0, 2]), float(mat[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("fx and fy must be > 0")
    good = np.isfinite(depth) & (depth > 0.0)
    vs, us = np.nonzero(good)
    z = depth[vs, us]
    x = (us.astype(np.float64) - cx) * z / fx
    y = (vs.astype(np.float64) - cy) * z / fy
    return np.stack([x, y, z], axis=1).astype(np.float32)


# ImageNet normalisation of DA3's input. The one definition: geometry_gpu imports these.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MEAN, STD = IMAGENET_MEAN, IMAGENET_STD  # the names base code (run_*_from_rgb callers) uses
_MEAN = np.array(IMAGENET_MEAN, dtype=np.float64)
_STD = np.array(IMAGENET_STD, dtype=np.float64)


def _resize_u8(rgb: np.ndarray, dst_hw: tuple[int, int]) -> np.ndarray:
    if (int(rgb.shape[0]), int(rgb.shape[1])) == dst_hw:
        return rgb
    planes = [
        _bilinear(rgb[:, :, c].astype(np.float64), dst_hw[0], dst_hw[1]) for c in range(3)
    ]
    return np.clip(np.stack(planes, axis=2), 0.0, 255.0).astype(np.uint8)


def preprocess_nchw(rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Two-step official resize, ImageNet norm, NCHW. Same sizes as k_model."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise TypeError("rgb must be uint8 HWC")
    first, second = two_step_hw(int(rgb.shape[0]), int(rgb.shape[1]))
    stepped = _resize_u8(rgb, first)
    stepped = _resize_u8(stepped, second)
    arr = stepped.astype(np.float64) / 255.0
    arr = (arr - _MEAN) / _STD
    blob = np.transpose(arr, (2, 0, 1))[None].astype(np.float32)
    return blob, second


def geometry_is_fresh(now_ns: int, stamp_ns: int, max_age_s: float = MAX_AGE_S) -> bool:
    """False means ignore the cloud. It does not mean free space."""
    if now_ns < stamp_ns:
        return False
    return (now_ns - stamp_ns) / _NS <= max_age_s
