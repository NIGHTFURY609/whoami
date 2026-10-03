"""Torch twins of the DA3 pre/post-processing in geometry.py. Same maths, float32, on any torch device.

geometry.py stays the numpy reference and the path for OpenVINO and CPU. These run on the device that
holds the model, so the camera frame goes up once and only the camera-sized depth comes back. Importing
this module imports torch; nothing outside the CUDA backend should import it at module level.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from ugv_perception.depth.geometry import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SKY_THRESHOLD,
    VALID_COVERAGE,
    half_pixel_positions,
    two_step_hw,
)


@lru_cache(maxsize=32)
def _axis(n_src: int, n_dst: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample positions along one axis as (i0, i1, weight of i1). Tables built in float64 like geometry._bilinear."""
    pos = half_pixel_positions(n_src, n_dst)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, n_src - 1)
    kwargs = {"non_blocking": True} if str(device).startswith("cuda") else {}
    return (
        torch.from_numpy(i0).to(device, **kwargs),
        torch.from_numpy(i1).to(device, **kwargs),
        torch.from_numpy((pos - i0).astype(np.float32)).to(device, **kwargs),
    )


def _bilinear(src: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """Bilinear resize of dims 0 and 1 (half-pixel centres, edge clamp). Extra trailing dims ride along."""
    mh, mw = int(src.shape[0]), int(src.shape[1])
    if (mh, mw) == (out_h, out_w):
        return src
    device = str(src.device)
    y0, y1, wy = _axis(mh, out_h, device)
    x0, x1, wx = _axis(mw, out_w, device)
    tail = (1,) * (src.dim() - 2)
    wy = wy.view(-1, 1, *tail)
    wx = wx.view(1, -1, *tail)
    rows = src[y0] * (1.0 - wy) + src[y1] * wy
    return rows[:, x0] * (1.0 - wx) + rows[:, x1] * wx


def _resize_u8(rgb: torch.Tensor, dst_hw: tuple[int, int]) -> torch.Tensor:
    """uint8 HWC to uint8 HWC. Truncates back to 8 bits like geometry._resize_u8 does."""
    if (int(rgb.shape[0]), int(rgb.shape[1])) == dst_hw:
        return rgb
    return _bilinear(rgb.float(), dst_hw[0], dst_hw[1]).clamp_(0.0, 255.0).to(torch.uint8)


def preprocess_nchw_gpu(rgb: np.ndarray, device: str) -> tuple[torch.Tensor, tuple[int, int]]:
    """geometry.preprocess_nchw on `device`: two-step official resize, ImageNet norm, NCHW float32."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise TypeError("rgb must be uint8 HWC")
    first, second = two_step_hw(int(rgb.shape[0]), int(rgb.shape[1]))
    kwargs = {"non_blocking": True} if str(device).startswith("cuda") else {}
    image = torch.from_numpy(np.ascontiguousarray(rgb)).to(device, **kwargs)
    image = _resize_u8(_resize_u8(image, first), second)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(3, 1, 1)
    chw = image.permute(2, 0, 1).float() / 255.0
    return ((chw - mean) / std).unsqueeze(0).contiguous(), second


def hole_safe_resize_gpu(
    depth_m: torch.Tensor, sky: torch.Tensor, out_hw: tuple[int, int]
) -> torch.Tensor:
    """geometry.hole_safe_resize on the device of `depth_m`: float32, NaN where under half the area is valid."""
    depth = depth_m.float()
    if sky.dtype == torch.bool:
        hole = sky
    else:
        hole = sky.float() >= SKY_THRESHOLD
    valid = torch.isfinite(depth) & ~hole
    valid_f = valid.float()
    filled = torch.where(valid, depth, torch.zeros_like(depth))
    weighted = _bilinear(filled * valid_f, out_hw[0], out_hw[1])
    weight = _bilinear(valid_f, out_hw[0], out_hw[1])
    keep = weight >= VALID_COVERAGE
    return torch.where(keep, weighted / weight, torch.full_like(weighted, float("nan")))
