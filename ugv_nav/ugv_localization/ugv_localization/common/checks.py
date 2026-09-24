"""Stamp, float and angle checks. Stamps are Python int nanoseconds > 0 (same law as Dev 1)."""

from __future__ import annotations

import math

NS_PER_S = 1_000_000_000


def require_stamp(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python int > 0, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def require_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a Python bool, got {type(value).__name__}")
    return value


def require_finite(value: object, *, name: str) -> float:
    if type(value) is bool or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out}")
    return out


def require_positive_float(value: object, *, name: str, allow_zero: bool = False) -> float:
    """YAML thresholds must be written as floats (0.5, not 1) so intent is explicit."""
    if type(value) is not float:
        raise TypeError(f"{name} must be a float, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if value < 0.0 or (value == 0.0 and not allow_zero):
        bound = ">= 0.0" if allow_zero else "> 0.0"
        raise ValueError(f"{name} must be {bound}, got {value}")
    return value


def age_s(stamp_ns: int, now_ns: int) -> float:
    return (now_ns - stamp_ns) / NS_PER_S


def wrap_angle(rad: float) -> float:
    """Wrap to (-pi, pi]."""
    out = math.fmod(rad + math.pi, 2.0 * math.pi)
    if out <= 0.0:
        out += 2.0 * math.pi
    return out - math.pi


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
