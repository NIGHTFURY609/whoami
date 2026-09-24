"""Wheel odometry gate → odom->base_link TF edge.

Mono + wheel odom (mindmap D5/D6): Dev 5 publishes /wheel/odom without TF; Dev 2 owns the
odom->base_link edge. This kernel decides whether a sample may become that edge.
Never restamps, never fabricates, never smooths — a bad sample is dropped, not repaired
(except renormalizing a quaternion that is within float tolerance of unit length).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ugv_localization.common.checks import NS_PER_S, require_positive_float, require_stamp
from ugv_localization.common.yamlio import load_yaml_mapping, require_exact_keys

_COV_DIAG = (0, 7, 14, 21, 28, 35)


class Rejection(str, Enum):
    NON_FINITE = "non_finite"
    FRAME_MISMATCH = "frame_mismatch"
    STAMP_NOT_INCREASING = "stamp_not_increasing"
    FUTURE_STAMP = "future_stamp"
    BAD_QUATERNION = "bad_quaternion"
    BAD_COVARIANCE = "bad_covariance"


@dataclass(frozen=True, slots=True)
class OdomGateProfile:
    odom_frame: str = "odom"
    base_frame: str = "base_link"
    max_future_s: float = 0.05
    quat_norm_tol: float = 0.01


@dataclass(frozen=True, slots=True)
class OdomSample:
    stamp_ns: int
    frame_id: str
    child_frame_id: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]  # x, y, z, w
    pose_covariance: tuple[float, ...]  # 36, row-major


@dataclass(frozen=True, slots=True)
class TfEdge:
    stamp_ns: int
    parent: str
    child: str
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class GateResult:
    edge: TfEdge | None
    reason: Rejection | None


def load_odom_gate_profile(path: str | Path) -> OdomGateProfile:
    data = load_yaml_mapping(path)
    require_exact_keys(
        data, {"odom_frame", "base_frame", "max_future_s", "quat_norm_tol"}, where=str(path)
    )
    for key in ("odom_frame", "base_frame"):
        if not isinstance(data[key], str) or not data[key]:
            raise TypeError(f"{key} must be a non-empty string")
    return OdomGateProfile(
        odom_frame=data["odom_frame"],
        base_frame=data["base_frame"],
        max_future_s=require_positive_float(data["max_future_s"], name="max_future_s", allow_zero=True),
        quat_norm_tol=require_positive_float(data["quat_norm_tol"], name="quat_norm_tol"),
    )


class OdomGate:
    def __init__(self, profile: OdomGateProfile) -> None:
        if type(profile) is not OdomGateProfile:
            raise TypeError("profile must be an OdomGateProfile")
        self._p = profile
        self._last_stamp_ns: int | None = None
        self.accepted = 0
        self.rejected = 0

    @property
    def last_stamp_ns(self) -> int | None:
        return self._last_stamp_ns

    def reset(self) -> None:
        """Call when the clock jumps backwards (bag loop / sim reset)."""
        self._last_stamp_ns = None

    def _reject(self, reason: Rejection) -> GateResult:
        self.rejected += 1
        return GateResult(edge=None, reason=reason)

    def accept(self, sample: OdomSample, now_ns: int) -> GateResult:
        stamp = require_stamp(sample.stamp_ns, name="sample.stamp_ns")
        now_ns = require_stamp(now_ns, name="now_ns")

        if sample.frame_id != self._p.odom_frame or sample.child_frame_id != self._p.base_frame:
            return self._reject(Rejection.FRAME_MISMATCH)

        values = (*sample.position, *sample.orientation, *sample.pose_covariance)
        if not all(math.isfinite(v) for v in values):
            return self._reject(Rejection.NON_FINITE)

        if len(sample.pose_covariance) != 36 or any(
            sample.pose_covariance[i] < 0.0 for i in _COV_DIAG
        ):
            return self._reject(Rejection.BAD_COVARIANCE)

        norm = math.sqrt(sum(v * v for v in sample.orientation))
        if abs(norm - 1.0) > self._p.quat_norm_tol:
            return self._reject(Rejection.BAD_QUATERNION)

        if stamp > now_ns + int(self._p.max_future_s * NS_PER_S):
            return self._reject(Rejection.FUTURE_STAMP)

        if self._last_stamp_ns is not None and stamp <= self._last_stamp_ns:
            return self._reject(Rejection.STAMP_NOT_INCREASING)

        self._last_stamp_ns = stamp
        self.accepted += 1
        rotation = tuple(v / norm for v in sample.orientation)
        return GateResult(
            edge=TfEdge(
                stamp_ns=stamp,
                parent=self._p.odom_frame,
                child=self._p.base_frame,
                translation=tuple(float(v) for v in sample.position),  # type: ignore[arg-type]
                rotation=rotation,  # type: ignore[arg-type]
            ),
            reason=None,
        )
