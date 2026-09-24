"""Pose validity kernel (architecture §10.1, §12) → /ugv/pose_valid.

Valid only if ALL hold (fail closed; startup = invalid):
  * TF map->base_link present and fresh           (§10.1 required TFs, pose age)
  * gated wheel odom fresh, covariance bounded     (§10.1 "not exploded")
  * SLAM camera CameraInfo fresh and sane          (§12 camera watch)
  * RTAB-Map alive (info fresh)                    (§10.1 status not lost)
  * localize: visually localized at least once, and dead-reckoning since the last visual
    constraint within budget — the mono "VO-lost" equivalent (mindmap gap note)
  * no recent large map->odom correction jump
  * all of the above continuously for recover_hold_s (hysteresis)

Pure: the node feeds stamps/values and a clock. No ROS imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from ugv_localization.common.checks import NS_PER_S, require_bool, require_stamp, wrap_angle
from ugv_localization.modes import Mode


class Reason(str, Enum):
    TF_MISSING = "tf_missing"
    TF_STALE = "tf_stale"
    ODOM_MISSING = "odom_missing"
    ODOM_STALE = "odom_stale"
    ODOM_COVARIANCE = "odom_covariance"
    CAMERA_MISSING = "camera_missing"
    CAMERA_STALE = "camera_stale"
    CAMERA_INFO_INVALID = "camera_info_invalid"
    SLAM_MISSING = "slam_missing"
    SLAM_STALE = "slam_stale"
    FUTURE_STAMP = "future_stamp"
    NOT_LOCALIZED = "not_localized"
    DEAD_RECKON_DISTANCE = "dead_reckon_distance"
    DEAD_RECKON_TIME = "dead_reckon_time"
    CORRECTION_JUMP = "correction_jump"
    CLOCK_RESET = "clock_reset"
    RECOVERING = "recovering"


@dataclass(frozen=True, slots=True)
class ValidityProfile:
    localization_max_age_s: float
    odom_max_age_s: float
    camera_max_age_s: float
    slam_max_age_s: float
    max_future_s: float
    max_xy_variance_m2: float
    max_yaw_variance_rad2: float
    max_dead_reckon_m: float
    max_dead_reckon_s: float  # 0.0 = disabled
    dead_reckon_in_mapping: bool
    max_correction_jump_m: float
    max_correction_jump_rad: float
    correction_jump_hold_s: float
    recover_hold_s: float
    publish_rate_hz: float


@dataclass(frozen=True, slots=True)
class ValidityResult:
    valid: bool
    reasons: tuple[Reason, ...]
    distance_since_correction_m: float | None
    seconds_since_correction: float | None


def _ns(seconds: float) -> int:
    return int(round(seconds * NS_PER_S))


class PoseValidityMonitor:
    def __init__(self, profile: ValidityProfile, mode: Mode) -> None:
        if type(profile) is not ValidityProfile:
            raise TypeError("profile must be a ValidityProfile")
        if type(mode) is not Mode:
            raise TypeError("mode must be a Mode")
        self._p = profile
        self._mode = mode
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self._tf_ns: int | None = None
        self._odom_ns: int | None = None
        self._odom_xy: tuple[float, float] | None = None
        self._odom_cov_ok = False
        self._cam_ns: int | None = None
        self._cam_ok = False
        self._slam_ns: int | None = None
        self._path_m = 0.0
        mapping = self._mode is Mode.MAPPING
        # Mapping defines the map frame at start; localize must earn it visually.
        self._localized = mapping
        self._path_at_correction: float | None = 0.0 if mapping else None
        self._last_correction_ns: int | None = None
        self._last_m2o: tuple[float, float, float] | None = None
        self._jump_until_ns: int | None = None
        self._good_since_ns: int | None = None
        self._last_eval_ns: int | None = None

    # ----------------------------------------------------------------- inputs
    def on_tf(self, stamp_ns: int) -> None:
        """Stamp of the latest available map->base_link transform."""
        self._tf_ns = require_stamp(stamp_ns, name="stamp_ns")

    def on_odom(
        self, stamp_ns: int, x: float, y: float, var_x: float, var_y: float, var_yaw: float
    ) -> None:
        stamp_ns = require_stamp(stamp_ns, name="stamp_ns")
        if self._odom_ns is not None and stamp_ns <= self._odom_ns:
            return  # duplicate / reordered sample: never counts toward path or freshness
        finite = all(math.isfinite(v) for v in (x, y, var_x, var_y, var_yaw))
        self._odom_cov_ok = (
            finite
            and 0.0 <= var_x <= self._p.max_xy_variance_m2
            and 0.0 <= var_y <= self._p.max_xy_variance_m2
            and 0.0 <= var_yaw <= self._p.max_yaw_variance_rad2
        )
        if finite:
            if self._odom_xy is not None:
                self._path_m += math.hypot(x - self._odom_xy[0], y - self._odom_xy[1])
            self._odom_xy = (x, y)
        self._odom_ns = stamp_ns

    def on_camera_info(self, stamp_ns: int, ok: bool) -> None:
        self._cam_ns = require_stamp(stamp_ns, name="stamp_ns")
        self._cam_ok = require_bool(ok, name="ok")

    def on_slam_info(self, stamp_ns: int, visual_constraint: bool) -> None:
        """visual_constraint: RTAB-Map matched the current frame to the map (loop closure /
        proximity / relocalization) — the only thing that bounds mono+wheel drift."""
        self._slam_ns = require_stamp(stamp_ns, name="stamp_ns")
        if require_bool(visual_constraint, name="visual_constraint"):
            self._localized = True
            self._last_correction_ns = stamp_ns
            self._path_at_correction = self._path_m

    def on_map_to_odom(self, stamp_ns: int, x: float, y: float, yaw: float) -> None:
        """Call when a new map->odom transform (new stamp) is seen."""
        stamp_ns = require_stamp(stamp_ns, name="stamp_ns")
        if not self._localized:
            return  # pre-localization identity is meaningless; don't arm on it
        current = (float(x), float(y), float(yaw))
        prev = self._last_m2o
        self._last_m2o = current
        if prev is None:
            return  # first map->odom after (re)localization: the big jump is legit
        d_xy = math.hypot(current[0] - prev[0], current[1] - prev[1])
        d_yaw = abs(wrap_angle(current[2] - prev[2]))
        if d_xy > self._p.max_correction_jump_m or d_yaw > self._p.max_correction_jump_rad:
            self._jump_until_ns = stamp_ns + _ns(self._p.correction_jump_hold_s)

    # --------------------------------------------------------------- evaluate
    def evaluate(self, now_ns: int) -> ValidityResult:
        now_ns = require_stamp(now_ns, name="now_ns")
        if self._last_eval_ns is not None and now_ns < self._last_eval_ns:
            self.reset()
            self._last_eval_ns = now_ns
            return ValidityResult(False, (Reason.CLOCK_RESET,), None, None)
        self._last_eval_ns = now_ns

        reasons: list[Reason] = []
        future_ns = _ns(self._p.max_future_s)

        def fresh(stamp: int | None, max_age_s: float, missing: Reason, stale: Reason) -> None:
            if stamp is None:
                reasons.append(missing)
            elif stamp > now_ns + future_ns:
                reasons.append(Reason.FUTURE_STAMP)
            elif now_ns - stamp > _ns(max_age_s):
                reasons.append(stale)

        fresh(self._tf_ns, self._p.localization_max_age_s, Reason.TF_MISSING, Reason.TF_STALE)
        fresh(self._odom_ns, self._p.odom_max_age_s, Reason.ODOM_MISSING, Reason.ODOM_STALE)
        if self._odom_ns is not None and not self._odom_cov_ok:
            reasons.append(Reason.ODOM_COVARIANCE)
        fresh(self._cam_ns, self._p.camera_max_age_s, Reason.CAMERA_MISSING, Reason.CAMERA_STALE)
        if self._cam_ns is not None and not self._cam_ok:
            reasons.append(Reason.CAMERA_INFO_INVALID)
        fresh(self._slam_ns, self._p.slam_max_age_s, Reason.SLAM_MISSING, Reason.SLAM_STALE)

        distance: float | None = None
        seconds: float | None = None
        if self._path_at_correction is not None:
            distance = self._path_m - self._path_at_correction
        if self._last_correction_ns is not None:
            seconds = (now_ns - self._last_correction_ns) / NS_PER_S

        if not self._localized:
            reasons.append(Reason.NOT_LOCALIZED)
        elif self._mode is Mode.LOCALIZE or self._p.dead_reckon_in_mapping:
            if distance is not None and distance > self._p.max_dead_reckon_m:
                reasons.append(Reason.DEAD_RECKON_DISTANCE)
            if (
                self._p.max_dead_reckon_s > 0.0
                and seconds is not None
                and seconds > self._p.max_dead_reckon_s
            ):
                reasons.append(Reason.DEAD_RECKON_TIME)

        if self._jump_until_ns is not None and now_ns < self._jump_until_ns:
            reasons.append(Reason.CORRECTION_JUMP)

        hard = tuple(dict.fromkeys(reasons))  # dedupe, keep order
        if hard:
            self._good_since_ns = None
            return ValidityResult(False, hard, distance, seconds)

        if self._good_since_ns is None:
            self._good_since_ns = now_ns
        if now_ns - self._good_since_ns >= _ns(self._p.recover_hold_s):
            return ValidityResult(True, (), distance, seconds)
        return ValidityResult(False, (Reason.RECOVERING,), distance, seconds)
