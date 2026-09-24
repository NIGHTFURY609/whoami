"""Drift metrics (ATE / RPE / endpoint drift). Trajectory math tables, not sensor data."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ugv_localization.drift import (
    associate,
    ate,
    drift_report,
    path_length,
    rpe_translation,
    umeyama,
)

_NS = 1_000_000_000


def _square(n_per_side: int = 50, side: float = 10.0) -> np.ndarray:
    pts = []
    corners = [(0, 0), (side, 0), (side, side), (0, side), (0, 0)]
    for (x0, y0), (x1, y1) in zip(corners, corners[1:]):
        for i in range(n_per_side):
            a = i / n_per_side
            pts.append((x0 + a * (x1 - x0), y0 + a * (y1 - y0), 0.0))
    pts.append((0.0, 0.0, 0.0))
    return np.asarray(pts)


def _rigid(pts: np.ndarray, yaw: float, t: tuple[float, float, float]) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return pts @ r.T + np.asarray(t)


def test_d1_path_length_square() -> None:
    assert path_length(_square()) == pytest.approx(40.0)


def test_d2_umeyama_recovers_rigid_transform() -> None:
    gt = _square()
    est = _rigid(gt, 0.7, (3.0, -2.0, 0.0))
    r, t, s = umeyama(est, gt, with_scale=False)
    assert s == 1.0
    np.testing.assert_allclose(est @ r.T + t, gt, atol=1e-9)


def test_d3_umeyama_with_scale() -> None:
    gt = _square()
    est = _rigid(gt, -0.3, (1.0, 1.0, 0.0)) * 0.5
    r, t, s = umeyama(est, gt, with_scale=True)
    assert s == pytest.approx(2.0)
    np.testing.assert_allclose(s * est @ r.T + t, gt, atol=1e-9)


def test_d4_ate_zero_for_rigidly_moved_trajectory() -> None:
    gt = _square()
    res = ate(_rigid(gt, 1.1, (5.0, 5.0, 0.0)), gt)
    assert res.rmse_m == pytest.approx(0.0, abs=1e-9)


def test_d5_ate_constant_offset_unaligned() -> None:
    gt = _square()
    res = ate(gt + np.array([0.3, 0.4, 0.0]), gt, align=False)
    assert res.rmse_m == pytest.approx(0.5)
    assert res.max_m == pytest.approx(0.5)


def test_d6_rpe_zero_for_rigid_and_positive_for_scale_error() -> None:
    gt = _square()
    aligned = ate(_rigid(gt, 0.4, (1.0, 2.0, 0.0)), gt).aligned_est  # RPE input must be aligned
    assert max(rpe_translation(aligned, gt, 5.0)) == pytest.approx(0.0, abs=1e-6)
    # 10 % scale error (e.g. wheel radius wrong) on a straight run → exactly 10 % RPE.
    # (Segments wrapping a corner have a shorter chord, so a square would read < 10 %.)
    line = np.column_stack([np.linspace(0.0, 20.0, 101), np.zeros(101), np.zeros(101)])
    errs = rpe_translation(line * 1.1, line, 5.0)
    assert np.mean(errs) == pytest.approx(10.0, rel=1e-6)


def test_d7_rpe_segment_longer_than_path_returns_empty() -> None:
    assert rpe_translation(_square(), _square(), 100.0) == []


def test_d8_associate_nearest_within_tolerance() -> None:
    est_t = [100, 200, 300, 400]
    gt_t = [105, 190, 330, 1000]
    pairs = associate(est_t, gt_t, max_dt_ns=20)
    assert pairs == [(0, 0), (1, 1)]


def test_d9_drift_report_end_to_end() -> None:
    gt = _square()
    t = [(i + 1) * _NS // 10 for i in range(len(gt))]
    est = gt.copy()
    est[:, 0] += np.linspace(0.0, 0.8, len(gt))  # drift grows to 0.8 m
    rep = drift_report(t, est, t, gt, max_dt_ns=_NS // 100, segments_m=(5.0, 10.0))
    assert rep.matched == len(gt)
    assert rep.gt_path_length_m == pytest.approx(40.0)
    assert rep.ate_rmse_m > 0.0
    assert set(rep.rpe_mean_pct) == {5.0, 10.0}
    assert rep.endpoint_error_m >= 0.0
    assert rep.endpoint_drift_pct == pytest.approx(100.0 * rep.endpoint_error_m / 40.0)


def test_d10_drift_report_needs_matches() -> None:
    with pytest.raises(ValueError, match="matched"):
        drift_report([1, 2], np.zeros((2, 3)), [10**12, 2 * 10**12], np.zeros((2, 3)), max_dt_ns=1)
