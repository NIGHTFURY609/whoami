"""Trajectory drift metrics for sensor-honesty benchmarks (dev.md Dev 2 task 5).

Estimated = map->base_link (Dev 2). Ground truth = sim model pose (Dev 5) or survey.
ATE: RMSE after rigid (optionally similarity) Umeyama alignment.
RPE: translation-only, over ground-truth path segments of fixed length, as % of segment,
     on the aligned estimate. Endpoint drift: final aligned error / GT path length.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class AteResult:
    rmse_m: float
    mean_m: float
    max_m: float
    aligned_est: np.ndarray


@dataclass(frozen=True, slots=True)
class DriftReport:
    matched: int
    gt_path_length_m: float
    ate_rmse_m: float
    ate_max_m: float
    rpe_mean_pct: dict[float, float]
    endpoint_error_m: float
    endpoint_drift_pct: float


def _as_xyz(a: np.ndarray | Sequence) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"trajectory must be (N,3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("trajectory contains non-finite values")
    return arr


def path_length(xyz: np.ndarray) -> float:
    p = _as_xyz(xyz)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def umeyama(src: np.ndarray, dst: np.ndarray, *, with_scale: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
    """Find R, t, s minimizing ||dst - (s R src + t)||. Returns (R 3x3, t 3, s)."""
    x = _as_xyz(src)
    y = _as_xyz(dst)
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("umeyama needs >= 3 matched points")
    mx, my = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - mx, y - my
    cov = yc.T @ xc / len(x)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1.0
    r = u @ sign @ vt
    s = 1.0
    if with_scale:
        var_x = (xc**2).sum() / len(x)
        s = float(np.trace(np.diag(d) @ sign) / var_x)
    t = my - s * r @ mx
    return r, t, s


def ate(est: np.ndarray, gt: np.ndarray, *, align: bool = True, with_scale: bool = False) -> AteResult:
    e = _as_xyz(est)
    g = _as_xyz(gt)
    if len(e) != len(g):
        raise ValueError("est and gt must be matched (same length)")
    if align:
        r, t, s = umeyama(e, g, with_scale=with_scale)
        e = s * e @ r.T + t
    err = np.linalg.norm(e - g, axis=1)
    return AteResult(
        rmse_m=float(np.sqrt(np.mean(err**2))),
        mean_m=float(err.mean()),
        max_m=float(err.max()),
        aligned_est=e,
    )


def rpe_translation(est: np.ndarray, gt: np.ndarray, segment_m: float) -> list[float]:
    """Relative translation error (% of segment) for every GT segment of length >= segment_m.

    `est` must already be expressed in the GT frame (e.g. `ate(...).aligned_est`); displacement
    vectors are compared directly, without per-segment rotation.
    """
    e = _as_xyz(est)
    g = _as_xyz(gt)
    if len(e) != len(g):
        raise ValueError("est and gt must be matched (same length)")
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(g, axis=0), axis=1))])
    out: list[float] = []
    for i in range(len(g)):
        j = int(np.searchsorted(cum, cum[i] + segment_m, side="left"))
        if j >= len(g):
            break
        seg = cum[j] - cum[i]
        err = np.linalg.norm((e[j] - e[i]) - (g[j] - g[i]))
        out.append(float(100.0 * err / seg))
    return out


def associate(est_t_ns: Sequence[int], gt_t_ns: Sequence[int], *, max_dt_ns: int) -> list[tuple[int, int]]:
    """Nearest GT stamp for each estimate stamp, within max_dt_ns; each GT used once."""
    gt_sorted = list(gt_t_ns)
    if any(b < a for a, b in zip(gt_sorted, gt_sorted[1:])):
        raise ValueError("gt stamps must be sorted")
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for i, t in enumerate(est_t_ns):
        k = bisect.bisect_left(gt_sorted, t)
        best = None
        for cand in (k - 1, k):
            if 0 <= cand < len(gt_sorted) and cand not in used:
                dt = abs(gt_sorted[cand] - t)
                if dt <= max_dt_ns and (best is None or dt < best[1]):
                    best = (cand, dt)
        if best is not None:
            used.add(best[0])
            pairs.append((i, best[0]))
    return pairs


def drift_report(
    est_t_ns: Sequence[int],
    est_xyz: np.ndarray,
    gt_t_ns: Sequence[int],
    gt_xyz: np.ndarray,
    *,
    max_dt_ns: int,
    segments_m: Sequence[float] = (5.0, 10.0, 20.0),
    with_scale: bool = False,
) -> DriftReport:
    pairs = associate(est_t_ns, gt_t_ns, max_dt_ns=max_dt_ns)
    if len(pairs) < 3:
        raise ValueError(f"only {len(pairs)} matched poses; need >= 3")
    e = _as_xyz(est_xyz)[[p[0] for p in pairs]]
    g = _as_xyz(gt_xyz)[[p[1] for p in pairs]]
    a = ate(e, g, align=True, with_scale=with_scale)
    length = path_length(g)
    rpe = {}
    for seg in segments_m:
        errs = rpe_translation(a.aligned_est, g, seg)
        rpe[float(seg)] = float(np.mean(errs)) if errs else float("nan")
    end_err = float(np.linalg.norm(a.aligned_est[-1] - g[-1]))
    return DriftReport(
        matched=len(pairs),
        gt_path_length_m=length,
        ate_rmse_m=a.rmse_m,
        ate_max_m=a.max_m,
        rpe_mean_pct=rpe,
        endpoint_error_m=end_err,
        endpoint_drift_pct=100.0 * end_err / length if length > 0 else float("nan"),
    )
