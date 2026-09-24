"""Per-edge TF publish cadence from header stamps.

rate_hz      = unique stamps / span
jitter_p95_s = 95th percentile of |period_i - median period|
Contract (dev.md §3): rate >= 15 Hz, jitter < 50 ms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ugv_localization.common.checks import NS_PER_S


@dataclass(frozen=True, slots=True)
class TfCheckProfile:
    min_rate_hz: float = 15.0
    max_jitter_s: float = 0.05


@dataclass(frozen=True, slots=True)
class EdgeStats:
    parent: str
    child: str
    samples: int
    duplicates: int
    rate_hz: float
    median_period_s: float
    jitter_p95_s: float
    max_gap_s: float
    passed: bool
    reasons: tuple[str, ...]


def edge_stats(
    parent: str, child: str, stamps_ns: Sequence[int], profile: TfCheckProfile
) -> EdgeStats:
    raw = [int(s) for s in stamps_ns]
    non_monotonic = any(b < a for a, b in zip(raw, raw[1:]))
    uniq = sorted(set(raw))
    duplicates = len(raw) - len(uniq)
    if len(uniq) < 3:
        return EdgeStats(parent, child, len(raw), duplicates, 0.0, 0.0, 0.0, 0.0, False, ("too_few_samples",))

    t = np.asarray(uniq, dtype=np.int64)
    periods = np.diff(t).astype(np.float64) / NS_PER_S
    span = (t[-1] - t[0]) / NS_PER_S
    rate = (len(t) - 1) / span
    median = float(np.median(periods))
    jitter = float(np.percentile(np.abs(periods - median), 95))
    max_gap = float(periods.max())

    reasons: list[str] = []
    if non_monotonic:
        reasons.append("non_monotonic")
    if rate < profile.min_rate_hz:
        reasons.append("rate_below_min")
    if jitter >= profile.max_jitter_s:
        reasons.append("jitter_above_max")
    return EdgeStats(
        parent=parent,
        child=child,
        samples=len(raw),
        duplicates=duplicates,
        rate_hz=float(rate),
        median_period_s=median,
        jitter_p95_s=jitter,
        max_gap_s=max_gap,
        passed=not reasons,
        reasons=tuple(reasons),
    )
