"""TF edge rate/jitter stats (dev.md §3: rate >= 15 Hz, jitter < 50 ms). Stamp tables only."""

from __future__ import annotations

import pytest

from ugv_localization.tfcheck import TfCheckProfile, edge_stats

_NS = 1_000_000_000
_P = TfCheckProfile(min_rate_hz=15.0, max_jitter_s=0.05)


def _uniform(hz: float, n: int, t0: int = 10 * _NS) -> list[int]:
    return [t0 + int(round(i * _NS / hz)) for i in range(n)]


def test_t1_uniform_20hz_passes() -> None:
    s = edge_stats("map", "odom", _uniform(20.0, 100), _P)
    assert s.passed and s.reasons == ()
    assert s.rate_hz == pytest.approx(20.0, rel=1e-3)
    assert s.jitter_p95_s == pytest.approx(0.0, abs=1e-6)


def test_t2_10hz_fails_rate() -> None:
    s = edge_stats("map", "odom", _uniform(10.0, 50), _P)
    assert not s.passed and "rate_below_min" in s.reasons


def test_t3_irregular_fails_jitter() -> None:
    stamps = _uniform(20.0, 100)
    # every 5th gap stretched by 120 ms
    shifted = []
    offset = 0
    for i, t in enumerate(stamps):
        if i and i % 5 == 0:
            offset += int(0.12 * _NS)
        shifted.append(t + offset)
    s = edge_stats("odom", "base_link", shifted, _P)
    assert "jitter_above_max" in s.reasons
    assert s.max_gap_s == pytest.approx(0.17, abs=1e-3)


def test_t4_duplicate_stamps_counted_not_rated() -> None:
    stamps = _uniform(20.0, 50)
    dup = sorted(stamps + stamps[:10])
    s = edge_stats("map", "odom", dup, _P)
    assert s.duplicates == 10
    assert s.rate_hz == pytest.approx(20.0, rel=1e-3)


def test_t5_non_monotonic_flagged() -> None:
    stamps = _uniform(20.0, 20)
    stamps[5], stamps[6] = stamps[6], stamps[5]
    s = edge_stats("map", "odom", stamps, _P)
    assert "non_monotonic" in s.reasons and not s.passed


def test_t6_too_few_samples() -> None:
    s = edge_stats("map", "odom", _uniform(20.0, 2), _P)
    assert not s.passed and s.reasons == ("too_few_samples",)


def test_t7_no_samples() -> None:
    s = edge_stats("map", "odom", [], _P)
    assert not s.passed and s.samples == 0
