"""Odom gate: wheel odometry → odom->base_link edge. Clocks + numeric tables."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from ugv_localization.odom import (
    OdomGate,
    OdomGateProfile,
    OdomSample,
    Rejection,
    load_odom_gate_profile,
)

_NS = 1_000_000_000
_T0 = 100 * _NS
_COV = tuple(0.01 if i in (0, 7, 14, 21, 28, 35) else 0.0 for i in range(36))


def _sample(stamp_ns: int = _T0, **over) -> OdomSample:
    base = dict(
        stamp_ns=stamp_ns,
        frame_id="odom",
        child_frame_id="base_link",
        position=(1.0, 2.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        pose_covariance=_COV,
    )
    base.update(over)
    return OdomSample(**base)


def _gate() -> OdomGate:
    return OdomGate(OdomGateProfile())


def test_o1_accepts_and_preserves_stamp() -> None:
    res = _gate().accept(_sample(), now_ns=_T0)
    assert res.reason is None
    edge = res.edge
    assert edge is not None
    assert edge.stamp_ns == _T0  # never restamped to "now"
    assert (edge.parent, edge.child) == ("odom", "base_link")
    assert edge.translation == (1.0, 2.0, 0.0)


def test_o2_frame_mismatch_rejected() -> None:
    g = _gate()
    assert g.accept(_sample(frame_id="world"), _T0).reason is Rejection.FRAME_MISMATCH
    assert g.accept(_sample(child_frame_id="base_footprint"), _T0).reason is Rejection.FRAME_MISMATCH


def test_o3_non_increasing_stamp_rejected() -> None:
    g = _gate()
    assert g.accept(_sample(_T0), _T0).edge is not None
    assert g.accept(_sample(_T0), _T0).reason is Rejection.STAMP_NOT_INCREASING
    assert g.accept(_sample(_T0 - 1), _T0).reason is Rejection.STAMP_NOT_INCREASING
    assert g.accept(_sample(_T0 + 1), _T0 + 1).edge is not None


def test_o4_future_stamp_rejected() -> None:
    g = _gate()
    res = g.accept(_sample(_T0 + int(0.2 * _NS)), now_ns=_T0)
    assert res.reason is Rejection.FUTURE_STAMP


def test_o5_small_future_within_tolerance_ok() -> None:
    res = _gate().accept(_sample(_T0 + int(0.01 * _NS)), now_ns=_T0)
    assert res.edge is not None


@pytest.mark.parametrize(
    "field, value",
    [
        ("position", (float("nan"), 0.0, 0.0)),
        ("position", (0.0, float("inf"), 0.0)),
        ("orientation", (0.0, 0.0, float("nan"), 1.0)),
    ],
)
def test_o6_non_finite_rejected(field, value) -> None:
    assert _gate().accept(_sample(**{field: value}), _T0).reason is Rejection.NON_FINITE


def test_o7_near_unit_quaternion_normalized() -> None:
    q = (0.0, 0.0, 0.0, 1.0005)
    edge = _gate().accept(_sample(orientation=q), _T0).edge
    assert edge is not None
    assert math.isclose(sum(v * v for v in edge.rotation), 1.0, abs_tol=1e-12)


def test_o8_far_from_unit_quaternion_rejected() -> None:
    assert _gate().accept(_sample(orientation=(0.0, 0.0, 0.0, 0.5)), _T0).reason is Rejection.BAD_QUATERNION
    assert _gate().accept(_sample(orientation=(0.0, 0.0, 0.0, 0.0)), _T0).reason is Rejection.BAD_QUATERNION


def test_o9_negative_covariance_diag_rejected() -> None:
    cov = list(_COV)
    cov[7] = -0.1
    assert _gate().accept(_sample(pose_covariance=tuple(cov)), _T0).reason is Rejection.BAD_COVARIANCE


def test_o10_covariance_wrong_length_rejected() -> None:
    assert _gate().accept(_sample(pose_covariance=(0.0,) * 35), _T0).reason is Rejection.BAD_COVARIANCE


def test_o11_nan_covariance_rejected() -> None:
    cov = list(_COV)
    cov[0] = float("nan")
    assert _gate().accept(_sample(pose_covariance=tuple(cov)), _T0).reason is Rejection.NON_FINITE


def test_o12_rejected_sample_does_not_advance_last_stamp() -> None:
    g = _gate()
    assert g.accept(_sample(_T0, frame_id="world"), _T0).edge is None
    assert g.last_stamp_ns is None
    assert g.accept(_sample(_T0), _T0).edge is not None
    assert g.last_stamp_ns == _T0


def test_o13_counters() -> None:
    g = _gate()
    g.accept(_sample(_T0), _T0)
    g.accept(_sample(_T0), _T0)
    assert (g.accepted, g.rejected) == (1, 1)


def test_o14_reset_allows_restart_after_clock_reset() -> None:
    g = _gate()
    g.accept(_sample(_T0), _T0)
    g.reset()
    assert g.accept(_sample(_T0 - _NS), _T0 - _NS).edge is not None


def test_o15_stamp_type_strict() -> None:
    with pytest.raises(TypeError):
        _gate().accept(_sample(stamp_ns=1.5), _T0)  # type: ignore[arg-type]


def test_o16_profile_yaml(tmp_path: Path) -> None:
    p = tmp_path / "odom.yaml"
    p.write_text(
        "odom_frame: odom\nbase_frame: base_link\nmax_future_s: 0.05\nquat_norm_tol: 0.01\n",
        encoding="utf-8",
    )
    prof = load_odom_gate_profile(p)
    assert prof == OdomGateProfile("odom", "base_link", 0.05, 0.01)


def test_o17_profile_yaml_rejects_unknown_key(tmp_path: Path) -> None:
    p = tmp_path / "odom.yaml"
    p.write_text(
        "odom_frame: odom\nbase_frame: base_link\nmax_future_s: 0.05\nquat_norm_tol: 0.01\nextra: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(KeyError, match="extra"):
        load_odom_gate_profile(p)


def test_o18_product_profile_loads() -> None:
    product = Path(__file__).resolve().parents[1] / "config" / "odom_bridge.yaml"
    prof = load_odom_gate_profile(product)
    assert prof.odom_frame == "odom" and prof.base_frame == "base_link"
