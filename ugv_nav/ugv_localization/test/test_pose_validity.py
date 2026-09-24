"""Pose validity (architecture §10.1, §12). Clocks and numbers only — no ROS, no images."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from ugv_localization.modes import Mode
from ugv_localization.validity import (
    PoseValidityMonitor,
    Reason,
    ValidityProfile,
    load_validity_profile,
)

_NS = 1_000_000_000
_T0 = 1_000 * _NS
_PRODUCT = Path(__file__).resolve().parents[1] / "config" / "pose_validity.yaml"


def _profile(**over) -> ValidityProfile:
    base = ValidityProfile(
        localization_max_age_s=0.5,
        odom_max_age_s=0.5,
        camera_max_age_s=0.5,
        slam_max_age_s=2.0,
        max_future_s=0.05,
        max_xy_variance_m2=1.0,
        max_yaw_variance_rad2=0.25,
        max_dead_reckon_m=8.0,
        max_dead_reckon_s=0.0,
        dead_reckon_in_mapping=False,
        max_correction_jump_m=1.0,
        max_correction_jump_rad=0.35,
        correction_jump_hold_s=1.0,
        recover_hold_s=0.0,
        publish_rate_hz=20.0,
    )
    return dataclasses.replace(base, **over)


def _feed(m: PoseValidityMonitor, t: int, *, x: float = 0.0, cam_ok: bool = True, visual: bool = False) -> None:
    m.on_tf(t)
    m.on_odom(t, x, 0.0, 0.01, 0.01, 0.01)
    m.on_camera_info(t, cam_ok)
    m.on_slam_info(t, visual)


def _mapping(**over) -> PoseValidityMonitor:
    return PoseValidityMonitor(_profile(**over), Mode.MAPPING)


def _localize(**over) -> PoseValidityMonitor:
    return PoseValidityMonitor(_profile(**over), Mode.LOCALIZE)


# --- fail closed at startup ---------------------------------------------------------------


def test_v1_startup_is_invalid_with_all_missing_reasons() -> None:
    res = _mapping().evaluate(_T0)
    assert res.valid is False
    assert set(res.reasons) >= {
        Reason.TF_MISSING,
        Reason.ODOM_MISSING,
        Reason.CAMERA_MISSING,
        Reason.SLAM_MISSING,
    }


def test_v2_mapping_all_fresh_is_valid() -> None:
    m = _mapping()
    _feed(m, _T0)
    res = m.evaluate(_T0)
    assert res.valid is True and res.reasons == ()


# --- freshness (each input independently trips) -------------------------------------------


@pytest.mark.parametrize(
    "stale, reason",
    [
        ("tf", Reason.TF_STALE),
        ("odom", Reason.ODOM_STALE),
        ("camera", Reason.CAMERA_STALE),
    ],
)
def test_v3_stale_input_invalidates(stale: str, reason: Reason) -> None:
    m = _mapping()
    _feed(m, _T0)
    t1 = _T0 + int(0.6 * _NS)
    if stale != "tf":
        m.on_tf(t1)
    if stale != "odom":
        m.on_odom(t1, 0.0, 0.0, 0.01, 0.01, 0.01)
    if stale != "camera":
        m.on_camera_info(t1, True)
    m.on_slam_info(t1, False)
    res = m.evaluate(t1)
    assert res.valid is False
    assert reason in res.reasons


def test_v4_age_exactly_at_limit_is_fresh() -> None:
    m = _mapping()
    _feed(m, _T0)
    assert m.evaluate(_T0 + int(0.5 * _NS)).valid is True


def test_v5_slam_uses_its_own_longer_budget() -> None:
    m = _mapping()
    _feed(m, _T0)
    t1 = _T0 + int(1.5 * _NS)
    m.on_tf(t1)
    m.on_odom(t1, 0.0, 0.0, 0.01, 0.01, 0.01)
    m.on_camera_info(t1, True)
    assert m.evaluate(t1).valid is True  # slam 1.5 s old < 2.0 s
    t2 = _T0 + int(2.5 * _NS)
    m.on_tf(t2)
    m.on_odom(t2, 0.0, 0.0, 0.01, 0.01, 0.01)
    m.on_camera_info(t2, True)
    res = m.evaluate(t2)
    assert res.valid is False and Reason.SLAM_STALE in res.reasons


def test_v6_future_stamp_is_a_lie() -> None:
    m = _mapping()
    _feed(m, _T0 + int(0.2 * _NS))
    res = m.evaluate(_T0)
    assert res.valid is False and Reason.FUTURE_STAMP in res.reasons
    assert res.reasons.count(Reason.FUTURE_STAMP) == 1


# --- camera info / covariance -------------------------------------------------------------


def test_v7_invalid_camera_info_invalidates() -> None:
    m = _mapping()
    _feed(m, _T0, cam_ok=False)
    res = m.evaluate(_T0)
    assert res.valid is False and Reason.CAMERA_INFO_INVALID in res.reasons


@pytest.mark.parametrize(
    "var",
    [(2.0, 0.01, 0.01), (0.01, 2.0, 0.01), (0.01, 0.01, 1.0), (float("nan"), 0.01, 0.01)],
)
def test_v8_covariance_explodes(var) -> None:
    m = _mapping()
    _feed(m, _T0)
    m.on_odom(_T0 + 1, 0.0, 0.0, *var)
    res = m.evaluate(_T0 + 1)
    assert res.valid is False and Reason.ODOM_COVARIANCE in res.reasons


# --- localize mode: must relocalize, then drift budget ------------------------------------


def test_v9_localize_needs_initial_visual_localization() -> None:
    m = _localize()
    _feed(m, _T0)
    res = m.evaluate(_T0)
    assert res.valid is False and res.reasons == (Reason.NOT_LOCALIZED,)
    _feed(m, _T0 + 1, visual=True)
    assert m.evaluate(_T0 + 1).valid is True


def test_v10_localize_dead_reckon_distance_budget() -> None:
    m = _localize()
    _feed(m, _T0, x=0.0, visual=True)
    t = _T0
    for i in range(1, 8):  # drive 7 m with no visual correction
        t = _T0 + i * _NS // 10
        _feed(m, t, x=float(i))
    res = m.evaluate(t)
    assert res.valid is True
    assert res.distance_since_correction_m == pytest.approx(7.0)
    t += _NS // 10
    _feed(m, t, x=9.0)  # 9 m > 8 m budget
    res = m.evaluate(t)
    assert res.valid is False and Reason.DEAD_RECKON_DISTANCE in res.reasons
    t += _NS // 10
    _feed(m, t, x=9.5, visual=True)  # visual correction resets the budget
    res = m.evaluate(t)
    assert res.valid is True and res.distance_since_correction_m == pytest.approx(0.0)


def test_v11_path_length_counts_back_and_forth() -> None:
    m = _localize()
    _feed(m, _T0, x=0.0, visual=True)
    xs = [2.0, 0.0, 2.0, 0.0, 2.0]  # 10 m travelled, net 2 m
    t = _T0
    for i, x in enumerate(xs, start=1):
        t = _T0 + i * _NS // 10
        _feed(m, t, x=x)
    res = m.evaluate(t)
    assert res.distance_since_correction_m == pytest.approx(10.0)
    assert Reason.DEAD_RECKON_DISTANCE in res.reasons


def test_v12_dead_reckon_time_budget_optional() -> None:
    m = _localize(max_dead_reckon_s=3.0, slam_max_age_s=10.0)
    _feed(m, _T0, visual=True)
    t = _T0 + 4 * _NS
    _feed(m, t)
    res = m.evaluate(t)
    assert Reason.DEAD_RECKON_TIME in res.reasons
    assert res.seconds_since_correction == pytest.approx(4.0)


def test_v13_mapping_ignores_dead_reckon_by_default() -> None:
    m = _mapping()
    _feed(m, _T0)
    t = _T0 + _NS // 10
    _feed(m, t, x=50.0)
    assert m.evaluate(t).valid is True


def test_v14_mapping_dead_reckon_when_enabled() -> None:
    m = _mapping(dead_reckon_in_mapping=True)
    _feed(m, _T0)
    t = _T0 + _NS // 10
    _feed(m, t, x=50.0)
    res = m.evaluate(t)
    assert Reason.DEAD_RECKON_DISTANCE in res.reasons


def test_v15_odom_non_increasing_stamp_ignored_for_path() -> None:
    m = _localize()
    _feed(m, _T0, x=0.0, visual=True)
    m.on_odom(_T0, 100.0, 0.0, 0.01, 0.01, 0.01)  # duplicate stamp, bogus jump
    res = m.evaluate(_T0)
    assert res.distance_since_correction_m == pytest.approx(0.0)


# --- map->odom correction jumps -----------------------------------------------------------


def test_v16_first_relocalization_jump_is_not_flagged() -> None:
    m = _localize()
    m.on_map_to_odom(_T0, 0.0, 0.0, 0.0)  # identity before localization: ignored
    _feed(m, _T0, visual=True)
    m.on_map_to_odom(_T0 + 1, 25.0, -4.0, 1.2)  # relocalized far away: legit
    _feed(m, _T0 + 1)
    assert m.evaluate(_T0 + 1).valid is True


def test_v17_later_big_jump_holds_then_releases() -> None:
    m = _localize()
    _feed(m, _T0, visual=True)
    m.on_map_to_odom(_T0, 0.0, 0.0, 0.0)
    t1 = _T0 + _NS // 10
    m.on_map_to_odom(t1, 3.0, 0.0, 0.0)  # 3 m > 1 m
    _feed(m, t1, visual=True)
    res = m.evaluate(t1)
    assert res.valid is False and Reason.CORRECTION_JUMP in res.reasons
    t2 = t1 + int(1.1 * _NS)
    _feed(m, t2, visual=True)
    assert m.evaluate(t2).valid is True


def test_v18_yaw_jump_uses_wrapped_angle() -> None:
    m = _mapping()
    _feed(m, _T0)
    m.on_map_to_odom(_T0, 0.0, 0.0, 3.1)
    m.on_map_to_odom(_T0 + 1, 0.0, 0.0, -3.1)  # 0.083 rad across ±pi, not 6.2
    _feed(m, _T0 + 1)
    assert m.evaluate(_T0 + 1).valid is True
    m.on_map_to_odom(_T0 + 2, 0.0, 0.0, -2.5)  # 0.6 rad > 0.35
    _feed(m, _T0 + 2)
    assert Reason.CORRECTION_JUMP in m.evaluate(_T0 + 2).reasons


# --- hysteresis + clock reset -------------------------------------------------------------


def test_v19_recover_hold_hysteresis() -> None:
    m = _mapping(recover_hold_s=1.0)
    _feed(m, _T0)
    res = m.evaluate(_T0)
    assert res.valid is False and res.reasons == (Reason.RECOVERING,)
    t = _T0 + _NS // 2
    _feed(m, t)
    assert m.evaluate(t).valid is False
    t = _T0 + _NS
    _feed(m, t)
    assert m.evaluate(t).valid is True
    # one bad tick restarts the hold
    t2 = t + _NS // 10
    _feed(m, t2, cam_ok=False)
    assert m.evaluate(t2).valid is False
    t3 = t2 + _NS // 10
    _feed(m, t3)
    assert m.evaluate(t3).reasons == (Reason.RECOVERING,)


def test_v20_clock_going_backwards_resets_and_invalidates() -> None:
    m = _localize()
    _feed(m, _T0, visual=True)
    assert m.evaluate(_T0).valid is True
    earlier = _T0 - 10 * _NS  # bag loop / sim reset
    res = m.evaluate(earlier)
    assert res.valid is False and Reason.CLOCK_RESET in res.reasons
    _feed(m, earlier)  # fresh inputs but localization state was wiped
    assert m.evaluate(earlier).reasons == (Reason.NOT_LOCALIZED,)


def test_v21_constructor_type_checks() -> None:
    with pytest.raises(TypeError):
        PoseValidityMonitor(_profile(), "mapping")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _mapping().evaluate(1.0)  # type: ignore[arg-type]


# --- profile YAML -------------------------------------------------------------------------


def test_v22_product_profile_loads_and_matches_contract() -> None:
    prof = load_validity_profile(_PRODUCT)
    # dev.md §12 / architecture §12 timeout table: localization/TF 0.5 s
    assert prof.localization_max_age_s == 0.5
    assert prof.publish_rate_hz >= 15.0


def test_v23_profile_int_threshold_rejected(tmp_path: Path) -> None:
    text = _PRODUCT.read_text(encoding="utf-8").replace(
        "localization_max_age_s: 0.5", "localization_max_age_s: 1"
    )
    p = tmp_path / "v.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(TypeError, match="localization_max_age_s"):
        load_validity_profile(p)


def test_v24_profile_unknown_key_rejected(tmp_path: Path) -> None:
    p = tmp_path / "v.yaml"
    p.write_text(_PRODUCT.read_text(encoding="utf-8") + "\nsurprise: 1.0\n", encoding="utf-8")
    with pytest.raises(KeyError, match="surprise"):
        load_validity_profile(p)


def test_v25_profile_zero_age_rejected(tmp_path: Path) -> None:
    text = _PRODUCT.read_text(encoding="utf-8").replace(
        "odom_max_age_s: 0.5", "odom_max_age_s: 0.0"
    )
    p = tmp_path / "v.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="odom_max_age_s"):
        load_validity_profile(p)
