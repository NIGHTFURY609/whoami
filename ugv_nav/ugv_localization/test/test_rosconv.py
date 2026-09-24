"""Message → kernel conversions with plain fixture objects (no ROS install needed)."""

from __future__ import annotations

import math
from types import SimpleNamespace as NS

import pytest

from ugv_localization.odom import OdomGate, OdomGateProfile
from ugv_localization.rosconv import (
    calibration_from_camera_info_msg,
    camera_info_ok,
    ns_to_sec_nanosec,
    odom_msg_to_sample,
    odom_variances,
    slam_info_visual_constraint,
    stamp_to_ns,
    transform_to_xy_yaw,
)

_COV = [0.0] * 36
_COV[0], _COV[7], _COV[35] = 0.02, 0.03, 0.04
_K = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]


def _stamp(sec: int = 12, nsec: int = 500) -> NS:
    return NS(sec=sec, nanosec=nsec)


def _odom(**over) -> NS:
    msg = NS(
        header=NS(stamp=_stamp(), frame_id="odom"),
        child_frame_id="base_link",
        pose=NS(
            pose=NS(position=NS(x=1.0, y=2.0, z=0.0), orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0)),
            covariance=list(_COV),
        ),
    )
    for k, v in over.items():
        setattr(msg, k, v)
    return msg


def _cinfo(k=None, d=None, w=640, h=480) -> NS:
    return NS(
        width=w,
        height=h,
        distortion_model="plumb_bob",
        d=[0.0] * 5 if d is None else d,
        k=list(_K) if k is None else k,
        r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        p=[500.0, 0.0, 320.0, 0.0, 0.0, 500.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )


def test_r1_stamp_roundtrip() -> None:
    ns = stamp_to_ns(_stamp(12, 500))
    assert ns == 12_000_000_500
    assert ns_to_sec_nanosec(ns) == (12, 500)


@pytest.mark.parametrize("sec,nsec,exc", [(0, 0, ValueError), (-1, 0, ValueError), (1, 10**9, ValueError), (1.0, 0, TypeError)])
def test_r2_bad_stamps(sec, nsec, exc) -> None:
    with pytest.raises(exc):
        stamp_to_ns(NS(sec=sec, nanosec=nsec))


def test_r3_odom_to_sample_feeds_gate() -> None:
    sample = odom_msg_to_sample(_odom())
    assert sample.stamp_ns == 12_000_000_500
    assert sample.frame_id == "odom" and sample.child_frame_id == "base_link"
    res = OdomGate(OdomGateProfile()).accept(sample, now_ns=sample.stamp_ns)
    assert res.edge is not None and res.edge.stamp_ns == sample.stamp_ns


def test_r4_odom_variances() -> None:
    assert odom_variances(_odom()) == (0.02, 0.03, 0.04)
    bad = _odom()
    bad.pose.covariance = [0.0] * 10
    assert all(math.isnan(v) for v in odom_variances(bad))


def test_r5_camera_info_ok() -> None:
    assert camera_info_ok(_cinfo()) is True
    assert camera_info_ok(_cinfo(k=[0.0] * 9)) is False  # uncalibrated driver default
    assert camera_info_ok(_cinfo(d=[float("nan")] * 5)) is False
    assert camera_info_ok(_cinfo(w=0)) is False


def test_r6_calibration_from_msg() -> None:
    cal = calibration_from_camera_info_msg(_cinfo(d=[]), "sim_front")
    assert cal.camera_name == "sim_front" and cal.d == (0.0,) * 5


@pytest.mark.parametrize(
    "fields, expected",
    [
        ({"loop_closure_id": 0, "proximity_detection_id": 0}, False),
        ({"loop_closure_id": 7, "proximity_detection_id": 0}, True),
        ({"loop_closure_id": 0, "proximity_detection_id": 3}, True),
        ({"loop_closure_id": 0, "proximity_detection_id": 0, "landmark_id": 2}, True),
        ({}, False),
    ],
)
def test_r7_visual_constraint(fields, expected) -> None:
    assert slam_info_visual_constraint(NS(**fields)) is expected


def test_r8_transform_yaw() -> None:
    half = math.pi / 4
    tf = NS(translation=NS(x=1.0, y=-2.0, z=0.0), rotation=NS(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half)))
    x, y, yaw = transform_to_xy_yaw(tf)
    assert (x, y) == (1.0, -2.0)
    assert yaw == pytest.approx(math.pi / 2)
