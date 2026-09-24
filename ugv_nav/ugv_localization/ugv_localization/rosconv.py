"""ROS message → kernel inputs. Duck-typed on message attributes; imports no ROS packages,
so it is unit-tested with plain fixture objects (same pattern as Dev 1 ros_bridge)."""

from __future__ import annotations

import math

from ugv_localization.camera import CalibrationError, CameraCalibration, calibration_from_camera_info, validate_intrinsics
from ugv_localization.common.checks import NS_PER_S, yaw_from_quaternion
from ugv_localization.odom import OdomSample


def stamp_to_ns(stamp: object) -> int:
    sec = stamp.sec  # type: ignore[attr-defined]
    nsec = stamp.nanosec  # type: ignore[attr-defined]
    if type(sec) is not int or type(nsec) is not int:
        raise TypeError("ROS stamp sec/nanosec must be Python ints")
    if sec < 0 or not 0 <= nsec < NS_PER_S:
        raise ValueError(f"invalid ROS stamp {sec}.{nsec:09d}")
    ns = sec * NS_PER_S + nsec
    if ns <= 0:
        raise ValueError("ROS stamp is zero (clock not started / unstamped message)")
    return ns


def ns_to_sec_nanosec(ns: int) -> tuple[int, int]:
    return divmod(int(ns), NS_PER_S)


def odom_msg_to_sample(msg: object) -> OdomSample:
    pose = msg.pose.pose  # type: ignore[attr-defined]
    p, q = pose.position, pose.orientation
    cov = tuple(float(v) for v in msg.pose.covariance)  # type: ignore[attr-defined]
    return OdomSample(
        stamp_ns=stamp_to_ns(msg.header.stamp),  # type: ignore[attr-defined]
        frame_id=str(msg.header.frame_id),  # type: ignore[attr-defined]
        child_frame_id=str(msg.child_frame_id),  # type: ignore[attr-defined]
        position=(float(p.x), float(p.y), float(p.z)),
        orientation=(float(q.x), float(q.y), float(q.z), float(q.w)),
        pose_covariance=cov,
    )


def odom_variances(msg: object) -> tuple[float, float, float]:
    """(var_x, var_y, var_yaw) from the 6x6 row-major pose covariance."""
    cov = msg.pose.covariance  # type: ignore[attr-defined]
    if len(cov) != 36:
        return (math.nan, math.nan, math.nan)
    return (float(cov[0]), float(cov[7]), float(cov[35]))


def camera_info_ok(msg: object) -> bool:
    """True if CameraInfo carries a plausible real calibration (no zero / fake K)."""
    try:
        validate_intrinsics(int(msg.width), int(msg.height), tuple(float(v) for v in msg.k))  # type: ignore[attr-defined]
    except (CalibrationError, TypeError, ValueError):
        return False
    return all(math.isfinite(float(v)) for v in msg.d)  # type: ignore[attr-defined]


def calibration_from_camera_info_msg(msg: object, camera_name: str) -> CameraCalibration:
    return calibration_from_camera_info(
        camera_name=camera_name,
        width=int(msg.width),  # type: ignore[attr-defined]
        height=int(msg.height),  # type: ignore[attr-defined]
        distortion_model=str(msg.distortion_model),  # type: ignore[attr-defined]
        d=tuple(msg.d),  # type: ignore[attr-defined]
        k=tuple(msg.k),  # type: ignore[attr-defined]
        r=tuple(msg.r),  # type: ignore[attr-defined]
        p=tuple(msg.p),  # type: ignore[attr-defined]
    )


def slam_info_visual_constraint(msg: object) -> bool:
    """rtabmap_msgs/Info: the current frame was matched to the map (loop closure, proximity
    detection, or landmark). Missing fields (older releases) count as 0."""
    ids = (
        getattr(msg, "loop_closure_id", 0),
        getattr(msg, "proximity_detection_id", 0),
        getattr(msg, "landmark_id", 0),
    )
    return any(int(v) > 0 for v in ids)


def transform_to_xy_yaw(transform: object) -> tuple[float, float, float]:
    t = transform.translation  # type: ignore[attr-defined]
    q = transform.rotation  # type: ignore[attr-defined]
    return float(t.x), float(t.y), yaw_from_quaternion(float(q.x), float(q.y), float(q.z), float(q.w))
