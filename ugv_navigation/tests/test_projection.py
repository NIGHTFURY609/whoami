import math

import numpy as np
import pytest

from costmap_core.projection import (
    CameraGroundGeometry,
    CameraIntrinsics,
    GroundPoint,
    ProjectionError,
    camera_ray_to_ground_point,
    pixel_to_camera_ray,
    project_pixel_to_ground,
)

# Test/example-only values. These are NOT real camera calibration and must
# never be treated as such outside this test file.
INTRINSICS = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)


def test_pixel_to_camera_ray_at_principal_point():
    ray = pixel_to_camera_ray(INTRINSICS.cx, INTRINSICS.cy, INTRINSICS)
    assert ray == pytest.approx([0.0, 0.0, 1.0])


def test_pixel_to_camera_ray_offset_pixel():
    ray = pixel_to_camera_ray(420.0, 240.0, INTRINSICS)
    # x_cam = (420 - 320) / 500 = 0.2
    assert ray == pytest.approx([0.2, 0.0, 1.0])


def test_pixel_to_camera_ray_rejects_non_finite_pixel():
    with pytest.raises(ProjectionError):
        pixel_to_camera_ray(float("nan"), 240.0, INTRINSICS)
    with pytest.raises(ProjectionError):
        pixel_to_camera_ray(320.0, float("inf"), INTRINSICS)


def test_camera_intrinsics_rejects_invalid_focal_lengths():
    with pytest.raises(ProjectionError):
        CameraIntrinsics(fx=0.0, fy=500.0, cx=320.0, cy=240.0)
    with pytest.raises(ProjectionError):
        CameraIntrinsics(fx=500.0, fy=-10.0, cx=320.0, cy=240.0)


def test_camera_ground_geometry_rejects_invalid_camera_height():
    with pytest.raises(ProjectionError):
        CameraGroundGeometry(camera_height=0.0, pitch_rad=0.0)
    with pytest.raises(ProjectionError):
        CameraGroundGeometry(camera_height=-1.0, pitch_rad=0.0)


def test_center_pixel_at_zero_pitch_is_parallel_to_ground():
    geometry = CameraGroundGeometry(camera_height=2.0, pitch_rad=0.0)
    ray = pixel_to_camera_ray(INTRINSICS.cx, INTRINSICS.cy, INTRINSICS)
    with pytest.raises(ProjectionError):
        camera_ray_to_ground_point(ray, geometry)


def test_ray_pointing_upward_raises():
    geometry = CameraGroundGeometry(camera_height=2.0, pitch_rad=math.radians(-10.0))
    ray = pixel_to_camera_ray(INTRINSICS.cx, INTRINSICS.cy, INTRINSICS)
    with pytest.raises(ProjectionError):
        camera_ray_to_ground_point(ray, geometry)


def test_center_pixel_downward_pitch_hits_expected_ground_distance():
    geometry = CameraGroundGeometry(camera_height=2.0, pitch_rad=math.radians(30.0))
    ray = pixel_to_camera_ray(INTRINSICS.cx, INTRINSICS.cy, INTRINSICS)
    point = camera_ray_to_ground_point(ray, geometry)
    expected_x = geometry.camera_height / math.tan(geometry.pitch_rad)
    assert point.x == pytest.approx(expected_x)
    assert point.y == pytest.approx(0.0, abs=1e-9)


def test_offset_pixel_downward_pitch_lateral_sign():
    geometry = CameraGroundGeometry(camera_height=2.0, pitch_rad=math.radians(30.0))

    right_point = project_pixel_to_ground(420.0, 240.0, INTRINSICS, geometry)
    left_point = project_pixel_to_ground(220.0, 240.0, INTRINSICS, geometry)

    # Pixel to the right of the principal point should land to the world's
    # right (negative y, since world y is "left"); pixel to the left should
    # land to the world's left (positive y).
    assert right_point.y < 0.0
    assert left_point.y > 0.0


def test_project_pixel_to_ground_matches_manual_numeric_expectation():
    geometry = CameraGroundGeometry(camera_height=2.0, pitch_rad=math.radians(30.0))
    point = project_pixel_to_ground(420.0, 240.0, INTRINSICS, geometry)

    # x_cam = 0.2, t = height / sin(pitch) = 2.0 / 0.5 = 4.0
    # ground_x = t * cos(pitch) = 4.0 * cos(30deg)
    # ground_y = -x_cam * t = -0.2 * 4.0
    assert point.x == pytest.approx(4.0 * math.cos(math.radians(30.0)))
    assert point.y == pytest.approx(-0.8)


def test_project_pixel_to_ground_matches_two_step_pipeline():
    geometry = CameraGroundGeometry(camera_height=1.5, pitch_rad=math.radians(45.0))

    combined = project_pixel_to_ground(350.0, 260.0, INTRINSICS, geometry)

    ray = pixel_to_camera_ray(350.0, 260.0, INTRINSICS)
    two_step = camera_ray_to_ground_point(ray, geometry)

    assert combined == two_step


def test_ground_point_is_plain_dataclass_with_x_y():
    point = GroundPoint(x=1.0, y=2.0)
    assert point.x == 1.0
    assert point.y == 2.0
