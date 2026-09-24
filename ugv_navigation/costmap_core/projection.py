"""ROS-independent pixel-to-ground projection core.

Implements the geometry-only part of:

    pixel (u, v)
        -> camera intrinsics
        -> 3D camera ray
        -> ground-plane intersection
        -> ground position

This module does not know about images, masks, or costmaps -- it only
answers "where on the ground does this pixel look?" for a single pixel at a
time. Combining this with a semantic mask (to project a whole mask onto a
costmap grid) is a separate, not-yet-implemented component.

No real camera calibration values are defined here (see PROJECT_CONTEXT.md
Section 5): `CameraIntrinsics` and `CameraGroundGeometry` have no default
values, so callers must supply real or test values explicitly.
"""

import math

import numpy as np
from dataclasses import dataclass


class ProjectionError(ValueError):
    """Raised when projection inputs or geometry make the result invalid.

    Covers: invalid intrinsics, invalid camera/ground geometry, non-finite
    pixel coordinates, and camera rays that do not intersect the ground
    plane (parallel to or diverging away from the ground).
    """


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics.

    fx, fy: focal lengths in pixels.
    cx, cy: principal point in pixels.

    No defaults are provided on purpose: these values must come from a real
    CameraInfo message or a configuration file, never be invented here.
    """

    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.fx) and self.fx > 0):
            raise ProjectionError(f"Invalid fx: {self.fx!r}. fx must be a finite value > 0.")
        if not (math.isfinite(self.fy) and self.fy > 0):
            raise ProjectionError(f"Invalid fy: {self.fy!r}. fy must be a finite value > 0.")


@dataclass(frozen=True)
class CameraGroundGeometry:
    """Camera pose relative to a flat ground plane.

    The ground plane is the world XY plane at z=0, in a right-handed world
    frame with x-forward, y-left, z-up (matching the map/odom/base_link
    convention). The camera is mounted at `camera_height` above the ground,
    directly above the world-frame origin, and pitched downward from
    horizontal by `pitch_rad` (0 = looking along the horizon).

    Roll and yaw are assumed zero. This is a deliberate simplification for
    this first projection core -- a full 6-DoF extrinsic pose is not
    implemented here.
    """

    camera_height: float
    pitch_rad: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.camera_height) and self.camera_height > 0):
            raise ProjectionError(
                f"Invalid camera_height: {self.camera_height!r}. "
                "camera_height must be a finite value > 0 (camera must be above the ground)."
            )
        if not math.isfinite(self.pitch_rad):
            raise ProjectionError(f"Invalid pitch_rad: {self.pitch_rad!r}. Must be finite.")


@dataclass(frozen=True)
class GroundPoint:
    """A point on the ground plane (z=0 implicit), in the world frame."""

    x: float
    y: float


# Maps a camera-optical-frame vector (x-right, y-down, z-forward) to the
# equivalent direction in the world frame (x-forward, y-left, z-up) when the
# camera has zero pitch: camera-forward (z) becomes world-forward (x),
# camera-right (x) becomes world-right (-y), camera-down (y) becomes
# world-down (-z).
_OPTICAL_TO_LEVEL_WORLD = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)

# Ray must point at least this far below horizontal (in world-frame z) to be
# treated as intersecting the ground plane. Guards against dividing by a
# near-zero value for rays that are only numerically, not meaningfully,
# downward.
_MIN_DOWNWARD_Z_COMPONENT = 1e-9


def pixel_to_camera_ray(u: float, v: float, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Convert a pixel coordinate into a 3D ray direction in the camera's
    optical frame (x-right, y-down, z-forward). The returned vector is not
    normalized -- only its direction matters for the ground intersection.

    Raises ProjectionError if u or v is not finite.
    """
    if not (math.isfinite(u) and math.isfinite(v)):
        raise ProjectionError(f"Invalid pixel coordinates: u={u!r}, v={v!r}. Both must be finite.")

    x_cam = (u - intrinsics.cx) / intrinsics.fx
    y_cam = (v - intrinsics.cy) / intrinsics.fy
    return np.array([x_cam, y_cam, 1.0])


def camera_ray_to_ground_point(
    ray_camera: np.ndarray, geometry: CameraGroundGeometry
) -> GroundPoint:
    """Intersect a camera-frame ray with the ground plane.

    The ray is transformed into the world frame using `geometry`'s pitch,
    then intersected with the z=0 plane.

    Raises ProjectionError if the world-frame ray does not point downward
    (i.e. it is parallel to, or diverges away from, the ground plane).
    """
    pitch = geometry.pitch_rad
    rot_pitch = np.array(
        [
            [math.cos(pitch), 0.0, math.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-math.sin(pitch), 0.0, math.cos(pitch)],
        ]
    )
    world_dir = rot_pitch @ (_OPTICAL_TO_LEVEL_WORLD @ ray_camera)

    if world_dir[2] >= -_MIN_DOWNWARD_Z_COMPONENT:
        raise ProjectionError(
            "Camera ray does not intersect the ground plane: world-frame "
            f"downward component is {world_dir[2]!r} (must be < 0). The ray "
            "is parallel to or diverging away from the ground."
        )

    t = -geometry.camera_height / world_dir[2]
    ground_x = t * world_dir[0]
    ground_y = t * world_dir[1]
    return GroundPoint(x=ground_x, y=ground_y)


def project_pixel_to_ground(
    u: float, v: float, intrinsics: CameraIntrinsics, geometry: CameraGroundGeometry
) -> GroundPoint:
    """Project a single pixel to a ground-plane point.

    Convenience wrapper chaining `pixel_to_camera_ray` and
    `camera_ray_to_ground_point`.
    """
    ray_camera = pixel_to_camera_ray(u, v, intrinsics)
    return camera_ray_to_ground_point(ray_camera, geometry)
