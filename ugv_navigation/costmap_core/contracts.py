"""ROS-independent input contract for the Dev 3 costmap core.

This module is the boundary between future ROS adapters (subscribers,
TF lookups, CameraInfo parsing, parameter loading) and the existing
costmap algorithms (`mask_projection`, `geometry_costmap`,
`costmap_fusion`, `inflation`, `footprint`). An adapter's only job is to
fill these dataclasses; the algorithms never see ROS types.

It deliberately:
- holds NO defaults for real-world values (camera intrinsics, camera
  mounting, grid resolution/extent, frame names, footprint, max age) --
  PROJECT_CONTEXT.md §5/§6 forbid inventing them;
- reuses the existing core value types (`CameraIntrinsics`,
  `CameraGroundGeometry`, `CostmapGridGeometry`, `validate_footprint`)
  instead of redefining them;
- adds only what the core types lack: frame ids, stamps, validity flags,
  and cross-input consistency checks (frame ids agree, shapes agree);
- does NOT import Dev 1's `ugv_perception` package. The mask fields
  mirror Dev 1's published port (`/segmentation/mask`, mono8, header
  stamp/frame_id, port_meta `valid`) but Dev 3 stays decoupled from Dev 1's
  implementation (PROJECT_CONTEXT.md §7).

See costmap_core/README.md for which inputs are available today, which are
synthetic, and which still depend on Dev 2 / Dev 5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from costmap_core.class_to_cost import SemanticClass
from costmap_core.footprint import FootprintError, Point, validate_footprint
from costmap_core.grid import CostmapGridGeometry
from costmap_core.projection import CameraGroundGeometry, CameraIntrinsics

_NS_PER_S = 1_000_000_000
_CANONICAL_IDS = frozenset(int(c) for c in SemanticClass)


class ContractError(ValueError):
    """Raised when an input violates the Dev 3 core contract.

    Deliberately never coerced: a malformed or inconsistent input must fail
    loudly rather than silently produce a costmap with undefined meaning.
    """


# --- field validators --------------------------------------------------------


def _require_stamp(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ContractError(f"{name} must be an integer (ns), got {type(value).__name__}")
    value = int(value)
    if value <= 0:
        raise ContractError(f"{name} must be > 0 ns, got {value}")
    return value


def _require_frame_id(value: object, *, name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ContractError(f"{name} must be a non-empty str, got {value!r}")
    return value


def _require_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _require_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{name} must be a positive int, got {value!r}")
    return value


def _require_type(value: object, expected: type, *, name: str) -> None:
    if not isinstance(value, expected):
        raise ContractError(
            f"{name} must be a {expected.__name__}, got {type(value).__name__}"
        )


def _readonly_copy(array: np.ndarray) -> np.ndarray:
    copy = np.array(array, copy=True)
    copy.setflags(write=False)
    return copy


# --- individual inputs -------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SemanticMaskInput:
    """One semantic mask from Dev 1's perception port.

    Source (Dev 1, available): `/segmentation/mask` (sensor_msgs/Image, mono8)
    plus `valid` from `/segmentation/port_meta`.

    classes: 2D uint8 array, every pixel in {0 unknown, 1 traversable,
        2 hazard}. Pixel (row, col) is image coordinate (v=row, u=col).
        Stored as a read-only copy.
    stamp_ns: image capture time (header.stamp), ns.
    frame_id: camera optical frame of the mask (header.frame_id).
    valid: Dev 1's port validity flag. Carried, not acted on, here.
    """

    classes: np.ndarray
    stamp_ns: int
    frame_id: str
    valid: bool

    def __post_init__(self) -> None:
        classes = self.classes
        if not isinstance(classes, np.ndarray):
            raise ContractError(f"classes must be a numpy ndarray, got {type(classes).__name__}")
        if classes.dtype != np.uint8:
            raise ContractError(f"classes dtype must be uint8 (mono8), got {classes.dtype}")
        if classes.ndim != 2 or classes.size == 0:
            raise ContractError(f"classes must be a non-empty 2D array, got shape {classes.shape}")
        bad = set(np.unique(classes).tolist()) - _CANONICAL_IDS
        if bad:
            raise ContractError(
                f"classes contains non-canonical ids {sorted(bad)}; "
                f"expected only {sorted(_CANONICAL_IDS)}"
            )
        object.__setattr__(self, "classes", _readonly_copy(classes))
        object.__setattr__(self, "stamp_ns", _require_stamp(self.stamp_ns, name="stamp_ns"))
        object.__setattr__(self, "frame_id", _require_frame_id(self.frame_id, name="frame_id"))
        _require_bool(self.valid, name="valid")

    @property
    def shape(self) -> tuple[int, int]:
        return self.classes.shape


@dataclass(frozen=True)
class CameraIntrinsicsInput:
    """Pinhole intrinsics of the camera that produced the mask.

    Source (eventually): sensor_msgs/CameraInfo (K, width, height,
    header.frame_id). Dev 1 republishes it on `/segmentation/camera_info`;
    the camera driver itself is owned by Dev 5. No real values exist yet.

    intrinsics: fx, fy, cx, cy in pixels (validated by CameraIntrinsics).
    image_width, image_height: image size the intrinsics refer to.
    frame_id: camera optical frame the intrinsics belong to.

    Distortion (CameraInfo D / distortion_model) is intentionally absent:
    the current projection core is pure pinhole. See README limitations.
    """

    intrinsics: CameraIntrinsics
    image_width: int
    image_height: int
    frame_id: str

    def __post_init__(self) -> None:
        _require_type(self.intrinsics, CameraIntrinsics, name="intrinsics")
        _require_positive_int(self.image_width, name="image_width")
        _require_positive_int(self.image_height, name="image_height")
        _require_frame_id(self.frame_id, name="frame_id")


@dataclass(frozen=True)
class CameraGroundInput:
    """Camera pose relative to the ground plane the costmap lies on.

    Source (eventually): TF, camera optical frame -> costmap frame. Dev 2
    owns map -> odom -> base_link; the base_link -> camera extrinsic comes
    from Dev 5's robot description. Neither is available yet.

    geometry: the core's simplified pose (height above ground, downward
        pitch; roll = yaw = 0; camera directly above the ground frame origin,
        ground x along the camera's forward heading). See CameraGroundGeometry.
    camera_frame_id: optical frame the pose is for (must match the mask).
    ground_frame_id: frame the projected ground points are expressed in
        (must match the grid's frame).
    stamp_ns: time the pose refers to.
    """

    geometry: CameraGroundGeometry
    camera_frame_id: str
    ground_frame_id: str
    stamp_ns: int

    def __post_init__(self) -> None:
        _require_type(self.geometry, CameraGroundGeometry, name="geometry")
        _require_frame_id(self.camera_frame_id, name="camera_frame_id")
        _require_frame_id(self.ground_frame_id, name="ground_frame_id")
        object.__setattr__(self, "stamp_ns", _require_stamp(self.stamp_ns, name="stamp_ns"))


@dataclass(frozen=True)
class GridInput:
    """Costmap grid geometry plus the frame it is expressed in.

    Source: Dev 3 configuration (config/robots/). Resolution, extent and
    frame name are not decided yet and have no defaults.
    """

    geometry: CostmapGridGeometry
    frame_id: str

    def __post_init__(self) -> None:
        _require_type(self.geometry, CostmapGridGeometry, name="geometry")
        _require_frame_id(self.frame_id, name="frame_id")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.geometry.height, self.geometry.width)


@dataclass(frozen=True, eq=False)
class OccupancyInput:
    """Optional geometric occupancy already rasterised onto the costmap grid.

    Source (eventually): Depth Anything / VoxelLayer geometry side-channel
    (architecture §9). No producer or message format exists yet.

    occupied: 2D bool array, grid-shaped (row = y, col = x as in grid.py).
        True = geometrically occupied (lethal). Stored as a read-only copy.
    stamp_ns: time the occupancy observation refers to.
    frame_id: frame of the grid the occupancy is rasterised on.
    """

    occupied: np.ndarray
    stamp_ns: int
    frame_id: str

    def __post_init__(self) -> None:
        occupied = self.occupied
        if not isinstance(occupied, np.ndarray):
            raise ContractError(f"occupied must be a numpy ndarray, got {type(occupied).__name__}")
        if occupied.dtype != np.bool_:
            raise ContractError(f"occupied dtype must be bool, got {occupied.dtype}")
        if occupied.ndim != 2:
            raise ContractError(f"occupied must be 2D, got shape {occupied.shape}")
        object.__setattr__(self, "occupied", _readonly_copy(occupied))
        object.__setattr__(self, "stamp_ns", _require_stamp(self.stamp_ns, name="stamp_ns"))
        object.__setattr__(self, "frame_id", _require_frame_id(self.frame_id, name="frame_id"))


@dataclass(frozen=True)
class FootprintInput:
    """Robot footprint polygon, once Dev 5 provides one.

    vertices: convex polygon, (x, y) metres, validated and normalised by
        footprint.validate_footprint (stored as a tuple of float pairs).
    frame_id: robot frame the polygon is expressed in.
    """

    vertices: tuple[Point, ...]
    frame_id: str

    def __post_init__(self) -> None:
        try:
            points = validate_footprint(self.vertices)
        except FootprintError as exc:
            raise ContractError(f"invalid footprint: {exc}") from exc
        object.__setattr__(self, "vertices", tuple(points))
        _require_frame_id(self.frame_id, name="frame_id")


# --- bundle ------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class CostmapCoreInputs:
    """Everything one costmap update needs, checked for mutual consistency.

    Cross-checks (all fail with ContractError):
    - mask.frame_id == intrinsics.frame_id == camera_ground.camera_frame_id
    - mask shape == (intrinsics.image_height, intrinsics.image_width)
      (Dev 1's v1 port publishes masks at source resolution, scale 1.0)
    - camera_ground.ground_frame_id == grid.frame_id
    - if occupancy given: shape == grid shape, frame_id == grid.frame_id

    Not checked here (no agreed values yet -- see README): stamp alignment
    tolerance between inputs, and freshness limits. Use `age_s` / `is_fresh`
    with an explicitly supplied max age.
    """

    mask: SemanticMaskInput
    intrinsics: CameraIntrinsicsInput
    camera_ground: CameraGroundInput
    grid: GridInput
    occupancy: Optional[OccupancyInput] = None
    footprint: Optional[FootprintInput] = None

    def __post_init__(self) -> None:
        _require_type(self.mask, SemanticMaskInput, name="mask")
        _require_type(self.intrinsics, CameraIntrinsicsInput, name="intrinsics")
        _require_type(self.camera_ground, CameraGroundInput, name="camera_ground")
        _require_type(self.grid, GridInput, name="grid")
        if self.occupancy is not None:
            _require_type(self.occupancy, OccupancyInput, name="occupancy")
        if self.footprint is not None:
            _require_type(self.footprint, FootprintInput, name="footprint")

        camera_frames = {
            "mask.frame_id": self.mask.frame_id,
            "intrinsics.frame_id": self.intrinsics.frame_id,
            "camera_ground.camera_frame_id": self.camera_ground.camera_frame_id,
        }
        if len(set(camera_frames.values())) != 1:
            raise ContractError(f"camera frame ids disagree: {camera_frames}")

        expected_hw = (self.intrinsics.image_height, self.intrinsics.image_width)
        if self.mask.shape != expected_hw:
            raise ContractError(
                f"mask shape {self.mask.shape} does not match intrinsics image size "
                f"(height, width) = {expected_hw}"
            )

        if self.camera_ground.ground_frame_id != self.grid.frame_id:
            raise ContractError(
                f"camera_ground.ground_frame_id {self.camera_ground.ground_frame_id!r} "
                f"does not match grid.frame_id {self.grid.frame_id!r}"
            )

        if self.occupancy is not None:
            if self.occupancy.occupied.shape != self.grid.shape:
                raise ContractError(
                    f"occupancy shape {self.occupancy.occupied.shape} does not match "
                    f"grid shape (height, width) = {self.grid.shape}"
                )
            if self.occupancy.frame_id != self.grid.frame_id:
                raise ContractError(
                    f"occupancy.frame_id {self.occupancy.frame_id!r} does not match "
                    f"grid.frame_id {self.grid.frame_id!r}"
                )


# --- freshness ---------------------------------------------------------------


def age_s(stamp_ns: int, now_ns: int) -> float:
    """Age of an input in seconds: (now - stamp). Negative if stamped in the future."""
    stamp_ns = _require_stamp(stamp_ns, name="stamp_ns")
    now_ns = _require_stamp(now_ns, name="now_ns")
    return (now_ns - stamp_ns) / _NS_PER_S


def is_fresh(stamp_ns: int, now_ns: int, max_age_s: float) -> bool:
    """True iff 0 <= age <= max_age_s.

    `max_age_s` has no default on purpose: the value Dev 3 should use is not
    agreed (Dev 1's own `perception_max_age` lives in Dev 1 config). A
    future-stamped input (negative age) is never fresh.
    """
    if isinstance(max_age_s, bool) or not isinstance(max_age_s, (int, float)):
        raise ContractError(f"max_age_s must be a real number, got {type(max_age_s).__name__}")
    if not math.isfinite(max_age_s) or max_age_s <= 0:
        raise ContractError(f"max_age_s must be finite and > 0, got {max_age_s!r}")
    age = age_s(stamp_ns, now_ns)
    return 0.0 <= age <= max_age_s
