"""Tests for costmap_core.contracts -- the Dev 3 core input boundary.

EVERY value below (intrinsics, camera height/pitch, grid, frame names,
stamps, footprint, max age) is SYNTHETIC test data. None of them are real
Dev 1 / Dev 2 / Dev 5 parameters.
"""

import math

import numpy as np
import pytest

from costmap_core.contracts import (
    CameraGroundInput,
    CameraIntrinsicsInput,
    ContractError,
    CostmapCoreInputs,
    FootprintInput,
    GridInput,
    OccupancyInput,
    SemanticMaskInput,
    age_s,
    is_fresh,
)
from costmap_core.grid import CostmapGridGeometry
from costmap_core.mask_projection import project_mask_to_costmap
from costmap_core.projection import CameraGroundGeometry, CameraIntrinsics

CAM = "test_camera_frame"
GROUND = "test_ground_frame"
STAMP = 1_000_000_000
H, W = 4, 6


def make_mask(**overrides):
    kwargs = dict(
        classes=np.ones((H, W), dtype=np.uint8), stamp_ns=STAMP, frame_id=CAM, valid=True
    )
    kwargs.update(overrides)
    return SemanticMaskInput(**kwargs)


def make_intrinsics(**overrides):
    kwargs = dict(
        intrinsics=CameraIntrinsics(fx=10.0, fy=10.0, cx=2.5, cy=1.5),
        image_width=W,
        image_height=H,
        frame_id=CAM,
    )
    kwargs.update(overrides)
    return CameraIntrinsicsInput(**kwargs)


def make_camera_ground(**overrides):
    kwargs = dict(
        geometry=CameraGroundGeometry(camera_height=1.0, pitch_rad=math.pi / 2),
        camera_frame_id=CAM,
        ground_frame_id=GROUND,
        stamp_ns=STAMP,
    )
    kwargs.update(overrides)
    return CameraGroundInput(**kwargs)


def make_grid(**overrides):
    kwargs = dict(
        geometry=CostmapGridGeometry(
            resolution=0.1, origin_x=-1.0, origin_y=-1.0, width=20, height=10
        ),
        frame_id=GROUND,
    )
    kwargs.update(overrides)
    return GridInput(**kwargs)


def make_occupancy(**overrides):
    kwargs = dict(occupied=np.zeros((10, 20), dtype=bool), stamp_ns=STAMP, frame_id=GROUND)
    kwargs.update(overrides)
    return OccupancyInput(**kwargs)


def make_inputs(**overrides):
    kwargs = dict(
        mask=make_mask(),
        intrinsics=make_intrinsics(),
        camera_ground=make_camera_ground(),
        grid=make_grid(),
    )
    kwargs.update(overrides)
    return CostmapCoreInputs(**kwargs)


# --- SemanticMaskInput -------------------------------------------------------


def test_mask_accepts_canonical_uint8():
    mask = make_mask(classes=np.array([[0, 1, 2]], dtype=np.uint8))
    assert mask.shape == (1, 3)
    assert mask.valid is True


def test_mask_is_readonly_copy():
    source = np.ones((H, W), dtype=np.uint8)
    mask = make_mask(classes=source)
    source[0, 0] = 2
    assert mask.classes[0, 0] == 1
    with pytest.raises(ValueError):
        mask.classes[0, 0] = 2


@pytest.mark.parametrize(
    "classes",
    [
        np.ones((H, W), dtype=np.int64),  # not mono8
        np.ones((H, W, 1), dtype=np.uint8),  # not 2D
        np.zeros((0, 0), dtype=np.uint8),  # empty
        np.full((H, W), 3, dtype=np.uint8),  # non-canonical id
        [[1, 1]],  # not an ndarray
    ],
)
def test_mask_rejects_bad_classes(classes):
    with pytest.raises(ContractError):
        make_mask(classes=classes)


@pytest.mark.parametrize("stamp", [0, -1, 1.5, True, "1"])
def test_mask_rejects_bad_stamp(stamp):
    with pytest.raises(ContractError):
        make_mask(stamp_ns=stamp)


@pytest.mark.parametrize("frame_id", ["", None, 5])
def test_mask_rejects_bad_frame_id(frame_id):
    with pytest.raises(ContractError):
        make_mask(frame_id=frame_id)


@pytest.mark.parametrize("valid", [1, 0, None, np.bool_(True)])
def test_mask_rejects_non_bool_valid(valid):
    with pytest.raises(ContractError):
        make_mask(valid=valid)


def test_mask_carries_invalid_flag_without_rejecting():
    assert make_mask(valid=False).valid is False


# --- other individual inputs -------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        dict(intrinsics=(10.0, 10.0, 2.5, 1.5)),
        dict(image_width=0),
        dict(image_height=-1),
        dict(image_width=6.0),
        dict(image_height=True),
        dict(frame_id=""),
    ],
)
def test_intrinsics_rejects_bad_fields(overrides):
    with pytest.raises(ContractError):
        make_intrinsics(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(geometry=(1.0, 0.5)),
        dict(camera_frame_id=""),
        dict(ground_frame_id=""),
        dict(stamp_ns=0),
    ],
)
def test_camera_ground_rejects_bad_fields(overrides):
    with pytest.raises(ContractError):
        make_camera_ground(**overrides)


@pytest.mark.parametrize("overrides", [dict(geometry=None), dict(frame_id="")])
def test_grid_rejects_bad_fields(overrides):
    with pytest.raises(ContractError):
        make_grid(**overrides)


def test_grid_shape_is_height_width():
    assert make_grid().shape == (10, 20)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(occupied=np.zeros((10, 20), dtype=np.uint8)),  # 0/1 ints not accepted
        dict(occupied=np.zeros((10,), dtype=bool)),
        dict(occupied=[[False]]),
        dict(stamp_ns=-5),
        dict(frame_id=""),
    ],
)
def test_occupancy_rejects_bad_fields(overrides):
    with pytest.raises(ContractError):
        make_occupancy(**overrides)


def test_occupancy_is_readonly_copy():
    source = np.zeros((10, 20), dtype=bool)
    occ = make_occupancy(occupied=source)
    source[0, 0] = True
    assert not occ.occupied[0, 0]
    with pytest.raises(ValueError):
        occ.occupied[0, 0] = True


def test_footprint_normalises_vertices():
    fp = FootprintInput(vertices=[[1, 1], [-1, 1], [-1, -1], [1, -1]], frame_id="test_robot")
    assert fp.vertices == ((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0))


@pytest.mark.parametrize(
    "vertices",
    [
        [(0, 0), (1, 0)],  # too few
        [(0, 0), (2, 0), (1, 1), (2, 2), (0, 2)],  # concave
    ],
)
def test_footprint_rejects_invalid_polygon(vertices):
    with pytest.raises(ContractError):
        FootprintInput(vertices=vertices, frame_id="test_robot")


def test_footprint_rejects_empty_frame():
    with pytest.raises(ContractError):
        FootprintInput(vertices=[(1, 1), (-1, 1), (-1, -1)], frame_id="")


# --- CostmapCoreInputs cross-checks ------------------------------------------


def test_bundle_accepts_consistent_inputs():
    fp = FootprintInput(vertices=[(1, 1), (-1, 1), (-1, -1), (1, -1)], frame_id="test_robot")
    inputs = make_inputs(occupancy=make_occupancy(), footprint=fp)
    assert inputs.occupancy is not None and inputs.footprint is fp


def test_bundle_optional_inputs_default_to_none():
    inputs = make_inputs()
    assert inputs.occupancy is None
    assert inputs.footprint is None


@pytest.mark.parametrize(
    "overrides",
    [
        dict(mask=make_mask(frame_id="other")),
        dict(intrinsics=make_intrinsics(frame_id="other")),
        dict(camera_ground=make_camera_ground(camera_frame_id="other")),
    ],
)
def test_bundle_rejects_camera_frame_mismatch(overrides):
    with pytest.raises(ContractError, match="camera frame ids disagree"):
        make_inputs(**overrides)


def test_bundle_rejects_mask_size_mismatch():
    with pytest.raises(ContractError, match="mask shape"):
        make_inputs(intrinsics=make_intrinsics(image_width=W + 1))


def test_bundle_rejects_ground_grid_frame_mismatch():
    with pytest.raises(ContractError, match="ground_frame_id"):
        make_inputs(grid=make_grid(frame_id="other"))


def test_bundle_rejects_occupancy_shape_mismatch():
    with pytest.raises(ContractError, match="occupancy shape"):
        make_inputs(occupancy=make_occupancy(occupied=np.zeros((20, 10), dtype=bool)))


def test_bundle_rejects_occupancy_frame_mismatch():
    with pytest.raises(ContractError, match="occupancy.frame_id"):
        make_inputs(occupancy=make_occupancy(frame_id="other"))


def test_bundle_rejects_wrong_types():
    with pytest.raises(ContractError):
        make_inputs(mask=np.ones((H, W), dtype=np.uint8))
    with pytest.raises(ContractError):
        make_inputs(occupancy=np.zeros((10, 20), dtype=bool))


def test_bundle_feeds_existing_projection_unchanged():
    # The contract only packages inputs; the algorithm is called as before.
    inputs = make_inputs()
    costmap = project_mask_to_costmap(
        inputs.mask.classes,
        inputs.intrinsics.intrinsics,
        inputs.camera_ground.geometry,
        inputs.grid.geometry,
    )
    assert costmap.shape == inputs.grid.shape


# --- freshness ---------------------------------------------------------------


def test_age_s():
    assert age_s(STAMP, STAMP + 250_000_000) == pytest.approx(0.25)
    assert age_s(STAMP, STAMP - 1_000_000) < 0


def test_is_fresh_boundaries():
    assert is_fresh(STAMP, STAMP, 0.5)
    assert is_fresh(STAMP, STAMP + 500_000_000, 0.5)
    assert not is_fresh(STAMP, STAMP + 500_000_001, 0.5)


def test_future_stamp_is_not_fresh():
    assert not is_fresh(STAMP + 1, STAMP, 0.5)


@pytest.mark.parametrize("max_age", [0, -1.0, math.inf, math.nan, True, "0.5", None])
def test_is_fresh_rejects_bad_max_age(max_age):
    with pytest.raises(ContractError):
        is_fresh(STAMP, STAMP, max_age)


@pytest.mark.parametrize("stamp,now", [(0, STAMP), (STAMP, 0), (1.0, STAMP)])
def test_age_rejects_bad_stamps(stamp, now):
    with pytest.raises(ContractError):
        age_s(stamp, now)
