"""Tests for costmap_core.pipeline.run_costmap_pipeline -- the single entry
point that runs every costmap stage on a validated CostmapCoreInputs.

EVERY camera, grid, frame name, stamp, inflation and footprint value below
is SYNTHETIC test data (same hand-derivable setup as
test_costmap_pipeline.py). None of them are real Dev 1 / Dev 2 / Dev 5
parameters.
"""

import math

import numpy as np
import pytest

from costmap_core.class_to_cost import DEFAULT_COST_VALUES, CostValues, SemanticClass
from costmap_core.contracts import (
    CameraGroundInput,
    CameraIntrinsicsInput,
    CostmapCoreInputs,
    FootprintInput,
    GridInput,
    OccupancyInput,
    SemanticMaskInput,
)
from costmap_core.costmap_fusion import fuse_costmaps
from costmap_core.footprint import FootprintError
from costmap_core.geometry_costmap import (
    DEFAULT_GEOMETRY_COST_VALUES,
    GeometryCostValues,
    build_geometry_costmap,
)
from costmap_core.grid import CostmapGridGeometry
from costmap_core.inflation import InflationError, inflate_costmap
from costmap_core.mask_projection import project_mask_to_costmap
from costmap_core.pipeline import PipelineError, run_costmap_pipeline
from costmap_core.projection import CameraGroundGeometry, CameraIntrinsics

UNKNOWN_COST = DEFAULT_COST_VALUES.unknown_cost
FREE_COST = DEFAULT_COST_VALUES.traversable_cost
LETHAL_COST = DEFAULT_GEOMETRY_COST_VALUES.lethal_cost

UNKNOWN = SemanticClass.UNKNOWN.value
TRAVERSABLE = SemanticClass.TRAVERSABLE.value
HAZARD = SemanticClass.HAZARD.value

# Downward camera 1 m up, fx = fy = 10, 20x20 image over a 20x20 grid of
# 0.1 m cells at (-1, -1): pixel (u, v) lands in cell (row=19-u, col=19-v).
SIZE = 20
RESOLUTION = 0.1
CAM = "test_camera_frame"
GROUND = "test_ground_frame"
ROBOT = "test_robot_frame"
STAMP = 1_000_000_000
INTRINSICS = CameraIntrinsics(fx=10.0, fy=10.0, cx=9.5, cy=9.5)
DOWNWARD_CAMERA = CameraGroundGeometry(camera_height=1.0, pitch_rad=math.pi / 2)
GRID = CostmapGridGeometry(resolution=RESOLUTION, origin_x=-1.0, origin_y=-1.0, width=SIZE, height=SIZE)
INFLATION_RADIUS = 0.3
SYNTHETIC_FOOTPRINT = ((0.2, 0.15), (-0.2, 0.15), (-0.2, -0.15), (0.2, -0.15))


def cell_to_pixel(row, col):
    """Mask index (v, u) of the pixel that lands in grid cell (row, col)."""
    return SIZE - 1 - col, SIZE - 1 - row


def make_inputs(classes, occupied=None, footprint=None, valid=True, grid=GRID):
    return CostmapCoreInputs(
        mask=SemanticMaskInput(classes=classes, stamp_ns=STAMP, frame_id=CAM, valid=valid),
        intrinsics=CameraIntrinsicsInput(
            intrinsics=INTRINSICS, image_width=SIZE, image_height=SIZE, frame_id=CAM
        ),
        camera_ground=CameraGroundInput(
            geometry=DOWNWARD_CAMERA, camera_frame_id=CAM, ground_frame_id=GROUND, stamp_ns=STAMP
        ),
        grid=GridInput(geometry=grid, frame_id=GROUND),
        occupancy=(
            None
            if occupied is None
            else OccupancyInput(occupied=occupied, stamp_ns=STAMP, frame_id=GROUND)
        ),
        footprint=None if footprint is None else FootprintInput(vertices=footprint, frame_id=ROBOT),
    )


def build_scenario():
    classes = np.full((SIZE, SIZE), TRAVERSABLE, dtype=np.uint8)
    classes[cell_to_pixel(3, 3)] = HAZARD
    classes[cell_to_pixel(3, 16)] = UNKNOWN
    classes[cell_to_pixel(16, 16)] = HAZARD
    occupied = np.zeros((SIZE, SIZE), dtype=bool)
    occupied[10, 10] = True  # under semantic traversable
    occupied[16, 16] = True  # under semantic hazard
    return classes, occupied


# --- Equivalence with the manual stage chain ---------------------------------


def test_matches_manual_stage_chain_with_geometry():
    classes, occupied = build_scenario()
    result = run_costmap_pipeline(make_inputs(classes, occupied), inflation_radius=INFLATION_RADIUS)

    semantic = project_mask_to_costmap(classes, INTRINSICS, DOWNWARD_CAMERA, GRID)
    geometry = build_geometry_costmap(occupied)
    fused = fuse_costmaps(semantic, geometry)
    final = inflate_costmap(fused, RESOLUTION, INFLATION_RADIUS)

    assert np.array_equal(result.semantic, semantic)
    assert np.array_equal(result.geometry, geometry)
    assert np.array_equal(result.fused, fused)
    assert np.array_equal(result.final, final)
    for array in (result.semantic, result.geometry, result.fused, result.final):
        assert array.shape == (SIZE, SIZE)
        assert array.dtype == np.int64


def test_without_geometry_matches_chain_with_all_free_occupancy():
    classes, _ = build_scenario()
    result = run_costmap_pipeline(make_inputs(classes), inflation_radius=INFLATION_RADIUS)
    with_free_geometry = run_costmap_pipeline(
        make_inputs(classes, np.zeros((SIZE, SIZE), dtype=bool)),
        inflation_radius=INFLATION_RADIUS,
    )

    assert result.geometry is None
    assert np.array_equal(result.fused, result.semantic)
    assert np.array_equal(result.final, with_free_geometry.final)


# --- Precedence rules ----------------------------------------------------------


def test_geometry_lethal_wins_over_semantic_traversable():
    classes = np.full((SIZE, SIZE), TRAVERSABLE, dtype=np.uint8)
    occupied = np.zeros((SIZE, SIZE), dtype=bool)
    occupied[12, 2:9] = True

    result = run_costmap_pipeline(make_inputs(classes, occupied), inflation_radius=INFLATION_RADIUS)

    assert np.all(result.semantic == FREE_COST)
    assert np.array_equal(result.final == LETHAL_COST, occupied)


def test_geometry_lethal_wins_over_semantic_unknown():
    classes = np.full((SIZE, SIZE), UNKNOWN, dtype=np.uint8)
    occupied = np.zeros((SIZE, SIZE), dtype=bool)
    occupied[5, 5] = True

    result = run_costmap_pipeline(make_inputs(classes, occupied), inflation_radius=0.0)

    assert result.semantic[5, 5] == UNKNOWN_COST
    assert result.final[5, 5] == LETHAL_COST


def test_semantic_class_precedence_in_shared_cell():
    # 0.2 m synthetic grid: pixels (u=0, v=0) and (u=1, v=0) share cell (9, 9).
    coarse = CostmapGridGeometry(resolution=0.2, origin_x=-1.0, origin_y=-1.0, width=10, height=10)

    classes = np.full((SIZE, SIZE), TRAVERSABLE, dtype=np.uint8)
    classes[0, 0], classes[0, 1] = HAZARD, UNKNOWN
    hazard_vs_unknown = run_costmap_pipeline(make_inputs(classes, grid=coarse), inflation_radius=0.0)

    classes = np.full((SIZE, SIZE), TRAVERSABLE, dtype=np.uint8)
    classes[0, 1] = UNKNOWN
    unknown_vs_traversable = run_costmap_pipeline(make_inputs(classes, grid=coarse), inflation_radius=0.0)

    assert hazard_vs_unknown.final[9, 9] == LETHAL_COST
    assert unknown_vs_traversable.final[9, 9] == UNKNOWN_COST


def test_unknown_and_unseen_cells_are_never_free():
    classes = np.full((SIZE, SIZE), TRAVERSABLE, dtype=np.uint8)
    classes[cell_to_pixel(3, 16)] = UNKNOWN
    result = run_costmap_pipeline(make_inputs(classes), inflation_radius=INFLATION_RADIUS)
    assert result.final[3, 16] == UNKNOWN_COST

    level = CameraGroundGeometry(camera_height=1.0, pitch_rad=0.0)
    inputs = make_inputs(classes)
    unseen = CostmapCoreInputs(
        mask=inputs.mask,
        intrinsics=inputs.intrinsics,
        camera_ground=CameraGroundInput(
            geometry=level, camera_frame_id=CAM, ground_frame_id=GROUND, stamp_ns=STAMP
        ),
        grid=inputs.grid,
    )
    assert np.all(run_costmap_pipeline(unseen, inflation_radius=INFLATION_RADIUS).final == UNKNOWN_COST)


def test_inflation_is_applied_and_never_lowers_cost():
    classes, occupied = build_scenario()
    result = run_costmap_pipeline(make_inputs(classes, occupied), inflation_radius=INFLATION_RADIUS)

    assert np.all(result.final >= result.fused)
    assert np.array_equal(result.final == LETHAL_COST, result.fused == LETHAL_COST)
    assert 0 < result.final[10, 11] < LETHAL_COST  # next to geometric lethal
    assert 0 < result.final[4, 3] < LETHAL_COST  # next to semantic hazard


def test_zero_inflation_radius_leaves_fused_unchanged():
    classes, occupied = build_scenario()
    result = run_costmap_pipeline(make_inputs(classes, occupied), inflation_radius=0.0)
    assert np.array_equal(result.final, result.fused)


def test_custom_lethal_cost_is_used_for_fusion_and_inflation():
    classes = np.full((SIZE, SIZE), TRAVERSABLE, dtype=np.uint8)
    classes[cell_to_pixel(3, 3)] = HAZARD
    occupied = np.zeros((SIZE, SIZE), dtype=bool)
    occupied[10, 10] = True

    result = run_costmap_pipeline(
        make_inputs(classes, occupied),
        inflation_radius=INFLATION_RADIUS,
        cost_values=CostValues(hazard_cost=200),
        geometry_cost_values=GeometryCostValues(lethal_cost=200),
    )

    assert result.final[10, 10] == 200
    assert 0 < result.final[10, 11] < 200
    # Semantic hazard carries the custom value through every stage and is
    # inflated like a geometric lethal cell.
    assert result.semantic[3, 3] == result.fused[3, 3] == result.final[3, 3] == 200
    assert 0 < result.final[4, 3] < 200
    assert not np.any(result.final == LETHAL_COST)


@pytest.mark.parametrize(
    "cost_values, geometry_cost_values",
    [
        (CostValues(hazard_cost=200), DEFAULT_GEOMETRY_COST_VALUES),
        (DEFAULT_COST_VALUES, GeometryCostValues(lethal_cost=200)),
        (CostValues(hazard_cost=253), GeometryCostValues(lethal_cost=254)),
    ],
)
def test_inconsistent_hazard_and_lethal_cost_is_rejected(cost_values, geometry_cost_values):
    classes, occupied = build_scenario()
    for inputs in (make_inputs(classes), make_inputs(classes, occupied)):
        with pytest.raises(PipelineError, match="hazard_cost"):
            run_costmap_pipeline(
                inputs,
                inflation_radius=INFLATION_RADIUS,
                cost_values=cost_values,
                geometry_cost_values=geometry_cost_values,
            )


# --- Footprint -----------------------------------------------------------------


def test_footprint_absent_gives_none():
    classes, _ = build_scenario()
    assert run_costmap_pipeline(make_inputs(classes), inflation_radius=0.0).footprint is None


def test_footprint_passed_through_unpadded_and_does_not_change_costs():
    classes, occupied = build_scenario()
    without = run_costmap_pipeline(make_inputs(classes, occupied), inflation_radius=INFLATION_RADIUS)
    result = run_costmap_pipeline(
        make_inputs(classes, occupied, footprint=SYNTHETIC_FOOTPRINT),
        inflation_radius=INFLATION_RADIUS,
    )
    assert result.footprint == SYNTHETIC_FOOTPRINT
    assert np.array_equal(result.final, without.final)


def test_footprint_padding_is_applied():
    classes, _ = build_scenario()
    result = run_costmap_pipeline(
        make_inputs(classes, footprint=SYNTHETIC_FOOTPRINT),
        inflation_radius=0.0,
        footprint_padding=0.05,
    )
    assert isinstance(result.footprint, tuple)
    assert result.footprint == pytest.approx([(0.25, 0.2), (-0.25, 0.2), (-0.25, -0.2), (0.25, -0.2)])


def test_padding_without_footprint_is_rejected():
    classes, _ = build_scenario()
    with pytest.raises(PipelineError, match="footprint_padding was given"):
        run_costmap_pipeline(make_inputs(classes), inflation_radius=0.0, footprint_padding=0.05)


@pytest.mark.parametrize("padding", [-0.1, float("nan"), True])
def test_invalid_footprint_padding_is_rejected(padding):
    classes, _ = build_scenario()
    with pytest.raises(FootprintError):
        run_costmap_pipeline(
            make_inputs(classes, footprint=SYNTHETIC_FOOTPRINT),
            inflation_radius=0.0,
            footprint_padding=padding,
        )


# --- Input validation ------------------------------------------------------------


def test_invalid_mask_is_refused():
    classes, _ = build_scenario()
    with pytest.raises(PipelineError, match=r"mask\.valid is False"):
        run_costmap_pipeline(make_inputs(classes, valid=False), inflation_radius=INFLATION_RADIUS)


@pytest.mark.parametrize("inputs", [None, {"mask": None}, np.zeros((SIZE, SIZE), dtype=np.uint8)])
def test_non_contract_inputs_are_rejected(inputs):
    with pytest.raises(PipelineError, match="must be a CostmapCoreInputs"):
        run_costmap_pipeline(inputs, inflation_radius=INFLATION_RADIUS)


def test_inflation_radius_has_no_default():
    classes, _ = build_scenario()
    with pytest.raises(TypeError):
        run_costmap_pipeline(make_inputs(classes))


@pytest.mark.parametrize("radius", [-0.1, float("nan"), float("inf"), True, "0.3"])
def test_invalid_inflation_radius_is_rejected(radius):
    classes, _ = build_scenario()
    with pytest.raises(InflationError):
        run_costmap_pipeline(make_inputs(classes), inflation_radius=radius)


# --- Immutability ----------------------------------------------------------------


def test_result_arrays_are_read_only():
    classes, occupied = build_scenario()
    result = run_costmap_pipeline(make_inputs(classes, occupied), inflation_radius=INFLATION_RADIUS)
    for array in (result.semantic, result.geometry, result.fused, result.final):
        with pytest.raises(ValueError):
            array[0, 0] = 1
    with pytest.raises(AttributeError):
        result.final = None


def test_inputs_are_not_mutated():
    classes, occupied = build_scenario()
    classes_before, occupied_before = classes.copy(), occupied.copy()
    inputs = make_inputs(classes, occupied, footprint=SYNTHETIC_FOOTPRINT)

    run_costmap_pipeline(inputs, inflation_radius=INFLATION_RADIUS, footprint_padding=0.05)

    assert np.array_equal(classes, classes_before)
    assert np.array_equal(occupied, occupied_before)
    assert np.array_equal(inputs.mask.classes, classes_before)
    assert np.array_equal(inputs.occupancy.occupied, occupied_before)
    assert inputs.footprint.vertices == SYNTHETIC_FOOTPRINT
