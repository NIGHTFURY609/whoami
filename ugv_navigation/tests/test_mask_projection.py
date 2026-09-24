import math

import numpy as np
import pytest

from costmap_core.class_to_cost import (
    DEFAULT_COST_VALUES,
    InvalidSemanticClassError,
    SemanticClass,
    class_to_cost,
)
from costmap_core.grid import CostmapGridGeometry, world_to_grid_cell
from costmap_core.mask_projection import project_mask_to_costmap
from costmap_core.projection import (
    CameraGroundGeometry,
    CameraIntrinsics,
    ProjectionError,
    project_pixel_to_ground,
)

# Test/example-only values. These are NOT real camera calibration and must
# never be treated as such outside this test file.
INTRINSICS = CameraIntrinsics(fx=100.0, fy=100.0, cx=2.0, cy=2.0)
GEOMETRY = CameraGroundGeometry(camera_height=2.0, pitch_rad=math.radians(30.0))
GRID = CostmapGridGeometry(resolution=1.0, origin_x=0.0, origin_y=-5.0, width=10, height=10)


def _expected_cell(u: float, v: float):
    """Ground truth cell for pixel (u, v), computed via the already-tested
    projection.py / grid.py primitives -- not re-derived by hand here."""
    point = project_pixel_to_ground(u, v, INTRINSICS, GEOMETRY)
    return world_to_grid_cell(point.x, point.y, GRID)


def test_traversable_mask_projects_successfully():
    mask = np.full((1, 1), SemanticClass.TRAVERSABLE, dtype=np.uint8)
    result = project_mask_to_costmap(mask, INTRINSICS, GEOMETRY, GRID)

    assert result.shape == (GRID.height, GRID.width)
    cell = _expected_cell(0.0, 0.0)
    assert result[cell.row, cell.col] == class_to_cost(SemanticClass.TRAVERSABLE)


def test_hazard_pixel_produces_hazard_cost():
    mask = np.full((1, 1), SemanticClass.HAZARD, dtype=np.uint8)
    result = project_mask_to_costmap(mask, INTRINSICS, GEOMETRY, GRID)

    cell = _expected_cell(0.0, 0.0)
    assert result[cell.row, cell.col] == class_to_cost(SemanticClass.HAZARD)


def test_unknown_pixel_produces_unknown_cost():
    mask = np.full((1, 1), SemanticClass.UNKNOWN, dtype=np.uint8)
    result = project_mask_to_costmap(mask, INTRINSICS, GEOMETRY, GRID)

    cell = _expected_cell(0.0, 0.0)
    assert result[cell.row, cell.col] == class_to_cost(SemanticClass.UNKNOWN)


def test_untouched_cells_default_to_unknown_cost():
    # A single traversable pixel only ever touches one cell; every other
    # cell must keep the "no data reached here" default.
    mask = np.full((1, 1), SemanticClass.TRAVERSABLE, dtype=np.uint8)
    result = project_mask_to_costmap(mask, INTRINSICS, GEOMETRY, GRID)

    cell = _expected_cell(0.0, 0.0)
    untouched = result.copy()
    untouched[cell.row, cell.col] = DEFAULT_COST_VALUES.unknown_cost
    assert np.all(untouched == DEFAULT_COST_VALUES.unknown_cost)


def test_multiple_pixels_mapping_to_same_cell_use_highest_cost():
    # Pixels (0,0) and (1,1) are both close to the principal point (2,2)
    # with this focal length, so at 1.0 m grid resolution they land in the
    # same cell (confirmed below rather than assumed).
    center_cell = _expected_cell(1.0, 1.0)
    corner_cell = _expected_cell(0.0, 0.0)
    assert center_cell == corner_cell

    mask = np.full((2, 2), SemanticClass.TRAVERSABLE, dtype=np.uint8)
    mask[1, 1] = SemanticClass.HAZARD
    result = project_mask_to_costmap(mask, INTRINSICS, GEOMETRY, GRID)

    assert result[center_cell.row, center_cell.col] == class_to_cost(SemanticClass.HAZARD)


T, U, H = SemanticClass.TRAVERSABLE, SemanticClass.UNKNOWN, SemanticClass.HAZARD


@pytest.mark.parametrize(
    "first, second, expected",
    [
        (T, T, T),
        (T, U, U),
        (T, H, H),
        (U, U, U),
        (U, H, H),
        (H, H, H),
    ],
)
@pytest.mark.parametrize("swap_order", [False, True])
def test_same_cell_collision_uses_class_precedence(first, second, expected, swap_order):
    # Same pixel pair as above: (0,0) and (1,1) land in the same cell. Both
    # orders are tested so the result cannot depend on pixel scan order.
    cell = _expected_cell(0.0, 0.0)
    assert _expected_cell(1.0, 1.0) == cell
    if swap_order:
        first, second = second, first

    mask = np.full((2, 2), SemanticClass.TRAVERSABLE, dtype=np.uint8)
    mask[0, 0] = first
    mask[1, 1] = second
    # (0,1) and (1,0) are traversable, the lowest precedence, so they never
    # change the outcome for this cell.
    result = project_mask_to_costmap(mask, INTRINSICS, GEOMETRY, GRID)

    assert result[cell.row, cell.col] == class_to_cost(expected)


def test_hazard_beats_unknown_even_though_unknown_cost_is_numerically_higher():
    assert DEFAULT_COST_VALUES.unknown_cost > DEFAULT_COST_VALUES.hazard_cost
    cell = _expected_cell(0.0, 0.0)

    mask = np.full((2, 2), SemanticClass.UNKNOWN, dtype=np.uint8)
    mask[1, 1] = SemanticClass.HAZARD
    result = project_mask_to_costmap(mask, INTRINSICS, GEOMETRY, GRID)

    assert result[cell.row, cell.col] == DEFAULT_COST_VALUES.hazard_cost


def test_rays_that_do_not_intersect_ground_are_skipped():
    # A pixel exactly at the principal point under zero pitch is parallel
    # to the ground plane and must raise ProjectionError on its own...
    level_geometry = CameraGroundGeometry(camera_height=2.0, pitch_rad=0.0)
    center_intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=0.0, cy=0.0)
    with pytest.raises(ProjectionError):
        project_pixel_to_ground(0.0, 0.0, center_intrinsics, level_geometry)

    # ...but the full mask pipeline must swallow that per-pixel failure
    # instead of crashing, and must not fabricate a cost for it.
    mask = np.full((1, 1), SemanticClass.HAZARD, dtype=np.uint8)
    result = project_mask_to_costmap(mask, center_intrinsics, level_geometry, GRID)

    assert result.shape == (GRID.height, GRID.width)
    assert np.all(result == DEFAULT_COST_VALUES.unknown_cost)


def test_projected_points_outside_grid_are_skipped_not_clamped():
    # The center pixel projects to about (3.46, 0.0) under GEOMETRY, which
    # is nowhere near this grid -- confirm the pixel is dropped rather than
    # clamped into whatever edge cell is closest.
    far_grid = CostmapGridGeometry(resolution=1.0, origin_x=100.0, origin_y=100.0, width=5, height=5)
    mask = np.full((1, 1), SemanticClass.HAZARD, dtype=np.uint8)
    result = project_mask_to_costmap(mask, INTRINSICS, GEOMETRY, far_grid)

    assert result.shape == (5, 5)
    assert np.all(result == DEFAULT_COST_VALUES.unknown_cost)


def test_invalid_mask_ndim_raises_clear_error():
    mask_1d = np.array([0, 1, 2], dtype=np.uint8)
    with pytest.raises(InvalidSemanticClassError):
        project_mask_to_costmap(mask_1d, INTRINSICS, GEOMETRY, GRID)

    mask_3d = np.zeros((2, 2, 2), dtype=np.uint8)
    with pytest.raises(InvalidSemanticClassError):
        project_mask_to_costmap(mask_3d, INTRINSICS, GEOMETRY, GRID)


def test_invalid_class_id_in_mask_raises():
    mask = np.array([[0, 1, 3]], dtype=np.uint8)
    with pytest.raises(InvalidSemanticClassError):
        project_mask_to_costmap(mask, INTRINSICS, GEOMETRY, GRID)
