"""End-to-end, ROS-independent validation of the Dev 3 costmap pipeline.

Chains the existing, independently-tested costmap_core stages:

    semantic mask
        -> mask_projection.project_mask_to_costmap   (semantic costmap on the grid)
    occupied mask
        -> geometry_costmap.build_geometry_costmap   (geometry costmap on the grid)
        -> costmap_fusion.fuse_costmaps              (geometry lethal wins)
        -> inflation.inflate_costmap                 (safety buffer)

plus footprint validation/padding, which has no existing interface into the
costmap stages and is therefore exercised on its own.

EVERY camera, grid, geometry, inflation and footprint value below is
SYNTHETIC test data chosen only to make expected cells easy to derive by
hand. None of them are real robot parameters: no real Dev 1 camera
calibration, Dev 2 TF, or Dev 5 footprint is assumed.
"""

import math

import numpy as np
import pytest

from costmap_core.class_to_cost import DEFAULT_COST_VALUES, SemanticClass
from costmap_core.costmap_fusion import fuse_costmaps
from costmap_core.footprint import FootprintError, pad_footprint, validate_footprint
from costmap_core.geometry_costmap import DEFAULT_GEOMETRY_COST_VALUES, build_geometry_costmap
from costmap_core.grid import CostmapGridGeometry
from costmap_core.inflation import inflate_costmap
from costmap_core.mask_projection import project_mask_to_costmap
from costmap_core.projection import CameraGroundGeometry, CameraIntrinsics

UNKNOWN_COST = DEFAULT_COST_VALUES.unknown_cost  # 255
FREE_COST = DEFAULT_COST_VALUES.traversable_cost  # 0
LETHAL_COST = DEFAULT_GEOMETRY_COST_VALUES.lethal_cost  # 254
assert DEFAULT_COST_VALUES.hazard_cost == LETHAL_COST

UNKNOWN = SemanticClass.UNKNOWN.value
TRAVERSABLE = SemanticClass.TRAVERSABLE.value
HAZARD = SemanticClass.HAZARD.value

# --- Synthetic setup -------------------------------------------------------
# A 20x20 "image" seen by a camera 1 m above the ground, pitched straight
# down (pi/2), with fx = fy = 10 px and the principal point at the image
# centre. Under that geometry pixel (u, v) lands on the ground at
#     x = (cy - v) / fy * h = (9.5 - v) * 0.1
#     y = (cx - u) / fx * h = (9.5 - u) * 0.1
# i.e. at the centre of a 0.1 m cell. With a 20x20 grid of 0.1 m cells whose
# origin is (-1, -1), that is cell (row = 19 - u, col = 19 - v): every
# pixel maps to exactly one cell and every cell receives exactly one pixel.
SIZE = 20
RESOLUTION = 0.1
INTRINSICS = CameraIntrinsics(fx=10.0, fy=10.0, cx=9.5, cy=9.5)
DOWNWARD_CAMERA = CameraGroundGeometry(camera_height=1.0, pitch_rad=math.pi / 2)
GRID = CostmapGridGeometry(
    resolution=RESOLUTION, origin_x=-1.0, origin_y=-1.0, width=SIZE, height=SIZE
)
INFLATION_RADIUS = 0.3  # metres, synthetic


def pixel_to_cell(u, v):
    """Hand-derived (row, col) for pixel (u, v) under the synthetic setup."""
    return SIZE - 1 - u, SIZE - 1 - v


def cell_to_pixel(row, col):
    """Inverse of pixel_to_cell, returned as mask indices (v, u)."""
    return SIZE - 1 - col, SIZE - 1 - row


def run_pipeline(mask, occupied, inflation_radius=INFLATION_RADIUS):
    """Run every costmap stage and return each intermediate result."""
    semantic = project_mask_to_costmap(mask, INTRINSICS, DOWNWARD_CAMERA, GRID)
    geometry = build_geometry_costmap(occupied)
    fused = fuse_costmaps(semantic, geometry)
    final = inflate_costmap(fused, RESOLUTION, inflation_radius)
    return {"semantic": semantic, "geometry": geometry, "fused": fused, "final": final}


# Scenario: traversable ground everywhere except a few labelled pixels and
# geometric obstacles, each placed far enough apart (> INFLATION_RADIUS)
# that their inflation zones do not overlap.
HAZARD_CELL = (3, 3)
UNKNOWN_CELL = (3, 16)
UNKNOWN_NEXT_TO_LETHAL_CELL = (16, 4)  # adjacent to GEOMETRY_ON_UNKNOWN_CELL
GEOMETRY_ON_TRAVERSABLE_CELL = (10, 10)
GEOMETRY_ON_HAZARD_CELL = (16, 16)
GEOMETRY_ON_UNKNOWN_CELL = (16, 3)
TRAVERSABLE_PROBE_CELL = (10, 3)  # far from every lethal cell


def build_scenario():
    mask = np.full((SIZE, SIZE), TRAVERSABLE, dtype=np.uint8)
    for cell, cls in [
        (HAZARD_CELL, HAZARD),
        (UNKNOWN_CELL, UNKNOWN),
        (UNKNOWN_NEXT_TO_LETHAL_CELL, UNKNOWN),
        (GEOMETRY_ON_HAZARD_CELL, HAZARD),
        (GEOMETRY_ON_UNKNOWN_CELL, UNKNOWN),
    ]:
        mask[cell_to_pixel(*cell)] = cls

    occupied = np.zeros((SIZE, SIZE), dtype=bool)
    for cell in [GEOMETRY_ON_TRAVERSABLE_CELL, GEOMETRY_ON_HAZARD_CELL, GEOMETRY_ON_UNKNOWN_CELL]:
        occupied[cell] = True
    return mask, occupied


def cells_within(center, radius_m):
    """All in-grid cells at Euclidean distance < radius_m from `center`."""
    cr, cc = center
    return [
        (r, c)
        for r in range(SIZE)
        for c in range(SIZE)
        if math.hypot((r - cr) * RESOLUTION, (c - cc) * RESOLUTION) < radius_m
    ]


@pytest.fixture
def scenario():
    mask, occupied = build_scenario()
    return mask, occupied, run_pipeline(mask, occupied)


# --- Synthetic setup sanity ------------------------------------------------


def test_synthetic_camera_maps_every_pixel_to_its_hand_derived_cell():
    for u in range(SIZE):
        for v in range(SIZE):
            mask = np.full((SIZE, SIZE), TRAVERSABLE, dtype=np.uint8)
            mask[v, u] = HAZARD
            semantic = project_mask_to_costmap(mask, INTRINSICS, DOWNWARD_CAMERA, GRID)
            rows, cols = np.nonzero(semantic == LETHAL_COST)
            assert list(zip(rows.tolist(), cols.tolist())) == [pixel_to_cell(u, v)]


def test_all_stages_agree_on_grid_shape(scenario):
    _, _, stages = scenario
    for name, array in stages.items():
        assert array.shape == (GRID.height, GRID.width), name


# --- Semantic flow ---------------------------------------------------------


def test_traversable_pixel_reaches_its_cell_as_free(scenario):
    _, _, stages = scenario
    for stage in ("semantic", "fused", "final"):
        assert stages[stage][TRAVERSABLE_PROBE_CELL] == FREE_COST, stage


def test_hazard_pixel_produces_lethal_cost(scenario):
    _, _, stages = scenario
    assert stages["semantic"][HAZARD_CELL] == LETHAL_COST
    assert stages["fused"][HAZARD_CELL] == LETHAL_COST
    assert stages["final"][HAZARD_CELL] == LETHAL_COST


def test_semantic_hazard_is_also_inflated(scenario):
    _, _, stages = scenario
    r, c = HAZARD_CELL
    assert 0 < stages["final"][r + 1, c] < LETHAL_COST


def test_unknown_pixel_stays_unknown_and_non_free(scenario):
    _, _, stages = scenario
    for stage in ("semantic", "fused", "final"):
        assert stages[stage][UNKNOWN_CELL] == UNKNOWN_COST, stage
        assert stages[stage][UNKNOWN_CELL] != FREE_COST, stage


def test_unknown_inside_inflation_zone_is_not_lowered(scenario):
    _, _, stages = scenario
    assert stages["final"][UNKNOWN_NEXT_TO_LETHAL_CELL] == UNKNOWN_COST


def test_cells_no_pixel_reaches_stay_unknown():
    # Under a level synthetic camera the upper half of the image is above the
    # horizon and the lower half hits the ground beyond the grid's far edge,
    # so no pixel reaches any cell: every cell must stay unknown, not free.
    level_camera = CameraGroundGeometry(camera_height=1.0, pitch_rad=0.0)
    mask = np.full((SIZE, SIZE), TRAVERSABLE, dtype=np.uint8)
    semantic = project_mask_to_costmap(mask, INTRINSICS, level_camera, GRID)
    final = inflate_costmap(
        fuse_costmaps(semantic, build_geometry_costmap(np.zeros((SIZE, SIZE), dtype=bool))),
        RESOLUTION,
        INFLATION_RADIUS,
    )
    assert np.all(final == UNKNOWN_COST)


# --- Geometry precedence ---------------------------------------------------


@pytest.mark.parametrize(
    "cell",
    [GEOMETRY_ON_TRAVERSABLE_CELL, GEOMETRY_ON_HAZARD_CELL, GEOMETRY_ON_UNKNOWN_CELL],
)
def test_geometric_lethal_wins_over_any_semantic_class(scenario, cell):
    _, _, stages = scenario
    assert stages["fused"][cell] == LETHAL_COST
    assert stages["final"][cell] == LETHAL_COST


def test_semantic_traversable_cannot_clear_geometric_lethal():
    # Semantics claim the whole grid is traversable; geometry disagrees.
    mask = np.full((SIZE, SIZE), TRAVERSABLE, dtype=np.uint8)
    occupied = np.zeros((SIZE, SIZE), dtype=bool)
    occupied[5, 7] = True
    occupied[12, 2:9] = True  # a synthetic wall

    stages = run_pipeline(mask, occupied)

    assert np.all(stages["semantic"] == FREE_COST)
    assert np.all(stages["fused"][occupied] == LETHAL_COST)
    assert np.all(stages["final"][occupied] == LETHAL_COST)
    assert np.array_equal(stages["final"] == LETHAL_COST, occupied)


# --- Inflation -------------------------------------------------------------


def test_inflation_never_reduces_any_cost(scenario):
    _, _, stages = scenario
    assert np.all(stages["final"] >= stages["fused"])


def test_inflation_never_creates_new_lethal_cells(scenario):
    _, _, stages = scenario
    assert np.array_equal(stages["final"] == LETHAL_COST, stages["fused"] == LETHAL_COST)


def test_inflation_raises_free_cells_within_radius(scenario):
    _, _, stages = scenario
    fused, final = stages["fused"], stages["final"]
    for center in [GEOMETRY_ON_TRAVERSABLE_CELL, HAZARD_CELL]:
        for cell in cells_within(center, INFLATION_RADIUS):
            if cell == center:
                continue
            assert fused[cell] == FREE_COST, cell
            assert 0 < final[cell] < LETHAL_COST, cell


def test_inflation_cost_decreases_with_distance(scenario):
    _, _, stages = scenario
    r, c = GEOMETRY_ON_TRAVERSABLE_CELL
    final = stages["final"]
    assert final[r, c] > final[r, c + 1] > final[r, c + 2] > final[r, c + 3] == FREE_COST


def test_cells_outside_inflation_radius_are_unchanged(scenario):
    _, _, stages = scenario
    fused, final = stages["fused"], stages["final"]
    lethal = list(zip(*np.nonzero(fused == LETHAL_COST)))
    for r in range(SIZE):
        for c in range(SIZE):
            nearest = min(math.hypot((r - lr) * RESOLUTION, (c - lc) * RESOLUTION) for lr, lc in lethal)
            if nearest >= INFLATION_RADIUS:
                assert final[r, c] == fused[r, c], (r, c)


def test_zero_inflation_radius_leaves_fused_costmap_unchanged():
    mask, occupied = build_scenario()
    stages = run_pipeline(mask, occupied, inflation_radius=0.0)
    assert np.array_equal(stages["final"], stages["fused"])


# --- Footprint (no existing interface into the costmap stages) -------------

# Synthetic origin-centred rectangle, NOT the real Dev 5 footprint.
SYNTHETIC_FOOTPRINT = [(0.2, 0.15), (-0.2, 0.15), (-0.2, -0.15), (0.2, -0.15)]


def test_synthetic_footprint_validates_unchanged():
    assert validate_footprint(SYNTHETIC_FOOTPRINT) == SYNTHETIC_FOOTPRINT


def test_padding_grows_rectangle_half_extents_by_padding():
    padded = pad_footprint(SYNTHETIC_FOOTPRINT, 0.05)
    assert padded == pytest.approx([(0.25, 0.2), (-0.25, 0.2), (-0.25, -0.2), (0.25, -0.2)])


def test_zero_padding_returns_equal_copy():
    padded = pad_footprint(SYNTHETIC_FOOTPRINT, 0.0)
    assert padded == SYNTHETIC_FOOTPRINT
    assert padded is not SYNTHETIC_FOOTPRINT


@pytest.mark.parametrize(
    "footprint, padding",
    [
        ([(0.0, 0.0), (1.0, 0.0)], 0.1),  # too few vertices
        ([(0, 0), (1, 0), (1, 1), (0.5, 0.2), (0, 1)], 0.1),  # concave
        (SYNTHETIC_FOOTPRINT + [SYNTHETIC_FOOTPRINT[0]], 0.1),  # explicitly closed
        (SYNTHETIC_FOOTPRINT, -0.1),  # negative padding
        (SYNTHETIC_FOOTPRINT, float("nan")),
    ],
)
def test_invalid_footprint_or_padding_is_rejected(footprint, padding):
    with pytest.raises(FootprintError):
        pad_footprint(footprint, padding)


# --- Robustness ------------------------------------------------------------


def test_out_of_grid_and_non_ground_pixels_do_not_crash_pipeline():
    # Tilted synthetic camera over a grid smaller than its field of view:
    # upper pixels are above the horizon (no ground intersection) and many
    # ground pixels land outside the grid. Both must be skipped silently.
    tilted = CameraGroundGeometry(camera_height=1.0, pitch_rad=0.3)
    small_grid = CostmapGridGeometry(resolution=0.1, origin_x=0.0, origin_y=-0.5, width=10, height=10)
    mask = np.full((SIZE, SIZE), HAZARD, dtype=np.uint8)

    semantic = project_mask_to_costmap(mask, INTRINSICS, tilted, small_grid)
    geometry = build_geometry_costmap(np.zeros(semantic.shape, dtype=bool))
    final = inflate_costmap(fuse_costmaps(semantic, geometry), small_grid.resolution, INFLATION_RADIUS)

    assert final.shape == (small_grid.height, small_grid.width)
    assert np.all((final >= FREE_COST) & (final <= UNKNOWN_COST))
    assert np.count_nonzero(semantic == LETHAL_COST) > 0


def test_mask_entirely_outside_grid_yields_all_unknown():
    far_grid = CostmapGridGeometry(resolution=0.1, origin_x=50.0, origin_y=50.0, width=5, height=5)
    mask = np.full((SIZE, SIZE), TRAVERSABLE, dtype=np.uint8)
    semantic = project_mask_to_costmap(mask, INTRINSICS, DOWNWARD_CAMERA, far_grid)
    final = inflate_costmap(
        fuse_costmaps(semantic, build_geometry_costmap(np.zeros((5, 5), dtype=bool))),
        far_grid.resolution,
        INFLATION_RADIUS,
    )
    assert np.all(final == UNKNOWN_COST)


def test_pipeline_does_not_mutate_its_inputs():
    mask, occupied = build_scenario()
    mask_before, occupied_before = mask.copy(), occupied.copy()
    footprint_before = list(SYNTHETIC_FOOTPRINT)

    stages = run_pipeline(mask, occupied)
    pad_footprint(SYNTHETIC_FOOTPRINT, 0.05)

    assert np.array_equal(mask, mask_before)
    assert np.array_equal(occupied, occupied_before)
    assert SYNTHETIC_FOOTPRINT == footprint_before
    # Each stage must also leave the previous stage's output untouched.
    semantic_before = stages["semantic"].copy()
    fused_before = stages["fused"].copy()
    fuse_costmaps(stages["semantic"], stages["geometry"])
    inflate_costmap(stages["fused"], RESOLUTION, INFLATION_RADIUS)
    assert np.array_equal(stages["semantic"], semantic_before)
    assert np.array_equal(stages["fused"], fused_before)


# --- Same-cell semantic collisions ----------------------------------------

# A coarser synthetic grid (0.2 m) so that pixel pairs share a cell:
# (u=0, v=0) and (u=1, v=0) both land in cell (row=9, col=9).
COARSE_GRID = CostmapGridGeometry(resolution=0.2, origin_x=-1.0, origin_y=-1.0, width=10, height=10)
SHARED_CELL = (9, 9)


def run_coarse_pipeline(mask, occupied):
    semantic = project_mask_to_costmap(mask, INTRINSICS, DOWNWARD_CAMERA, COARSE_GRID)
    fused = fuse_costmaps(semantic, build_geometry_costmap(occupied))
    final = inflate_costmap(fused, COARSE_GRID.resolution, INFLATION_RADIUS)
    return semantic, fused, final


def test_hazard_stays_lethal_when_sharing_a_cell_with_unknown():
    mask = np.full((SIZE, SIZE), TRAVERSABLE, dtype=np.uint8)
    mask[0, 0] = HAZARD
    mask[0, 1] = UNKNOWN

    semantic, fused, final = run_coarse_pipeline(mask, np.zeros((10, 10), dtype=bool))

    assert semantic[SHARED_CELL] == LETHAL_COST
    assert fused[SHARED_CELL] == LETHAL_COST
    assert final[SHARED_CELL] == LETHAL_COST
    # Being lethal, it now gets an inflation buffer.
    r, c = SHARED_CELL
    assert 0 < final[r, c - 1] < LETHAL_COST


def test_geometric_lethal_still_wins_over_a_collided_semantic_cell():
    mask = np.full((SIZE, SIZE), TRAVERSABLE, dtype=np.uint8)
    mask[0, 1] = UNKNOWN  # traversable + unknown -> unknown in SHARED_CELL
    occupied = np.zeros((10, 10), dtype=bool)
    occupied[SHARED_CELL] = True

    semantic, fused, final = run_coarse_pipeline(mask, occupied)

    assert semantic[SHARED_CELL] == UNKNOWN_COST
    assert fused[SHARED_CELL] == LETHAL_COST
    assert final[SHARED_CELL] == LETHAL_COST
