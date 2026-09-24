import numpy as np
import pytest

from costmap_core.inflation import InflationError, inflate_costmap

LETHAL = 254


def test_no_lethal_obstacles_leaves_costmap_unchanged():
    grid = np.zeros((4, 4), dtype=np.int64)
    grid[1, 1] = 50  # non-lethal cost, not an obstacle
    result = inflate_costmap(grid, resolution=1.0, inflation_radius=2.0, lethal_cost=LETHAL)
    assert np.array_equal(result, grid)


def test_single_lethal_obstacle_remains_lethal():
    grid = np.zeros((5, 5), dtype=np.int64)
    grid[2, 2] = LETHAL
    result = inflate_costmap(grid, resolution=1.0, inflation_radius=2.0, lethal_cost=LETHAL)
    assert result[2, 2] == LETHAL


def test_nearby_cells_receive_inflation_costs():
    grid = np.zeros((7, 7), dtype=np.int64)
    grid[3, 3] = LETHAL
    result = inflate_costmap(grid, resolution=1.0, inflation_radius=3.0, lethal_cost=LETHAL)

    # One cell away (Euclidean distance 1.0 of a 3.0 m radius).
    assert result[3, 4] == 169
    assert result[3, 4] > 0


def test_cells_outside_radius_remain_unchanged():
    grid = np.zeros((7, 7), dtype=np.int64)
    grid[3, 3] = LETHAL
    result = inflate_costmap(grid, resolution=1.0, inflation_radius=3.0, lethal_cost=LETHAL)

    # 3 cells away == exactly the radius boundary -> no contribution.
    assert result[3, 6] == 0
    # corners of this 7x7 grid are far outside a 3-cell radius from (3,3).
    assert result[0, 0] == 0
    assert result[6, 6] == 0


def test_increasing_distance_produces_lower_or_equal_inflation_cost():
    grid = np.zeros((7, 7), dtype=np.int64)
    grid[3, 3] = LETHAL
    result = inflate_costmap(grid, resolution=1.0, inflation_radius=3.0, lethal_cost=LETHAL)

    one_away = int(result[3, 4])
    two_away = int(result[3, 5])
    three_away = int(result[3, 6])
    assert one_away > two_away > three_away == 0


def test_existing_higher_cost_is_never_reduced():
    grid = np.zeros((3, 3), dtype=np.int64)
    grid[1, 1] = LETHAL
    # Neighbour already carries a cost higher than the inflation formula
    # would assign at distance 1.0 (169 at radius 3.0, see above).
    grid[1, 0] = 250
    result = inflate_costmap(grid, resolution=1.0, inflation_radius=3.0, lethal_cost=LETHAL)
    assert result[1, 0] == 250


def test_multiple_obstacles_inflate_independently_and_combine_with_max():
    grid = np.zeros((5, 7), dtype=np.int64)
    grid[2, 1] = LETHAL
    grid[2, 5] = LETHAL
    result = inflate_costmap(grid, resolution=1.0, inflation_radius=2.0, lethal_cost=LETHAL)

    assert result[2, 1] == LETHAL
    assert result[2, 5] == LETHAL
    # Directly between the two obstacles (distance 2 from each) gets no
    # contribution from either at this radius.
    assert result[2, 3] == 0
    # Adjacent to each obstacle, both halos apply independently.
    assert result[2, 0] == result[2, 2] == result[2, 4] == result[2, 6] == 126


def test_zero_inflation_radius_leaves_costmap_unchanged_except_lethal():
    grid = np.zeros((3, 3), dtype=np.int64)
    grid[1, 1] = LETHAL
    result = inflate_costmap(grid, resolution=1.0, inflation_radius=0.0, lethal_cost=LETHAL)
    assert np.array_equal(result, grid)


@pytest.mark.parametrize("bad_resolution", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_resolution_raises(bad_resolution):
    grid = np.zeros((3, 3), dtype=np.int64)
    with pytest.raises(InflationError):
        inflate_costmap(grid, resolution=bad_resolution, inflation_radius=1.0, lethal_cost=LETHAL)


@pytest.mark.parametrize("bad_radius", [-1.0, float("nan"), float("inf")])
def test_invalid_inflation_radius_raises(bad_radius):
    grid = np.zeros((3, 3), dtype=np.int64)
    with pytest.raises(InflationError):
        inflate_costmap(grid, resolution=1.0, inflation_radius=bad_radius, lethal_cost=LETHAL)


def test_non_2d_input_raises():
    with pytest.raises(InflationError):
        inflate_costmap(np.zeros(4), resolution=1.0, inflation_radius=1.0, lethal_cost=LETHAL)
    with pytest.raises(InflationError):
        inflate_costmap(np.zeros((2, 2, 2)), resolution=1.0, inflation_radius=1.0, lethal_cost=LETHAL)


def test_input_array_is_not_mutated():
    grid = np.zeros((5, 5), dtype=np.int64)
    grid[2, 2] = LETHAL
    original = grid.copy()
    inflate_costmap(grid, resolution=1.0, inflation_radius=2.0, lethal_cost=LETHAL)
    assert np.array_equal(grid, original)


def test_corner_obstacle_does_not_crash_and_clips_to_grid_bounds():
    grid = np.zeros((5, 5), dtype=np.int64)
    grid[0, 0] = LETHAL
    result = inflate_costmap(grid, resolution=1.0, inflation_radius=2.0, lethal_cost=LETHAL)

    assert result.shape == (5, 5)
    assert result[0, 0] == LETHAL
    # Diagonal neighbour still gets inflated even though half the notional
    # search box around (0,0) falls outside the grid.
    assert result[1, 1] == 74
    # Far corner, outside the radius, is untouched.
    assert result[4, 4] == 0
