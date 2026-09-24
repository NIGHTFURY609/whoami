import pytest

from costmap_core.grid import CostmapGridGeometry, GridCell, GridError, world_to_grid_cell


def test_basic_position_to_cell():
    geometry = CostmapGridGeometry(resolution=1.0, origin_x=0.0, origin_y=0.0, width=10, height=10)
    cell = world_to_grid_cell(2.5, 3.5, geometry)
    assert cell == GridCell(row=3, col=2)


def test_different_resolution():
    geometry = CostmapGridGeometry(resolution=0.5, origin_x=0.0, origin_y=0.0, width=20, height=20)
    cell = world_to_grid_cell(2.5, 3.5, geometry)
    assert cell == GridCell(row=7, col=5)


def test_origin_offset():
    geometry = CostmapGridGeometry(resolution=1.0, origin_x=10.0, origin_y=-5.0, width=10, height=10)
    cell = world_to_grid_cell(12.0, -2.0, geometry)
    assert cell == GridCell(row=3, col=2)


def test_point_exactly_on_cell_lower_boundary_belongs_to_that_cell():
    geometry = CostmapGridGeometry(resolution=1.0, origin_x=0.0, origin_y=0.0, width=10, height=10)
    cell = world_to_grid_cell(2.0, 3.0, geometry)
    assert cell == GridCell(row=3, col=2)


def test_point_just_below_cell_boundary_belongs_to_previous_cell():
    geometry = CostmapGridGeometry(resolution=1.0, origin_x=0.0, origin_y=0.0, width=10, height=10)
    cell = world_to_grid_cell(1.999999, 2.999999, geometry)
    assert cell == GridCell(row=2, col=1)


def test_point_at_grid_origin_lower_corner():
    geometry = CostmapGridGeometry(resolution=1.0, origin_x=0.0, origin_y=0.0, width=10, height=10)
    cell = world_to_grid_cell(0.0, 0.0, geometry)
    assert cell == GridCell(row=0, col=0)


def test_point_in_last_valid_cell_near_outer_edge():
    geometry = CostmapGridGeometry(resolution=1.0, origin_x=0.0, origin_y=0.0, width=10, height=10)
    cell = world_to_grid_cell(9.999999, 9.999999, geometry)
    assert cell == GridCell(row=9, col=9)


def test_point_exactly_at_outer_grid_boundary_is_outside():
    geometry = CostmapGridGeometry(resolution=1.0, origin_x=0.0, origin_y=0.0, width=10, height=10)
    with pytest.raises(GridError):
        world_to_grid_cell(10.0, 5.0, geometry)


def test_point_outside_grid_negative_coordinates_raises():
    geometry = CostmapGridGeometry(resolution=1.0, origin_x=0.0, origin_y=0.0, width=10, height=10)
    with pytest.raises(GridError):
        world_to_grid_cell(-0.5, 5.0, geometry)


def test_point_far_outside_grid_raises():
    geometry = CostmapGridGeometry(resolution=1.0, origin_x=0.0, origin_y=0.0, width=10, height=10)
    with pytest.raises(GridError):
        world_to_grid_cell(500.0, 500.0, geometry)


def test_non_finite_world_position_raises():
    geometry = CostmapGridGeometry(resolution=1.0, origin_x=0.0, origin_y=0.0, width=10, height=10)
    with pytest.raises(GridError):
        world_to_grid_cell(float("nan"), 5.0, geometry)
    with pytest.raises(GridError):
        world_to_grid_cell(5.0, float("inf"), geometry)


@pytest.mark.parametrize("invalid_resolution", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_resolution_raises(invalid_resolution):
    with pytest.raises(GridError):
        CostmapGridGeometry(
            resolution=invalid_resolution, origin_x=0.0, origin_y=0.0, width=10, height=10
        )


@pytest.mark.parametrize("invalid_width", [0, -5, 1.5, True])
def test_invalid_width_raises(invalid_width):
    with pytest.raises(GridError):
        CostmapGridGeometry(
            resolution=1.0, origin_x=0.0, origin_y=0.0, width=invalid_width, height=10
        )


@pytest.mark.parametrize("invalid_height", [0, -5, 2.5, False])
def test_invalid_height_raises(invalid_height):
    with pytest.raises(GridError):
        CostmapGridGeometry(
            resolution=1.0, origin_x=0.0, origin_y=0.0, width=10, height=invalid_height
        )


def test_invalid_origin_raises():
    with pytest.raises(GridError):
        CostmapGridGeometry(
            resolution=1.0, origin_x=float("nan"), origin_y=0.0, width=10, height=10
        )
