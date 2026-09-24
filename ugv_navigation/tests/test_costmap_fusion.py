import numpy as np
import pytest

from costmap_core.class_to_cost import DEFAULT_COST_VALUES
from costmap_core.costmap_fusion import CostmapFusionError, fuse_costmaps
from costmap_core.geometry_costmap import DEFAULT_GEOMETRY_COST_VALUES

# Explicit synthetic test data -- reuses the same default cost constants
# already defined in class_to_cost.py / geometry_costmap.py, not invented
# values.
UNKNOWN = DEFAULT_COST_VALUES.unknown_cost  # 255
TRAVERSABLE = DEFAULT_COST_VALUES.traversable_cost  # 0
HAZARD = DEFAULT_COST_VALUES.hazard_cost  # 254
LETHAL = DEFAULT_GEOMETRY_COST_VALUES.lethal_cost  # 254
FREE = DEFAULT_GEOMETRY_COST_VALUES.free_cost  # 0


def test_traversable_with_no_geometric_obstacle_stays_traversable():
    semantic = np.array([[TRAVERSABLE]])
    geometry = np.array([[FREE]])
    result = fuse_costmaps(semantic, geometry)
    assert result.tolist() == [[TRAVERSABLE]]


def test_hazard_with_no_geometric_obstacle_stays_hazard():
    semantic = np.array([[HAZARD]])
    geometry = np.array([[FREE]])
    result = fuse_costmaps(semantic, geometry)
    assert result.tolist() == [[HAZARD]]


def test_unknown_with_no_geometric_obstacle_stays_unknown():
    semantic = np.array([[UNKNOWN]])
    geometry = np.array([[FREE]])
    result = fuse_costmaps(semantic, geometry)
    assert result.tolist() == [[UNKNOWN]]


def test_traversable_with_geometric_lethal_becomes_lethal():
    # The critical rule: semantic traversable must never clear a
    # geometric lethal obstacle.
    semantic = np.array([[TRAVERSABLE]])
    geometry = np.array([[LETHAL]])
    result = fuse_costmaps(semantic, geometry)
    assert result.tolist() == [[LETHAL]]


def test_unknown_with_geometric_lethal_becomes_lethal():
    semantic = np.array([[UNKNOWN]])
    geometry = np.array([[LETHAL]])
    result = fuse_costmaps(semantic, geometry)
    assert result.tolist() == [[LETHAL]]


def test_hazard_with_geometric_lethal_stays_lethal():
    semantic = np.array([[HAZARD]])
    geometry = np.array([[LETHAL]])
    result = fuse_costmaps(semantic, geometry)
    assert result.tolist() == [[LETHAL]]


def test_mixed_matrix_covers_all_semantic_geometry_combinations():
    # rows: semantic {traversable, unknown, hazard}; cols: geometry {free, lethal}
    semantic = np.array(
        [
            [TRAVERSABLE, TRAVERSABLE],
            [UNKNOWN, UNKNOWN],
            [HAZARD, HAZARD],
        ]
    )
    geometry = np.array(
        [
            [FREE, LETHAL],
            [FREE, LETHAL],
            [FREE, LETHAL],
        ]
    )
    result = fuse_costmaps(semantic, geometry)
    expected = [
        [TRAVERSABLE, LETHAL],
        [UNKNOWN, LETHAL],
        [HAZARD, LETHAL],
    ]
    assert result.tolist() == expected


def test_shape_mismatch_raises_clear_error():
    semantic = np.zeros((2, 3))
    geometry = np.zeros((3, 2))
    with pytest.raises(CostmapFusionError):
        fuse_costmaps(semantic, geometry)


def test_non_2d_input_raises_clear_error():
    semantic_1d = np.array([TRAVERSABLE, HAZARD])
    geometry_1d = np.array([FREE, LETHAL])
    with pytest.raises(CostmapFusionError):
        fuse_costmaps(semantic_1d, geometry_1d)

    semantic_3d = np.zeros((2, 2, 2))
    geometry_3d = np.zeros((2, 2, 2))
    with pytest.raises(CostmapFusionError):
        fuse_costmaps(semantic_3d, geometry_3d)


def test_inputs_are_not_mutated():
    semantic = np.array([[TRAVERSABLE, HAZARD]])
    geometry = np.array([[LETHAL, FREE]])
    semantic_copy = semantic.copy()
    geometry_copy = geometry.copy()

    fuse_costmaps(semantic, geometry)

    assert np.array_equal(semantic, semantic_copy)
    assert np.array_equal(geometry, geometry_copy)


def test_result_is_a_new_array_not_a_view_into_the_inputs():
    semantic = np.array([[TRAVERSABLE]])
    geometry = np.array([[FREE]])
    result = fuse_costmaps(semantic, geometry)
    result[0, 0] = 999
    assert semantic[0, 0] == TRAVERSABLE
    assert geometry[0, 0] == FREE


def test_custom_lethal_cost_is_respected():
    semantic = np.array([[TRAVERSABLE, TRAVERSABLE]])
    geometry = np.array([[200, 100]])
    result = fuse_costmaps(semantic, geometry, lethal_cost=200)
    assert result.tolist() == [[200, TRAVERSABLE]]
