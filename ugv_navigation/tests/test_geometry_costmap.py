import numpy as np
import pytest

from costmap_core.geometry_costmap import (
    GeometryCostmapError,
    GeometryCostValues,
    build_geometry_costmap,
)


def test_all_false_mask_is_all_free_cost():
    mask = np.zeros((3, 4), dtype=bool)
    costmap = build_geometry_costmap(mask)
    assert costmap.shape == mask.shape
    assert np.all(costmap == 0)


def test_all_true_mask_is_all_lethal_cost():
    mask = np.ones((3, 4), dtype=bool)
    costmap = build_geometry_costmap(mask)
    assert costmap.shape == mask.shape
    assert np.all(costmap == 254)


def test_mixed_occupied_and_free_cells():
    mask = np.array(
        [
            [False, True, False],
            [True, True, False],
        ]
    )
    costmap = build_geometry_costmap(mask)
    expected = [
        [0, 254, 0],
        [254, 254, 0],
    ]
    assert costmap.tolist() == expected


def test_output_shape_matches_input_shape():
    mask = np.zeros((6, 9), dtype=bool)
    costmap = build_geometry_costmap(mask)
    assert costmap.shape == mask.shape


def test_empty_2d_mask_returns_empty_costmap_of_same_shape():
    mask = np.zeros((0, 0), dtype=bool)
    costmap = build_geometry_costmap(mask)
    assert costmap.shape == (0, 0)


def test_non_square_empty_mask_preserves_shape():
    mask = np.zeros((0, 5), dtype=bool)
    costmap = build_geometry_costmap(mask)
    assert costmap.shape == (0, 5)


def test_non_2d_input_raises():
    mask_1d = np.array([True, False, True])
    with pytest.raises(GeometryCostmapError):
        build_geometry_costmap(mask_1d)

    mask_3d = np.zeros((2, 2, 2), dtype=bool)
    with pytest.raises(GeometryCostmapError):
        build_geometry_costmap(mask_3d)


def test_non_boolean_dtype_raises():
    int_mask = np.array([[0, 1], [1, 0]])
    with pytest.raises(GeometryCostmapError):
        build_geometry_costmap(int_mask)

    float_mask = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(GeometryCostmapError):
        build_geometry_costmap(float_mask)


def test_custom_lethal_and_free_cost_are_applied():
    custom = GeometryCostValues(lethal_cost=200, free_cost=10)
    mask = np.array([[False, True]])
    costmap = build_geometry_costmap(mask, custom)
    assert costmap.tolist() == [[10, 200]]


def test_default_lethal_cost_is_254():
    assert GeometryCostValues().lethal_cost == 254
