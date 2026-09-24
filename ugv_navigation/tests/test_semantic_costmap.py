import numpy as np
import pytest

from costmap_core.class_to_cost import (
    CostValues,
    InvalidSemanticClassError,
    SemanticClass,
)
from costmap_core.semantic_costmap import build_semantic_costmap


def test_single_cell_unknown():
    mask = np.array([[SemanticClass.UNKNOWN]])
    costmap = build_semantic_costmap(mask)
    assert costmap.tolist() == [[255]]


def test_single_cell_traversable():
    mask = np.array([[SemanticClass.TRAVERSABLE]])
    costmap = build_semantic_costmap(mask)
    assert costmap.tolist() == [[0]]


def test_single_cell_hazard():
    mask = np.array([[SemanticClass.HAZARD]])
    costmap = build_semantic_costmap(mask)
    assert costmap.tolist() == [[254]]


def test_mixed_2d_mask():
    mask = np.array(
        [
            [0, 1, 2],
            [2, 0, 1],
        ]
    )
    costmap = build_semantic_costmap(mask)
    expected = [
        [255, 0, 254],
        [254, 255, 0],
    ]
    assert costmap.tolist() == expected


def test_output_shape_matches_input_shape():
    mask = np.zeros((7, 5), dtype=np.uint8)
    costmap = build_semantic_costmap(mask)
    assert costmap.shape == mask.shape


def test_invalid_class_id_in_mask_raises():
    mask = np.array([[0, 1], [2, 3]])
    with pytest.raises(InvalidSemanticClassError):
        build_semantic_costmap(mask)


def test_custom_cost_values_are_applied():
    custom = CostValues(unknown_cost=10, traversable_cost=20, hazard_cost=30)
    mask = np.array([[0, 1, 2]])
    costmap = build_semantic_costmap(mask, custom)
    assert costmap.tolist() == [[10, 20, 30]]


def test_empty_mask_returns_empty_costmap_of_same_shape():
    mask = np.zeros((0, 0), dtype=np.uint8)
    costmap = build_semantic_costmap(mask)
    assert costmap.shape == (0, 0)


def test_non_square_empty_mask_preserves_shape():
    mask = np.zeros((0, 5), dtype=np.uint8)
    costmap = build_semantic_costmap(mask)
    assert costmap.shape == (0, 5)


def test_non_2d_input_raises_explicit_error():
    mask_1d = np.array([0, 1, 2])
    with pytest.raises(InvalidSemanticClassError):
        build_semantic_costmap(mask_1d)
