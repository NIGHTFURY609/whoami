import pytest

from costmap_core.class_to_cost import (
    CostValues,
    InvalidSemanticClassError,
    SemanticClass,
    class_to_cost,
)


def test_unknown_maps_to_configured_unknown_cost():
    cost_values = CostValues(unknown_cost=255, traversable_cost=0, hazard_cost=254)
    assert class_to_cost(SemanticClass.UNKNOWN, cost_values) == 255


def test_traversable_maps_to_configured_free_cost():
    cost_values = CostValues(unknown_cost=255, traversable_cost=0, hazard_cost=254)
    assert class_to_cost(SemanticClass.TRAVERSABLE, cost_values) == 0


def test_hazard_maps_to_configured_lethal_cost():
    cost_values = CostValues(unknown_cost=255, traversable_cost=0, hazard_cost=254)
    assert class_to_cost(SemanticClass.HAZARD, cost_values) == 254


def test_default_cost_values_used_when_not_provided():
    assert class_to_cost(SemanticClass.TRAVERSABLE) == 0


def test_custom_cost_values_are_respected():
    custom = CostValues(unknown_cost=128, traversable_cost=10, hazard_cost=200)
    assert class_to_cost(SemanticClass.HAZARD, custom) == 200


@pytest.mark.parametrize("invalid_class_id", [-1, 3, 99, 255])
def test_invalid_class_id_raises_explicit_error(invalid_class_id):
    with pytest.raises(InvalidSemanticClassError):
        class_to_cost(invalid_class_id)
