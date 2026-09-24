"""Canonical semantic class -> costmap cost value mapping.

This is the single place where semantic class IDs (defined by Dev 1's
segmentation contract) are translated into costmap cost values. Keeping the
mapping here -- instead of scattering cost-value literals through other
modules -- means the mapping can be changed later without touching any
consumer of `class_to_cost`.
"""

from dataclasses import dataclass
from enum import IntEnum


class SemanticClass(IntEnum):
    """Canonical semantic classes published on /segmentation/mask."""

    UNKNOWN = 0
    TRAVERSABLE = 1
    HAZARD = 2


class InvalidSemanticClassError(ValueError):
    """Raised when a class id outside the canonical set is encountered.

    Deliberately not silently coerced to TRAVERSABLE or any other class --
    an invalid class id must fail loudly rather than be treated as safe.
    """


@dataclass(frozen=True)
class CostValues:
    """Cost values assigned to each canonical semantic class.

    Defaults follow Nav2's documented costmap_2d cost conventions
    (FREE_SPACE=0, LETHAL_OBSTACLE=254, NO_INFORMATION=255), since that is
    the interface this subsystem must eventually feed. These are still only
    defaults: construct a different CostValues to override them if the
    team's final conventions differ.
    """

    unknown_cost: int = 255
    traversable_cost: int = 0
    hazard_cost: int = 254

    def as_mapping(self) -> dict:
        return {
            SemanticClass.UNKNOWN: self.unknown_cost,
            SemanticClass.TRAVERSABLE: self.traversable_cost,
            SemanticClass.HAZARD: self.hazard_cost,
        }


DEFAULT_COST_VALUES = CostValues()


def class_to_cost(class_id: int, cost_values: CostValues = DEFAULT_COST_VALUES) -> int:
    """Map a canonical semantic class id to a costmap cost value.

    Raises InvalidSemanticClassError for any class_id outside {0, 1, 2}.
    """
    try:
        semantic_class = SemanticClass(class_id)
    except ValueError as exc:
        raise InvalidSemanticClassError(
            f"Invalid semantic class id: {class_id!r}. "
            f"Expected one of {[c.value for c in SemanticClass]}."
        ) from exc

    return cost_values.as_mapping()[semantic_class]
