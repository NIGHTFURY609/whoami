"""ROS-independent semantic costmap construction.

Converts a 2D semantic class mask into a 2D costmap of the same shape,
by reusing the canonical class -> cost conversion defined in
class_to_cost.py.

This module performs only class -> cost conversion. It does not interpret
mask cells as physical/ground coordinates -- pixel/ground projection is a
separate, not-yet-implemented component.
"""

import numpy as np

from costmap_core.class_to_cost import (
    CostValues,
    DEFAULT_COST_VALUES,
    InvalidSemanticClassError,
    class_to_cost,
)


def build_semantic_costmap(
    mask: np.ndarray, cost_values: CostValues = DEFAULT_COST_VALUES
) -> np.ndarray:
    """Convert a 2D semantic class mask into a 2D costmap.

    Args:
        mask: 2D array-like of canonical semantic class ids (0=unknown,
            1=traversable, 2=hazard). May be empty (e.g. shape (0, 0)).
        cost_values: cost values to use for the conversion; defaults to
            class_to_cost.DEFAULT_COST_VALUES.

    Returns:
        A 2D numpy array of costs, with the exact same shape as `mask`.

    Raises:
        InvalidSemanticClassError: if `mask` is not 2D, or contains any
            class id outside {0, 1, 2}. Invalid ids are never silently
            treated as traversable or any other class.
    """
    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise InvalidSemanticClassError(
            f"Expected a 2D semantic class mask, got array with shape "
            f"{mask.shape} (ndim={mask.ndim})."
        )

    height, width = mask.shape
    costmap = np.empty((height, width), dtype=np.int64)

    for row in range(height):
        for col in range(width):
            costmap[row, col] = class_to_cost(int(mask[row, col]), cost_values)

    return costmap
