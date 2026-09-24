"""ROS-independent geometry costmap construction.

Converts a 2D boolean occupancy mask into a 2D costmap of the same shape.

`occupied_mask` is the sole interface: True means geometry indicates
physical occupancy, False means no geometric occupancy. This module does
not know anything about how that mask was produced -- it does not assume a
Depth Anything output format, ROS messages, TF, CameraInfo, depth image
projection, or a VoxelLayer. A future adapter is expected to convert the
real Dev 1 geometry/depth output into this simple boolean mask.
"""

import numpy as np
from dataclasses import dataclass


class GeometryCostmapError(ValueError):
    """Raised when occupied_mask is invalid.

    Deliberately not silently coerced into a free costmap -- invalid input
    must fail loudly rather than be treated as "no obstacles".
    """


@dataclass(frozen=True)
class GeometryCostValues:
    """Cost values assigned to occupied vs. free geometry cells.

    Defaults follow Nav2's documented costmap_2d cost conventions
    (LETHAL_OBSTACLE=254, FREE_SPACE=0). These are still only defaults:
    construct a different GeometryCostValues to override them.
    """

    lethal_cost: int = 254
    free_cost: int = 0


DEFAULT_GEOMETRY_COST_VALUES = GeometryCostValues()


def build_geometry_costmap(
    occupied_mask: np.ndarray,
    cost_values: GeometryCostValues = DEFAULT_GEOMETRY_COST_VALUES,
) -> np.ndarray:
    """Convert a 2D boolean occupancy mask into a 2D geometry costmap.

    Args:
        occupied_mask: 2D boolean array-like. True = occupied (lethal),
            False = free. May be empty (e.g. shape (0, 0)).
        cost_values: cost values to use for the conversion; defaults to
            DEFAULT_GEOMETRY_COST_VALUES.

    Returns:
        A 2D numpy array of costs, with the exact same shape as
        `occupied_mask`.

    Raises:
        GeometryCostmapError: if `occupied_mask` is not a 2D array, or is
            not a boolean array. A wrong dtype (e.g. integer 0/1) is
            rejected explicitly rather than silently reinterpreted, since
            that could silently produce a free costmap from bad input.
    """
    mask = np.asarray(occupied_mask)

    if mask.ndim != 2:
        raise GeometryCostmapError(
            f"Expected a 2D occupied_mask, got array with shape {mask.shape} "
            f"(ndim={mask.ndim})."
        )

    if mask.dtype != np.bool_:
        raise GeometryCostmapError(
            f"Expected occupied_mask to have boolean dtype, got dtype={mask.dtype!r}. "
            "Convert to an explicit boolean array before calling this function."
        )

    return np.where(mask, cost_values.lethal_cost, cost_values.free_cost).astype(np.int64)
