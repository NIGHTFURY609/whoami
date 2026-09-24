"""Fuse a semantic costmap and a geometry costmap into a final costmap.

Design law (architecture.md §9 / PROJECT_CONTEXT.md §3 & §7, Dev 3):
GEOMETRY LETHAL ALWAYS WINS. Semantic "traversable" must never clear a
geometric lethal obstacle, and semantic "unknown" must never be silently
turned into free space.

This module is ROS-independent: it only combines two same-shaped numpy
cost arrays that are assumed to already be in "cost space" (e.g. the
outputs of `semantic_costmap.build_semantic_costmap` and
`geometry_costmap.build_geometry_costmap`). It does not know about
classes, occupancy masks, camera projection, TF, or ROS messages.
"""

from __future__ import annotations

import numpy as np

# Matches class_to_cost.CostValues.hazard_cost and
# geometry_costmap.GeometryCostValues.lethal_cost by default.
DEFAULT_LETHAL_COST = 254


class CostmapFusionError(ValueError):
    """Raised when semantic_costmap/geometry_costmap inputs are invalid.

    Deliberately not silently coerced or broadcast -- mismatched or
    malformed inputs must fail loudly rather than produce a costmap with
    undefined meaning.
    """


def fuse_costmaps(
    semantic_costmap: np.ndarray,
    geometry_costmap: np.ndarray,
    lethal_cost: int = DEFAULT_LETHAL_COST,
) -> np.ndarray:
    """Fuse a semantic costmap with a geometry costmap.

    For every cell:
        if geometry_costmap[cell] == lethal_cost: final_cost = lethal_cost
        else:                                     final_cost = semantic_costmap[cell]

    Geometry lethal always wins: semantic traversable can never clear a
    geometric lethal obstacle. Semantic unknown is passed through
    unchanged (never silently turned into free) whenever geometry is not
    lethal there.

    Args:
        semantic_costmap: 2D array of per-cell semantic costs (e.g. from
            `semantic_costmap.build_semantic_costmap`).
        geometry_costmap: 2D array of per-cell geometry costs, same shape
            as `semantic_costmap` (e.g. from
            `geometry_costmap.build_geometry_costmap`).
        lethal_cost: the geometry cost value treated as lethal. Defaults
            to 254, matching `class_to_cost.CostValues.hazard_cost` and
            `geometry_costmap.GeometryCostValues.lethal_cost`.

    Returns:
        A new 2D numpy array, the same shape as the inputs. Neither input
        is mutated or returned by reference.

    Raises:
        CostmapFusionError: if either input is not 2D, or the two inputs
            do not have identical shapes.
    """
    semantic = np.asarray(semantic_costmap)
    geometry = np.asarray(geometry_costmap)

    if semantic.ndim != 2:
        raise CostmapFusionError(
            f"semantic_costmap must be 2D, got shape {semantic.shape} "
            f"(ndim={semantic.ndim})."
        )
    if geometry.ndim != 2:
        raise CostmapFusionError(
            f"geometry_costmap must be 2D, got shape {geometry.shape} "
            f"(ndim={geometry.ndim})."
        )
    if semantic.shape != geometry.shape:
        raise CostmapFusionError(
            f"semantic_costmap shape {semantic.shape} does not match "
            f"geometry_costmap shape {geometry.shape}."
        )

    return np.where(geometry == lethal_cost, lethal_cost, semantic)
