"""Costmap inflation: a distance-based safety buffer around lethal cells.

Keeps a planner from routing directly against an obstacle by raising the
cost of cells near a lethal cell, decaying with distance. This is the
Dev 3 core implementation only: ROS-independent, no Nav2, no TF, no robot
footprint / footprint padding (those are separate tasks -- see
architecture.md / dev.md's Dev 3 "Inflation" and "Footprint Padding"
items, which are listed as distinct steps).

No inflation-cost formula is defined anywhere in PROJECT_CONTEXT.md or
architecture.md, so this module defines its own, kept deliberately
isolated in `_inflation_cost_at_distance` so it can be swapped for a real
Nav2-compatible formula (e.g. exponential decay driven by
`cost_scaling_factor` / `inscribed_radius`) once those parameters are
actually specified for this project.
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_LETHAL_COST = 254


class InflationError(ValueError):
    """Raised when inflate_costmap inputs are invalid.

    Deliberately not silently coerced -- a malformed costmap or malformed
    inflation geometry must fail loudly rather than silently produce a
    costmap with undefined safety meaning.
    """


def _inflation_cost_at_distance(
    distance_m: float, inflation_radius_m: float, lethal_cost: int
) -> float:
    """Dev 3 placeholder decay: linear from `lethal_cost - 1` at the
    obstacle itself down to 0 at `inflation_radius_m`.

    Kept strictly below `lethal_cost` at every distance so an inflated
    cell is never indistinguishable from an actual lethal obstacle cell.
    Distance >= inflation_radius_m (or a non-positive radius) contributes
    no cost.
    """
    if inflation_radius_m <= 0.0 or distance_m >= inflation_radius_m:
        return 0.0
    max_cost = lethal_cost - 1
    fraction = 1.0 - (distance_m / inflation_radius_m)
    return max_cost * fraction


def _require_finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InflationError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise InflationError(f"{name} must be finite, got {value!r}")
    return value


def inflate_costmap(
    costmap: np.ndarray,
    resolution: float,
    inflation_radius: float,
    lethal_cost: int = DEFAULT_LETHAL_COST,
) -> np.ndarray:
    """Inflate a 2D costmap around every cell equal to `lethal_cost`.

    For every cell within `inflation_radius` metres (Euclidean, using
    `resolution` metres/cell) of any lethal cell, the cost decays with
    distance via `_inflation_cost_at_distance`. A cell's final cost is
    `max(original_cost, inflation_cost)`: an inflated cost never reduces
    an existing higher cost (e.g. a nearby but separate lethal or hazard
    cell), and lethal cells always remain lethal (the inflation formula
    never reaches `lethal_cost` itself, so the original lethal value
    always wins the max).

    Args:
        costmap: 2D array of per-cell costs.
        resolution: grid resolution in metres/cell. Must be finite > 0.
        inflation_radius: safety buffer radius in metres. Must be finite
            and >= 0. A radius of 0 leaves the costmap unchanged.
        lethal_cost: the cost value treated as a lethal obstacle.
            Defaults to 254, matching `class_to_cost.CostValues` and
            `geometry_costmap.GeometryCostValues` / `costmap_fusion`.

    Returns:
        A new 2D numpy array (dtype int64), same shape as `costmap`.
        `costmap` itself is never mutated.

    Raises:
        InflationError: if `costmap` is not 2D or has a zero-length
            dimension, if `resolution` is not finite or <= 0, or if
            `inflation_radius` is not finite or < 0.
    """
    grid = np.asarray(costmap)

    if grid.ndim != 2:
        raise InflationError(
            f"costmap must be 2D, got shape {grid.shape} (ndim={grid.ndim})."
        )
    if grid.shape[0] == 0 or grid.shape[1] == 0:
        raise InflationError(
            f"costmap dimensions must be non-empty, got shape {grid.shape}."
        )

    resolution = _require_finite_number(resolution, name="resolution")
    if resolution <= 0.0:
        raise InflationError(f"resolution must be > 0, got {resolution!r}.")

    inflation_radius = _require_finite_number(inflation_radius, name="inflation_radius")
    if inflation_radius < 0.0:
        raise InflationError(f"inflation_radius must be >= 0, got {inflation_radius!r}.")

    result = grid.astype(np.int64, copy=True)

    lethal_rows, lethal_cols = np.nonzero(grid == lethal_cost)
    if lethal_rows.size == 0 or inflation_radius == 0.0:
        return result

    height, width = grid.shape
    radius_cells = math.ceil(inflation_radius / resolution)

    for lr, lc in zip(lethal_rows.tolist(), lethal_cols.tolist()):
        r_min = max(0, lr - radius_cells)
        r_max = min(height - 1, lr + radius_cells)
        c_min = max(0, lc - radius_cells)
        c_max = min(width - 1, lc + radius_cells)
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                distance_m = math.hypot((r - lr) * resolution, (c - lc) * resolution)
                cost = int(
                    round(_inflation_cost_at_distance(distance_m, inflation_radius, lethal_cost))
                )
                if cost > result[r, c]:
                    result[r, c] = cost

    return result
