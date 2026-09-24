"""Semantic mask -> costmap projection.

Bridges the pieces that already exist as separate, independently-tested
stages into the one pipeline described in PROJECT_CONTEXT.md's Dev 3 flow:

    semantic mask (per-pixel class ids)
        -> pixel (u, v)
        -> camera ray                (projection.pixel_to_camera_ray)
        -> ground point               (projection.camera_ray_to_ground_point)
        -> costmap grid cell          (grid.world_to_grid_cell)
        -> semantic cost written into that cell (class_to_cost.class_to_cost)

This module is ROS-independent: it does not touch sensor_msgs, TF, real
camera calibration, Depth Anything, or inflation. It only combines numpy
arrays and the existing plain-Python/numpy geometry helpers.
"""

from __future__ import annotations

import numpy as np

from costmap_core.class_to_cost import (
    CostValues,
    DEFAULT_COST_VALUES,
    InvalidSemanticClassError,
    SemanticClass,
    class_to_cost,
)
from costmap_core.grid import CostmapGridGeometry, GridError, world_to_grid_cell
from costmap_core.projection import (
    CameraGroundGeometry,
    CameraIntrinsics,
    ProjectionError,
    project_pixel_to_ground,
)

# Precedence used when more than one mask pixel projects into the same cell:
# the class with the higher rank wins, i.e. HAZARD > UNKNOWN > TRAVERSABLE.
# This is decided on semantic classes, NOT on numeric costs: with the default
# costs UNKNOWN (255) is numerically above HAZARD (254), so a numeric max
# would let "no information" overwrite known hazard evidence.
SEMANTIC_CLASS_PRECEDENCE = {
    SemanticClass.TRAVERSABLE: 0,
    SemanticClass.UNKNOWN: 1,
    SemanticClass.HAZARD: 2,
}


def project_mask_to_costmap(
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    geometry: CameraGroundGeometry,
    grid_geometry: CostmapGridGeometry,
    cost_values: CostValues = DEFAULT_COST_VALUES,
) -> np.ndarray:
    """Project a 2D semantic class mask into a costmap grid.

    Args:
        mask: 2D array-like of canonical semantic class ids (0=unknown,
            1=traversable, 2=hazard), one entry per image pixel. Pixel
            (row, col) is treated as image coordinates (u=col, v=row) --
            the same convention `projection.py` expects.
        intrinsics: pinhole camera intrinsics (see `projection.py`).
        geometry: camera pose relative to the ground plane (see
            `projection.py`).
        grid_geometry: costmap grid geometry (see `grid.py`).
        cost_values: cost values for each canonical class; defaults to
            `class_to_cost.DEFAULT_COST_VALUES`.

    Returns:
        A 2D numpy array of shape (grid_geometry.height,
        grid_geometry.width) and dtype int64.

        Cells that no mask pixel projects into keep
        `cost_values.unknown_cost`. This mirrors the existing convention
        (class 0 = "never free") elsewhere in this package: "no
        perception data reached this cell" is treated the same as
        "perception says unknown", not as "free".

        When more than one mask pixel projects into the same cell, the
        winning semantic class is chosen by `SEMANTIC_CLASS_PRECEDENCE`
        (HAZARD > UNKNOWN > TRAVERSABLE) and only then converted to a
        cost. A hazard pixel is therefore never overwritten by an unknown
        pixel, regardless of the numeric cost values.

    Pixel handling (never raises for these; the pixel is skipped instead):
        - A pixel whose camera ray does not intersect the ground plane
          (`ProjectionError` from `projection.py`, e.g. a "sky" pixel
          under a downward-pitched camera) is skipped.
        - A pixel whose ground point falls outside `grid_geometry`
          (`GridError` from `grid.py`) is skipped -- it is never clamped
          into the nearest in-bounds cell.

    Raises:
        InvalidSemanticClassError: if `mask` is not 2D, or any pixel
            holds a class id outside {0, 1, 2}. This is checked for
            every pixel regardless of whether that pixel's ray would
            have been skipped, so a malformed mask always fails loudly.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise InvalidSemanticClassError(
            f"Expected a 2D semantic class mask, got array with shape "
            f"{mask.shape} (ndim={mask.ndim})."
        )

    shape = (grid_geometry.height, grid_geometry.width)
    costmap = np.full(shape, cost_values.unknown_cost, dtype=np.int64)
    # Precedence rank of the class currently held by each cell; -1 = untouched.
    winning_rank = np.full(shape, -1, dtype=np.int64)

    height, width = mask.shape
    for row in range(height):
        for col in range(width):
            class_id = int(mask[row, col])
            cost = class_to_cost(class_id, cost_values)
            rank = SEMANTIC_CLASS_PRECEDENCE[SemanticClass(class_id)]

            try:
                ground_point = project_pixel_to_ground(
                    float(col), float(row), intrinsics, geometry
                )
            except ProjectionError:
                continue  # ray does not intersect the ground; skip this pixel

            try:
                cell = world_to_grid_cell(ground_point.x, ground_point.y, grid_geometry)
            except GridError:
                continue  # outside the grid; skip, never clamp

            if rank > winning_rank[cell.row, cell.col]:
                costmap[cell.row, cell.col] = cost
                winning_rank[cell.row, cell.col] = rank

    return costmap
