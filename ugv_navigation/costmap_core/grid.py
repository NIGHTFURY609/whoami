"""ROS-independent ground position -> costmap grid cell conversion.

Implements the next stage of the pipeline:

    ground/world position (x, y)
        -> costmap grid geometry
        -> grid cell (row, column)

This module does not know about semantic classes, cost values, or camera
geometry -- it only answers "which grid cell does this world point fall
into?" for a single point at a time.

No project-specific costmap resolution is defined here (see
PROJECT_CONTEXT.md, which forbids inventing final configuration values):
`CostmapGridGeometry` has no default resolution, so callers must supply it
explicitly (e.g. from configuration, once the team settles on one).
"""

import math
from dataclasses import dataclass


class GridError(ValueError):
    """Raised when grid geometry or a world position is invalid.

    Covers: invalid resolution, invalid grid dimensions, non-finite world
    coordinates, and world positions that fall outside the grid's extent.
    """


@dataclass(frozen=True)
class CostmapGridGeometry:
    """Geometry of a 2D costmap grid.

    Coordinate convention (see module docstring for the full explanation):
    - `resolution`: size of one grid cell, in metres/cell. Must be > 0.
    - `origin_x`, `origin_y`: world-frame coordinates of the grid's
      lower-left corner -- i.e. the minimum-x, minimum-y corner of cell
      (row=0, col=0). This matches the convention used by
      nav_msgs/OccupancyGrid.
    - `width`, `height`: grid size in cells (columns, rows respectively).
      Both must be positive integers.

    The grid therefore covers the half-open world-frame rectangle:
        x in [origin_x, origin_x + width * resolution)
        y in [origin_y, origin_y + height * resolution)
    """

    resolution: float
    origin_x: float
    origin_y: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if not (math.isfinite(self.resolution) and self.resolution > 0):
            raise GridError(
                f"Invalid resolution: {self.resolution!r}. "
                "resolution must be a finite value > 0 (metres/cell)."
            )
        if not (math.isfinite(self.origin_x) and math.isfinite(self.origin_y)):
            raise GridError(
                f"Invalid origin: ({self.origin_x!r}, {self.origin_y!r}). "
                "origin_x and origin_y must be finite."
            )
        if not (isinstance(self.width, int) and not isinstance(self.width, bool) and self.width > 0):
            raise GridError(f"Invalid width: {self.width!r}. width must be a positive integer.")
        if not (isinstance(self.height, int) and not isinstance(self.height, bool) and self.height > 0):
            raise GridError(f"Invalid height: {self.height!r}. height must be a positive integer.")


@dataclass(frozen=True)
class GridCell:
    """A discrete cell index into a costmap grid.

    `row` increases with world y (row 0 is closest to origin_y).
    `col` increases with world x (col 0 is closest to origin_x).
    """

    row: int
    col: int


def world_to_grid_cell(x: float, y: float, geometry: CostmapGridGeometry) -> GridCell:
    """Convert a world/ground position into a costmap grid cell.

    Raises GridError if:
    - x or y is not finite, or
    - the position falls outside the grid's extent (never silently
      clamped into the nearest cell).
    """
    if not (math.isfinite(x) and math.isfinite(y)):
        raise GridError(f"Invalid world position: x={x!r}, y={y!r}. Both must be finite.")

    col = math.floor((x - geometry.origin_x) / geometry.resolution)
    row = math.floor((y - geometry.origin_y) / geometry.resolution)

    if not (0 <= col < geometry.width) or not (0 <= row < geometry.height):
        raise GridError(
            f"World position ({x!r}, {y!r}) is outside the grid: "
            f"resolved to cell (row={row}, col={col}), but grid extent is "
            f"col in [0, {geometry.width}), row in [0, {geometry.height})."
        )

    return GridCell(row=row, col=col)
