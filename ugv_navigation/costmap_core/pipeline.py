"""Single ROS-independent entry point for one Dev 3 costmap update.

Runs the existing, independently-tested stages in order on a validated
`contracts.CostmapCoreInputs`:

    semantic mask
        -> pixel/ground/grid projection    (mask_projection.project_mask_to_costmap)
        -> semantic costmap
        -> optional geometry costmap       (geometry_costmap.build_geometry_costmap)
        -> geometry precedence             (costmap_fusion.fuse_costmaps)
        -> inflation                       (inflation.inflate_costmap)
    optional footprint -> optional padding (footprint.pad_footprint)

No stage logic lives here -- this module only wires the stages together,
so geometry-lethal-wins, semantic class precedence (HAZARD > UNKNOWN >
TRAVERSABLE per cell) and "unknown is never free" are exactly the rules
of the underlying modules.

No real-world value has a default: `inflation_radius` must be supplied,
and footprint padding is only applied when explicitly given.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from costmap_core.class_to_cost import DEFAULT_COST_VALUES, CostValues
from costmap_core.contracts import CostmapCoreInputs
from costmap_core.costmap_fusion import fuse_costmaps
from costmap_core.footprint import Point, pad_footprint
from costmap_core.geometry_costmap import (
    DEFAULT_GEOMETRY_COST_VALUES,
    GeometryCostValues,
    build_geometry_costmap,
)
from costmap_core.inflation import inflate_costmap
from costmap_core.mask_projection import project_mask_to_costmap


class PipelineError(ValueError):
    """Raised when the pipeline refuses to run on otherwise well-formed inputs.

    Covers: `inputs` not being a `CostmapCoreInputs`, a mask Dev 1 flagged
    as invalid, and footprint padding requested without a footprint.
    Deliberately never coerced into a costmap.
    """


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class CostmapPipelineResult:
    """Every stage output of one pipeline run. All arrays are read-only and
    grid-shaped (grid.height, grid.width), dtype int64.

    semantic: projected semantic costmap (unseen cells = unknown cost).
    geometry: geometry costmap, or None when no occupancy was supplied.
    fused: semantic after geometry precedence; the same array as `semantic`
        when no occupancy was supplied (there is nothing to fuse).
    final: `fused` after inflation. This is the costmap to publish.
    footprint: the input footprint (padded if padding was given), or None
        when no footprint was supplied. It is in the footprint's own robot
        frame and is NOT applied to any costmap cell.
    """

    semantic: np.ndarray
    geometry: Optional[np.ndarray]
    fused: np.ndarray
    final: np.ndarray
    footprint: Optional[tuple[Point, ...]]


def run_costmap_pipeline(
    inputs: CostmapCoreInputs,
    *,
    inflation_radius: float,
    footprint_padding: Optional[float] = None,
    cost_values: CostValues = DEFAULT_COST_VALUES,
    geometry_cost_values: GeometryCostValues = DEFAULT_GEOMETRY_COST_VALUES,
) -> CostmapPipelineResult:
    """Run one full costmap update.

    Args:
        inputs: validated, frame/shape-consistent inputs (see contracts.py).
            `inputs.occupancy` and `inputs.footprint` are optional.
        inflation_radius: inflation buffer in metres, finite and >= 0.
            Required -- no project value is defined yet. 0 disables it.
        footprint_padding: optional outward padding in metres for
            `inputs.footprint`. Requires a footprint.
        cost_values: semantic class -> cost values.
        geometry_cost_values: geometry cost values. Its `lethal_cost` is
            the value fusion treats as geometric lethal and inflation
            inflates around.

    Raises:
        PipelineError: if `inputs` is not a CostmapCoreInputs, if
            `inputs.mask.valid` is False (an invalid mask must not be
            treated as current -- architecture §8.4/§8.6; the fail-safe
            response is the caller's job), if `footprint_padding` is
            given without `inputs.footprint`, or if
            `cost_values.hazard_cost` != `geometry_cost_values.lethal_cost`
            (fusion and inflation use one lethal value; a mismatch would
            silently stop semantic hazards being treated as lethal).
        InflationError / FootprintError: propagated unchanged from the
            underlying stage for an invalid radius or padding.
    """
    if not isinstance(inputs, CostmapCoreInputs):
        raise PipelineError(
            f"inputs must be a CostmapCoreInputs, got {type(inputs).__name__}"
        )
    if not inputs.mask.valid:
        raise PipelineError(
            "mask.valid is False; refusing to build a costmap from an invalid mask."
        )
    if footprint_padding is not None and inputs.footprint is None:
        raise PipelineError("footprint_padding was given but inputs.footprint is None.")
    if cost_values.hazard_cost != geometry_cost_values.lethal_cost:
        raise PipelineError(
            f"cost_values.hazard_cost ({cost_values.hazard_cost!r}) must equal "
            f"geometry_cost_values.lethal_cost ({geometry_cost_values.lethal_cost!r}); "
            "the pipeline uses one lethal value for fusion and inflation."
        )

    grid = inputs.grid.geometry
    lethal_cost = geometry_cost_values.lethal_cost

    semantic = _readonly(
        project_mask_to_costmap(
            inputs.mask.classes,
            inputs.intrinsics.intrinsics,
            inputs.camera_ground.geometry,
            grid,
            cost_values,
        )
    )

    if inputs.occupancy is None:
        geometry = None
        fused = semantic
    else:
        geometry = _readonly(
            build_geometry_costmap(inputs.occupancy.occupied, geometry_cost_values)
        )
        fused = _readonly(fuse_costmaps(semantic, geometry, lethal_cost))

    final = _readonly(inflate_costmap(fused, grid.resolution, inflation_radius, lethal_cost))

    if inputs.footprint is None:
        footprint = None
    elif footprint_padding is None:
        footprint = inputs.footprint.vertices
    else:
        footprint = tuple(pad_footprint(inputs.footprint.vertices, footprint_padding))

    return CostmapPipelineResult(
        semantic=semantic,
        geometry=geometry,
        fused=fused,
        final=final,
        footprint=footprint,
    )
