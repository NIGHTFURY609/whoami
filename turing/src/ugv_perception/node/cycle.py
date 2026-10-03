"""One perception cycle: optional decode → compose_tick. No rclpy."""

from __future__ import annotations

from ugv_perception.adapter.frame import ImageFrame
from ugv_perception.compose.tick import ComposeOut, compose_tick
from ugv_perception.confidence.profile import GateProfile
from ugv_perception.freshness.profile import FreshnessProfile
from ugv_perception.ingest.decode import decode_frame
from ugv_perception.ingest.msgs import CameraInfoView, ImageView
from ugv_perception.remap.table import RemapTable


def decode_cycle_frame(
    image: ImageView | None, camera_info: CameraInfoView | None
) -> ImageFrame | None:
    """The decode step of a cycle. None when a half of the pair is missing or the pair is invalid."""
    if image is None or camera_info is None:
        return None
    try:
        return decode_frame(image, camera_info)
    except (TypeError, ValueError):
        return None


def cycle_on_frame(
    *,
    frame: ImageFrame | None,
    now_ns: int,
    adapter: object,
    remap_table: RemapTable,
    gate_profile: GateProfile,
    freshness_profile: FreshnessProfile,
    now_ns_after: int | None = None,
) -> ComposeOut:
    """The cycle after the decode, so a caller that already holds the frame does not decode it again."""
    return compose_tick(
        frame=frame,
        now_ns=now_ns,
        adapter=adapter,
        remap_table=remap_table,
        gate_profile=gate_profile,
        freshness_profile=freshness_profile,
        now_ns_after=now_ns_after,
    )


def perception_cycle(
    *,
    image: ImageView | None,
    camera_info: CameraInfoView | None,
    now_ns: int,
    adapter: object,
    remap_table: RemapTable,
    gate_profile: GateProfile,
    freshness_profile: FreshnessProfile,
    now_ns_after: int | None = None,
) -> ComposeOut:
    return cycle_on_frame(
        frame=decode_cycle_frame(image, camera_info),
        now_ns=now_ns,
        adapter=adapter,
        remap_table=remap_table,
        gate_profile=gate_profile,
        freshness_profile=freshness_profile,
        now_ns_after=now_ns_after,
    )
