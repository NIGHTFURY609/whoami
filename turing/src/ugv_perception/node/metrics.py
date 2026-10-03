"""In-process T11 counters. Not a ROS msg. Not PortMeta."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PerceptionMetrics:
    frames_in: int = 0
    masks_published: int = 0
    infer_calls: int = 0
    degraded_true: int = 0
    degraded_false: int = 0
    depth_errors: int = 0  # depth (DA3) failures: no depth published that frame
    latencies_ns: list[int] = field(default_factory=list)
