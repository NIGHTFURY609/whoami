"""Measure TF edge cadence and exit 0 (pass) / 1 (fail). Contract: >= 15 Hz, jitter < 50 ms.

    ros2 run ugv_localization tf_rate_check --ros-args -p duration_s:=20.0
    ros2 run ugv_localization tf_rate_check --ros-args -p use_sim_time:=true   # bag / sim
"""

from __future__ import annotations

import sys
from collections import defaultdict

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

from ugv_localization.rosconv import stamp_to_ns
from ugv_localization.tfcheck import TfCheckProfile, edge_stats


class TfRateCheck(Node):
    def __init__(self) -> None:
        super().__init__("tf_rate_check")
        edges = self.declare_parameter("edges", ["map:odom", "odom:base_link"]).value
        self._duration_ns = int(self.declare_parameter("duration_s", 20.0).value * 1e9)
        self._profile = TfCheckProfile(
            min_rate_hz=float(self.declare_parameter("min_rate_hz", 15.0).value),
            max_jitter_s=float(self.declare_parameter("max_jitter_s", 0.05).value),
        )
        self._edges = [tuple(e.split(":", 1)) for e in edges]
        self._stamps: dict[tuple[str, str], list[int]] = defaultdict(list)
        self._start_ns: int | None = None
        self.exit_code: int | None = None
        self.create_subscription(TFMessage, "/tf", self._on_tf, 100)
        self.create_timer(0.5, self._check_done)

    def _on_tf(self, msg: TFMessage) -> None:
        for t in msg.transforms:
            key = (t.header.frame_id.lstrip("/"), t.child_frame_id.lstrip("/"))
            if key in self._edges:
                try:
                    self._stamps[key].append(stamp_to_ns(t.header.stamp))
                except (TypeError, ValueError):
                    pass

    def _check_done(self) -> None:
        now = self.get_clock().now().nanoseconds
        if now <= 0:
            return
        if self._start_ns is None:
            self._start_ns = now
            self.get_logger().info(f"sampling {self._edges} for {self._duration_ns / 1e9:.1f} s")
            return
        if now - self._start_ns < self._duration_ns:
            return
        ok = True
        for parent, child in self._edges:
            s = edge_stats(parent, child, self._stamps[(parent, child)], self._profile)
            ok &= s.passed
            print(
                f"{parent}->{child}: {'PASS' if s.passed else 'FAIL'} "
                f"rate={s.rate_hz:.1f}Hz jitter_p95={s.jitter_p95_s * 1e3:.1f}ms "
                f"max_gap={s.max_gap_s * 1e3:.1f}ms n={s.samples} dup={s.duplicates} {list(s.reasons)}"
            )
        self.exit_code = 0 if ok else 1


def main(args: list[str] | None = None) -> None:
    code = 1
    try:
        with rclpy.init(args=args):
            node = TfRateCheck()
            while rclpy.ok() and node.exit_code is None:
                rclpy.spin_once(node, timeout_sec=0.1)
            code = 1 if node.exit_code is None else node.exit_code
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    sys.exit(code)
