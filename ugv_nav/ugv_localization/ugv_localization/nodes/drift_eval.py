"""Record estimate (TF map->base_link) vs ground truth, write CSVs + drift report on exit.

Subscribes : ground_truth  nav_msgs/Odometry  (remap to Dev 5 sim ground-truth topic; TBD)
Params     : out_dir, map_frame, base_frame, duration_s (0 = until Ctrl-C), max_dt_ms
Outputs    : <out_dir>/est.csv, gt.csv, report.json   (same format as tools/drift_report)

    ros2 run ugv_localization drift_eval --ros-args -r ground_truth:=/model/ugv/odometry \
        -p use_sim_time:=true -p out_dir:=eval_out/run1
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from ugv_localization.drift import drift_report
from ugv_localization.rosconv import stamp_to_ns

_PENDING_MAX_AGE_NS = 2_000_000_000


class DriftEval(Node):
    def __init__(self) -> None:
        super().__init__("drift_eval")
        self._out = Path(self.declare_parameter("out_dir", "eval_out/drift").value)
        self._map = self.declare_parameter("map_frame", "map").value
        self._base = self.declare_parameter("base_frame", "base_link").value
        self._duration_ns = int(self.declare_parameter("duration_s", 0.0).value * 1e9)
        self._max_dt_ns = int(self.declare_parameter("max_dt_ms", 20.0).value * 1e6)
        self._buf = Buffer()
        self._listener = TransformListener(self._buf, self)
        self._gt: list[tuple[int, float, float, float]] = []
        self._est: list[tuple[int, float, float, float]] = []
        self._pending: list[int] = []
        self._start_ns: int | None = None
        self.done = False
        self.create_subscription(Odometry, "ground_truth", self._on_gt, 50)
        self.create_timer(0.1, self._resolve_pending)

    def _on_gt(self, msg: Odometry) -> None:
        try:
            ns = stamp_to_ns(msg.header.stamp)
        except (TypeError, ValueError):
            return
        p = msg.pose.pose.position
        self._gt.append((ns, float(p.x), float(p.y), float(p.z)))
        self._pending.append(ns)
        if self._start_ns is None:
            self._start_ns = ns

    def _resolve_pending(self) -> None:
        keep: list[int] = []
        newest = self._gt[-1][0] if self._gt else 0
        for ns in self._pending:
            sec, nsec = divmod(ns, 1_000_000_000)
            try:
                t = self._buf.lookup_transform(self._map, self._base, Time(seconds=sec, nanoseconds=nsec))
                tr = t.transform.translation
                self._est.append((ns, float(tr.x), float(tr.y), float(tr.z)))
            except TransformException:
                if newest - ns < _PENDING_MAX_AGE_NS:
                    keep.append(ns)  # TF may not have arrived yet
        self._pending = keep
        if self._duration_ns and self._start_ns is not None and newest - self._start_ns >= self._duration_ns:
            self.done = True

    def write(self) -> None:
        self._out.mkdir(parents=True, exist_ok=True)
        for name, rows in (("est.csv", sorted(self._est)), ("gt.csv", self._gt)):
            with (self._out / name).open("w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["t_ns", "x", "y", "z"])
                w.writerows(rows)
        est = sorted(self._est)
        try:
            rep = drift_report(
                [r[0] for r in est],
                np.asarray([r[1:] for r in est]).reshape(-1, 3),
                [r[0] for r in self._gt],
                np.asarray([r[1:] for r in self._gt]).reshape(-1, 3),
                max_dt_ns=self._max_dt_ns,
            )
        except ValueError as exc:
            self.get_logger().error(f"no report: {exc} (est={len(est)}, gt={len(self._gt)})")
            return
        report = {
            "matched": rep.matched,
            "gt_path_length_m": rep.gt_path_length_m,
            "ate_rmse_m": rep.ate_rmse_m,
            "ate_max_m": rep.ate_max_m,
            "rpe_mean_pct": {str(k): (None if math.isnan(v) else v) for k, v in rep.rpe_mean_pct.items()},
            "endpoint_error_m": rep.endpoint_error_m,
            "endpoint_drift_pct": rep.endpoint_drift_pct,
        }
        (self._out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        self.get_logger().info(f"drift report → {self._out / 'report.json'}: {report}")


def main(args: list[str] | None = None) -> None:
    node: DriftEval | None = None
    try:
        with rclpy.init(args=args):
            node = DriftEval()
            while rclpy.ok() and not node.done:
                rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.write()  # plain file I/O; safe after rclpy shutdown
