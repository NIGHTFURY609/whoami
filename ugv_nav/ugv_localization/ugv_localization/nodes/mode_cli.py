"""Runtime RTAB-Map control: switch mapping/localize without restart, or back up the database.

    ros2 run ugv_localization mode_cli localize
    ros2 run ugv_localization mode_cli mapping
    ros2 run ugv_localization mode_cli backup
Note: the pose_validity node's `mode` param is set at launch. After a runtime switch to
localize, restart with mode:=localize for the relocalization gate to apply.
"""

from __future__ import annotations

import argparse
import sys

import rclpy
from std_srvs.srv import Empty

_SERVICES = {
    "localize": "set_mode_localization",
    "mapping": "set_mode_mapping",
    "backup": "backup",
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=sorted(_SERVICES))
    # rtabmap_slam (0.23, Lyrical) advertises services as private node services: /<ns>/<node>/...
    # (topics stay at /<ns>/..., e.g. /rtabmap/info).
    ap.add_argument("--namespace", default="/rtabmap/rtabmap")
    ap.add_argument("--timeout", type=float, default=5.0)
    args, ros_args = ap.parse_known_args(argv if argv is not None else sys.argv[1:])

    rclpy.init(args=ros_args)
    node = rclpy.create_node("ugv_mode_cli")
    name = f"{args.namespace.rstrip('/')}/{_SERVICES[args.action]}"
    client = node.create_client(Empty, name)
    code = 1
    try:
        if not client.wait_for_service(timeout_sec=args.timeout):
            print(f"service {name} not available", file=sys.stderr)
        else:
            future = client.call_async(Empty.Request())
            rclpy.spin_until_future_complete(node, future, timeout_sec=args.timeout)
            if future.done() and future.exception() is None:
                print(f"{args.action}: ok ({name})")
                code = 0
            else:
                print(f"{args.action}: failed ({name})", file=sys.stderr)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    sys.exit(code)
