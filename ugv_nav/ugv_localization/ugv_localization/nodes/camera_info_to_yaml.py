"""Capture ONE real CameraInfo and write it to config/cameras/<name>.yaml.

The only sanctioned way to create a camera YAML besides `camera_calibration` output:
from an actual driver / Gazebo sensor, never typed by hand. Refuses zero / fake K.

    ros2 run ugv_localization camera_info_to_yaml --topic /camera/camera_info \
        --name sim_front_mono --out ugv_nav/config/cameras/sim_front_mono.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rclpy
import yaml
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo

from ugv_localization.camera import CalibrationError, calibration_to_yaml_dict
from ugv_localization.rosconv import calibration_from_camera_info_msg


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", default="/camera/camera_info")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--force", action="store_true", help="overwrite an existing file")
    args, ros_args = ap.parse_known_args(argv if argv is not None else sys.argv[1:])

    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} exists; pass --force to overwrite", file=sys.stderr)
        sys.exit(1)

    rclpy.init(args=ros_args)
    node = rclpy.create_node("camera_info_to_yaml")
    got: list[CameraInfo] = []
    node.create_subscription(CameraInfo, args.topic, got.append, qos_profile_sensor_data)
    deadline = node.get_clock().now().nanoseconds + int(args.timeout * 1e9)
    try:
        while rclpy.ok() and not got and node.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    if not got:
        print(f"no CameraInfo on {args.topic} within {args.timeout}s", file=sys.stderr)
        sys.exit(1)

    msg = got[0]
    try:
        cal = calibration_from_camera_info_msg(msg, args.name)
    except CalibrationError as exc:
        print(f"refusing to write: {exc}", file=sys.stderr)
        sys.exit(1)
    header = (
        f"# Captured from {args.topic} (frame_id={msg.header.frame_id}, "
        f"stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}). Do not hand-edit.\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + yaml.safe_dump(calibration_to_yaml_dict(cal), sort_keys=False), encoding="utf-8")
    print(f"wrote {out}")
