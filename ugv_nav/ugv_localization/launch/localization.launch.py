"""Dev 2 localization stack: odom_tf_bridge + RTAB-Map (mono) + pose_validity.

    ros2 launch ugv_localization localization.launch.py mode:=mapping profile:=sim
    ros2 launch ugv_localization localization.launch.py mode:=localize profile:=sim
    ros2 launch ugv_localization localization.launch.py mode:=mapping fresh_db:=true

TF chain owned here (mindmap D6): odom->base_link (odom_tf_bridge), map->odom (rtabmap).
Camera driver + /wheel/odom come from Dev 5 bringup; this file never starts a camera.
"""

from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ugv_localization.modes import Mode, parse_mode, plan_mode

# profile -> default use_sim_time (architecture §4)
_PROFILES = {"live_cam": False, "sim": True, "bag": True}


def _to_bool(text: str, name: str) -> bool:
    low = text.strip().lower()
    if low in ("true", "1", "yes"):
        return True
    if low in ("false", "0", "no"):
        return False
    raise RuntimeError(f"{name} must be true/false, got {text!r}")


def _setup(context, *args, **kwargs):
    def arg(name: str) -> str:
        return LaunchConfiguration(name).perform(context)

    profile = arg("profile")
    if profile not in _PROFILES:
        raise RuntimeError(f"profile must be one of {sorted(_PROFILES)}, got {profile!r}")
    ust = arg("use_sim_time")
    use_sim_time = _PROFILES[profile] if ust == "auto" else _to_bool(ust, "use_sim_time")

    mode = parse_mode(arg("mode"))
    db = os.path.expanduser(arg("database_path"))
    if mode is Mode.MAPPING:
        Path(db).parent.mkdir(parents=True, exist_ok=True)
    plan = plan_mode(mode, db, fresh=_to_bool(arg("fresh_db"), "fresh_db"))  # fails fast

    cfg = os.path.join(get_package_share_directory("ugv_localization"), "config")
    image_topic = arg("image_topic")
    info_topic = arg("camera_info_topic")
    common = {"use_sim_time": use_sim_time}

    return [
        LogInfo(
            msg=f"[ugv_localization] profile={profile} mode={mode.value} db={plan.database_path} "
            f"fresh={bool(plan.rtabmap_args)} use_sim_time={use_sim_time}"
        ),
        Node(
            package="ugv_localization",
            executable="odom_tf_bridge",
            name="odom_tf_bridge",
            output="screen",
            parameters=[{**common, "profile_path": os.path.join(cfg, "odom_bridge.yaml")}],
            remappings=[("wheel/odom", arg("wheel_odom_topic")), ("odom", "/odom")],
        ),
        Node(
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            namespace="rtabmap",
            output="screen",
            parameters=[
                os.path.join(cfg, "rtabmap_mono.yaml"),
                {**common, "database_path": plan.database_path, **plan.rtabmap_params()},
            ],
            arguments=list(plan.rtabmap_args),
            remappings=[
                ("rgb/image", image_topic),
                ("rgb/camera_info", info_topic),
                ("odom", "/odom"),
            ],
        ),
        Node(
            package="ugv_localization",
            executable="pose_validity_node",
            name="pose_validity",
            output="screen",
            parameters=[
                {
                    **common,
                    "profile_path": os.path.join(cfg, "pose_validity.yaml"),
                    "mode": plan.mode.value,
                }
            ],
            remappings=[
                ("odom", "/odom"),
                ("camera_info", info_topic),
                ("rtabmap/info", "/rtabmap/info"),
            ],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("mode", default_value="mapping", description="mapping | localize"),
            DeclareLaunchArgument("database_path", default_value="~/.ros/ugv/rtabmap.db"),
            DeclareLaunchArgument("fresh_db", default_value="false", description="mapping only: delete db at start"),
            DeclareLaunchArgument("profile", default_value="live_cam", description="live_cam | sim | bag"),
            DeclareLaunchArgument("use_sim_time", default_value="auto", description="auto = from profile"),
            # Topic names pending Dev 5 confirmation (docs/localization/interfaces.md).
            DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
            DeclareLaunchArgument("camera_info_topic", default_value="/camera/camera_info"),
            DeclareLaunchArgument("wheel_odom_topic", default_value="/wheel/odom"),
            OpaqueFunction(function=_setup),
        ]
    )
