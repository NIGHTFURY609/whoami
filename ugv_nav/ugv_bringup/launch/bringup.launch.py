"""Full product bringup (architecture §4, dev.md Dev 5 task 5).

    ros2 launch ugv_bringup bringup.launch.py profile:=live_cam \
        calibration_file:=<camera yaml> device:=<V4L2 path or stream URL> \
        camera_x:=.. camera_y:=.. camera_z:=.. camera_pitch_deg:=.. \
        perception_src:=<repo>/turing/src [mode:=mapping|localize] [robot:=primary] \
        [transport_latency_s:=<seconds, network camera>] [allow_placeholder_calibration:=true] \
        [map_assembler:=true|false]

allow_placeholder_calibration (default false): the camera driver refuses a calibration flagged `placeholder: true`
(another camera's K) and the whole camera chain stays silent, so the arbiter holds. true loads it with a WARN, for
bring-up only (ugv_bringup README).

live_cam starts, in the order the data flows:
  camera driver (Dev 5)          /camera/image_raw + /camera/camera_info (+ UI stream)
  robot description (Dev 5)      base_link -> camera_link -> camera_optical_frame
  perception (Dev 1)             /segmentation/mask, /ugv/perception_degraded, /perception/depth/image + cloud
  localization (Dev 2)           RTAB-Map RGB-D on DA3 depth: map -> odom -> base_link, /ugv/pose_valid
  semantic costmap (Dev 3)       /semantic_costmap/grid for the Nav2 costmap StaticLayer
  navigation (Dev 4)             Smac2D + RPP -> /cmd_vel_nav2, /ugv/nav2_heartbeat
  safety arbiter (Dev 5)         the only /cmd_vel publisher
  operator API (ugv_api)         http://<api_host>:<api_port> for the UI

Dev 1's perception is not a colcon package (turing/ is a plain Python project), so it runs as
`python3 -m ugv_perception.node.adapter_node` with perception_src on PYTHONPATH; its config and weights
resolve from that tree (turing/config, turing/weights).

Profiles sim and bag are not wired here yet: this file refuses them instead of half-starting.
"""

from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, GroupAction, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

_PROFILES = ("live_cam",)
_MOUNT = ("camera_x", "camera_y", "camera_z", "camera_pitch_deg")


def _include(pkg: str, launch_file: str, args: dict[str, str]) -> GroupAction:
    """One subsystem launch in its own scope.

    Launch configurations leak out of an included file: Dev 2's `use_sim_time:=auto` would reach the safety and
    API launches, which parse it as a bool and abort the whole bringup. Each include gets only `args`.
    """
    path = Path(get_package_share_directory(pkg)) / "launch" / launch_file
    include = IncludeLaunchDescription(PythonLaunchDescriptionSource(str(path)), launch_arguments=args.items())
    return GroupAction([include], scoped=True, forwarding=False)


def _setup(context, *args, **kwargs):
    def arg(name: str) -> str:
        return LaunchConfiguration(name).perform(context)

    profile = arg("profile")
    if profile not in _PROFILES:
        raise RuntimeError(f"bringup.launch.py: profile {profile!r} is not wired yet (available: {_PROFILES})")
    if not arg("calibration_file"):
        raise RuntimeError("bringup.launch.py: calibration_file is required (calibrate first: ugv_bringup README)")
    src = Path(os.path.expanduser(arg("perception_src")))
    if not (src / "ugv_perception" / "node" / "adapter_node.py").is_file():
        raise RuntimeError(f"bringup.launch.py: perception_src {src} does not hold ugv_perception (turing/src)")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(src), env.get("PYTHONPATH", "")) if p)
    actions = [
        _include("ugv_bringup", "camera.launch.py",
                 {"calibration_file": arg("calibration_file"), "device": arg("device"),
                  "transport_latency_s": arg("transport_latency_s"),
                  "allow_placeholder_calibration": arg("allow_placeholder_calibration")}),
        _include("ugv_robot_description", "description.launch.py", {a: arg(a) for a in _MOUNT}),
        ExecuteProcess(
            cmd=["python3", "-m", "ugv_perception.node.adapter_node", "--ros-args", "-p",
                 f"adapter:={arg('adapter')}"],
            name="ugv_perception", output="screen", additional_env={"PYTHONPATH": env["PYTHONPATH"]},
        ),
        _include("ugv_localization", "localization.launch.py",
                 {"mode": arg("mode"), "profile": profile, "database_path": arg("database_path"),
                  "timing": arg("localization_timing"), "calibration_file": arg("calibration_file"),
                  "map_assembler": arg("map_assembler")}),
        _include("ugv_costmap", "semantic_costmap.launch.py", {}),
        _include("ugv_navigation", "navigation.launch.py", {"robot": arg("robot")}),
        _include("ugv_safety", "safety.launch.py", {}),
        _include("ugv_api", "api.launch.py", {"host": arg("api_host"), "port": arg("api_port")}),
    ]
    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("profile", default_value="live_cam"),
        DeclareLaunchArgument("calibration_file", default_value=""),
        DeclareLaunchArgument("device", default_value="/dev/video0"),
        DeclareLaunchArgument("transport_latency_s", default_value="0.0",
                              description="measured delay of a network camera stream, seconds (ugv_bringup README)"),
        DeclareLaunchArgument("allow_placeholder_calibration", default_value="false",
                              description="true: load a calibration flagged placeholder (bring-up only; refused otherwise)"),
        *[DeclareLaunchArgument(a, default_value="", description="measured camera mount (required)")
          for a in _MOUNT],
        DeclareLaunchArgument("perception_src", default_value=os.environ.get("UGV_PERCEPTION_SRC", "")),
        DeclareLaunchArgument("adapter", default_value="rugd"),
        DeclareLaunchArgument("mode", default_value="mapping"),
        DeclareLaunchArgument("database_path", default_value="~/.ros/ugv/rtabmap.db"),
        DeclareLaunchArgument("robot", default_value=""),
        # laptop: Dev 2 timing profile for a slow-GPU laptop (ugv_localization config/*_laptop.yaml)
        DeclareLaunchArgument("localization_timing", default_value="default"),
        # true: RTAB-Map's whole 3D map on /rtabmap/cloud_map (RViz, tools; not the web viewer, mindmap D27); off by
        # default for long missions (docs/mapping/README.md)
        DeclareLaunchArgument("map_assembler", default_value="false",
                              description="true: /rtabmap/cloud_map for RViz or tools (memory grows; ugv_localization)"),
        DeclareLaunchArgument("api_host", default_value="0.0.0.0"),
        DeclareLaunchArgument("api_port", default_value="8080"),
        OpaqueFunction(function=_setup),
    ])
