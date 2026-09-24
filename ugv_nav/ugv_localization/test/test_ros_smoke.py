"""ROS-side smoke tests. Skipped when ROS 2 (rclpy / launch_ros / rtabmap_msgs) is not installed."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module",
    [
        "ugv_localization.nodes.odom_tf_bridge",
        "ugv_localization.nodes.pose_validity_node",
        "ugv_localization.nodes.tf_rate_check",
        "ugv_localization.nodes.drift_eval",
        "ugv_localization.nodes.mode_cli",
        "ugv_localization.nodes.camera_info_to_yaml",
    ],
)
def test_s1_nodes_import(module: str) -> None:
    pytest.importorskip("rclpy")
    pytest.importorskip("rtabmap_msgs")
    mod = importlib.import_module(module)
    assert callable(mod.main)


def test_s3_no_removed_logger_warn_alias() -> None:
    # Lyrical rclpy removed RcutilsLogger.warn (use .warning); it crashed pose_validity_node at runtime.
    offenders = [
        p.name
        for p in (_PKG / "ugv_localization" / "nodes").glob("*.py")
        if ".warn(" in p.read_text(encoding="utf-8") or "get_logger().warn\n" in p.read_text(encoding="utf-8")
    ]
    assert offenders == []


@pytest.mark.parametrize("name", ["localization.launch.py", "bag_eval.launch.py"])
def test_s2_launch_descriptions_build(name: str) -> None:
    pytest.importorskip("launch_ros")
    pytest.importorskip("ament_index_python")
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), _PKG / "launch" / name)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    ld = mod.generate_launch_description()
    assert ld.entities
