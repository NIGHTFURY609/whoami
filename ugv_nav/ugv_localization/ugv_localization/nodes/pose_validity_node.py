"""/ugv/pose_valid publisher (architecture §10.1, §12; consumed by Dev 5 level-3 hold).

Subscribes : odom          nav_msgs/Odometry      (gated wheel odom from odom_tf_bridge)
             camera_info   sensor_msgs/CameraInfo (SLAM camera; freshness proxy for images)
             rtabmap/info  rtabmap_msgs/Info      (alive + visual constraint)
             TF map->base_link, map->odom
Publishes  : /ugv/pose_valid          std_msgs/Bool   every tick (heartbeat), false at startup
             /ugv/localization_status std_msgs/String reasons, on change
Params     : profile_path, mode (mapping|localize), map_frame, odom_frame, base_frame
"""

from __future__ import annotations

import rclpy
from rclpy.executors import ExternalShutdownException
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rtabmap_msgs.msg import Info
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener

from ugv_localization.modes import parse_mode
from ugv_localization.rosconv import (
    camera_info_ok,
    odom_variances,
    slam_info_visual_constraint,
    stamp_to_ns,
    transform_to_xy_yaw,
)
from ugv_localization.validity import PoseValidityMonitor, load_validity_profile


class PoseValidityNode(Node):
    def __init__(self) -> None:
        super().__init__("pose_validity")
        profile_path = self.declare_parameter("profile_path", "").value
        if not profile_path:
            raise RuntimeError("profile_path parameter is required")
        mode = parse_mode(self.declare_parameter("mode", "").value)
        self._map = self.declare_parameter("map_frame", "map").value
        self._odom = self.declare_parameter("odom_frame", "odom").value
        self._base = self.declare_parameter("base_frame", "base_link").value

        profile = load_validity_profile(profile_path)
        self._monitor = PoseValidityMonitor(profile, mode)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._last_m2o_ns: int | None = None
        self._last_status: str | None = None

        self._pub_valid = self.create_publisher(Bool, "/ugv/pose_valid", 10)
        self._pub_status = self.create_publisher(String, "/ugv/localization_status", 10)
        self.create_subscription(Odometry, "odom", self._on_odom, 20)
        self.create_subscription(CameraInfo, "camera_info", self._on_camera_info, qos_profile_sensor_data)
        self.create_subscription(Info, "rtabmap/info", self._on_info, 10)
        self.create_timer(1.0 / profile.publish_rate_hz, self._tick)
        self.get_logger().info(f"pose validity: mode={mode.value}, publishing /ugv/pose_valid")

    def _on_odom(self, msg: Odometry) -> None:
        try:
            ns = stamp_to_ns(msg.header.stamp)
        except (TypeError, ValueError):
            return
        p = msg.pose.pose.position
        self._monitor.on_odom(ns, float(p.x), float(p.y), *odom_variances(msg))

    def _on_camera_info(self, msg: CameraInfo) -> None:
        try:
            ns = stamp_to_ns(msg.header.stamp)
        except (TypeError, ValueError):
            return
        self._monitor.on_camera_info(ns, camera_info_ok(msg))

    def _on_info(self, msg: Info) -> None:
        try:
            ns = stamp_to_ns(msg.header.stamp)
        except (TypeError, ValueError):
            return
        self._monitor.on_slam_info(ns, slam_info_visual_constraint(msg))

    def _poll_tf(self) -> None:
        try:
            t = self._tf_buffer.lookup_transform(self._map, self._base, Time())
            self._monitor.on_tf(stamp_to_ns(t.header.stamp))
        except (TransformException, TypeError, ValueError):
            pass  # absence shows up as tf_missing / tf_stale
        try:
            m2o = self._tf_buffer.lookup_transform(self._map, self._odom, Time())
            ns = stamp_to_ns(m2o.header.stamp)
            if ns != self._last_m2o_ns:
                self._last_m2o_ns = ns
                self._monitor.on_map_to_odom(ns, *transform_to_xy_yaw(m2o.transform))
        except (TransformException, TypeError, ValueError):
            pass

    def _tick(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if now_ns <= 0:
            self._pub_valid.publish(Bool(data=False))  # sim clock not started: fail closed
            return
        self._poll_tf()
        res = self._monitor.evaluate(now_ns)
        self._pub_valid.publish(Bool(data=res.valid))
        status = "valid" if res.valid else ",".join(r.value for r in res.reasons)
        if status != self._last_status:
            self._last_status = status
            self._pub_status.publish(String(data=status))
            log = self.get_logger().info if res.valid else self.get_logger().warning
            log(f"pose_valid={res.valid} ({status})")


def main(args: list[str] | None = None) -> None:
    # Context manager owns shutdown; calling destroy_node() after SIGINT raises on Lyrical.
    try:
        with rclpy.init(args=args):
            rclpy.spin(PoseValidityNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
