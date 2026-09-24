"""wheel/odom (Dev 5, no TF) → gated odom->base_link TF + /odom republish (mindmap D5, D6).

Subscribes : wheel/odom  nav_msgs/Odometry   (remap to Dev 5's topic)
Publishes  : odom        nav_msgs/Odometry   (only gated samples; same message, same stamp)
             TF odom->base_link              (stamp = odometry stamp, never "now")
Params     : profile_path (config/odom_bridge.yaml)
"""

from __future__ import annotations

import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

from ugv_localization.odom import OdomGate, load_odom_gate_profile
from ugv_localization.rosconv import odom_msg_to_sample


class OdomTfBridge(Node):
    def __init__(self) -> None:
        super().__init__("odom_tf_bridge")
        profile_path = self.declare_parameter("profile_path", "").value
        if not profile_path:
            raise RuntimeError("profile_path parameter is required")
        self._gate = OdomGate(load_odom_gate_profile(profile_path))
        self._tf = TransformBroadcaster(self)
        self._pub = self.create_publisher(Odometry, "odom", 20)
        self.create_subscription(Odometry, "wheel/odom", self._on_odom, 20)
        self._last_now_ns: int | None = None

    def _on_odom(self, msg: Odometry) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if now_ns <= 0:
            return  # use_sim_time and /clock not received yet
        if self._last_now_ns is not None and now_ns < self._last_now_ns:
            self.get_logger().warning("clock jumped backwards (bag loop / sim reset): resetting odom gate")
            self._gate.reset()
        self._last_now_ns = now_ns

        try:
            sample = odom_msg_to_sample(msg)
        except (TypeError, ValueError) as exc:
            self.get_logger().warning(f"dropping odom: {exc}", throttle_duration_sec=2.0)
            return
        res = self._gate.accept(sample, now_ns)
        if res.edge is None:
            self.get_logger().warning(
                f"dropping odom: {res.reason.value if res.reason else 'unknown'}",
                throttle_duration_sec=2.0,
            )
            return

        edge = res.edge
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = edge.parent
        t.child_frame_id = edge.child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = edge.translation
        (
            t.transform.rotation.x,
            t.transform.rotation.y,
            t.transform.rotation.z,
            t.transform.rotation.w,
        ) = edge.rotation
        self._tf.sendTransform(t)
        self._pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    # Context manager owns shutdown; calling destroy_node() after SIGINT raises on Lyrical.
    try:
        with rclpy.init(args=args):
            rclpy.spin(OdomTfBridge())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
