"""ROS side of the operator gateway.

Subscribes : /camera/camera_info          sensor_msgs/CameraInfo  stamp only (§12 camera)
             /segmentation/mask           sensor_msgs/Image       stamp only (§12 perception)
             /ugv/perception_degraded     std_msgs/Bool           (Dev 1 -> Dev 5)
             /ugv/pose_valid              std_msgs/Bool           (Dev 2 -> Dev 5)
             /ugv/localization_status     std_msgs/String
             /ugv/nav2_heartbeat          std_msgs/Bool           (Dev 4 -> Dev 5)
             /ugv/nav2_status             std_msgs/String         transient local
             /ugv/e_stop                  std_msgs/Bool           any publisher (CLI, this gateway)
             /ugv/safety_status           std_msgs/String         Dev 5 arbiter (latched, on change)
             /cmd_vel                     geometry_msgs/Twist     final command, read only
             TF map->base_link (polled; the pose is what GET /map/pose and the SSE `pose` event report)
             map viewer inputs, on a node of their own (ugv_api_map) in a second rclpy context (class _MapInputs):
               always on : /ugv/map/stats                               std_msgs/String (JSON)
               on demand : /rtabmap/mapPath, /global_costmap/costmap, /perception/depth_cloud (live)
Publishes  : /ugv/e_stop                  std_msgs/Bool           latched; re-published while asserted
Clients    : /navigate_to_pose            nav2_msgs/action/NavigateToPose (map-frame goals, §11)
             <rtabmap ns>/set_mode_mapping, set_mode_localization  std_srvs/Empty (§10)

Never publishes /cmd_vel or /cmd_vel_nav2 (architecture §3.1).
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Callable
from typing import Any

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.context import Context
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Bool, String
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformException, TransformListener

from ugv_api import mapcodec as codec
from ugv_api import mapsources as ms
from ugv_api import state as k
from ugv_api.errors import ServiceUnavailable
from ugv_api.goals import GoalRecord, GoalRegistry, GoalState, state_from_status, yaw_to_quaternion
from ugv_api.mapsources import MapConfig
from ugv_api.mapstore import MapStore
from ugv_api.state import StateStore
from ugv_api.watches import Timeouts


# The map thread is reported not alive when its 1 Hz timer has not ticked for this long (checked every
# MAP_HEALTH_PERIOD_S on the gateway node).
MAP_ALIVE_S = 3.0
MAP_HEALTH_PERIOD_S = 0.5
_LAST_REJECT_CHARS = 160


def _short(text: str) -> str:
    return text if len(text) <= _LAST_REJECT_CHARS else text[: _LAST_REJECT_CHARS - 3] + "..."


def _stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class GatewayNode(Node):
    def __init__(self, store: StateStore, goals: GoalRegistry, maps: MapStore | None = None, **node_kwargs) -> None:
        """`maps` is the store the 3D map layers are put into; without one the gateway has no map inputs.
        `node_kwargs` go to rclpy's Node (tests pass `parameter_overrides`)."""
        super().__init__("ugv_api", **node_kwargs)
        p = self.declare_parameter
        self.host = str(p("host", "127.0.0.1").value)
        self.port = int(p("port", 8080).value)
        self.telemetry_hz = float(p("telemetry_hz", 5.0).value)
        self.cors_origins = [str(o) for o in p("cors_origins", [""]).value if str(o)]
        self.timeouts = Timeouts(
            camera=float(p("timeouts.camera", 0.5).value),
            perception=float(p("timeouts.perception", 0.5).value),
            localization=float(p("timeouts.localization", 0.5).value),
            tf=float(p("timeouts.tf", 0.5).value),
            nav2=float(p("timeouts.nav2", 0.5).value),
        )
        self._map = str(p("map_frame", "map").value)
        self._base = str(p("base_frame", "base_link").value)
        self._rtabmap_ns = str(p("rtabmap_service_ns", "/rtabmap/rtabmap").value).rstrip("/")
        estop_hz = float(p("e_stop_republish_hz", 5.0).value)
        if not self.telemetry_hz > 0 or not estop_hz > 0:
            raise RuntimeError("telemetry_hz and e_stop_republish_hz must be > 0")
        cam_topic = str(p("camera_info_topic", "/camera/camera_info").value)
        self._maps = maps
        self._map_inputs: _MapInputs | None = None
        self._map_error: str | None = None  # why the map inputs are off, when they are
        self.map_cfg = self._declare_map_config()

        self._store = store
        self._goals = goals
        self._lock = threading.Lock()  # guards e-stop state and the goal-handle table
        self._estop_asserted = False
        self._handles: dict[str, object] = {}
        self.requested_mode: str | None = None

        best_effort = qos_profile_sensor_data  # matches reliable and best-effort publishers
        flags = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.create_subscription(CameraInfo, cam_topic, self._on_camera_info, best_effort)
        self.create_subscription(Image, "/segmentation/mask", self._stamped(k.MASK), best_effort)
        self.create_subscription(Bool, "/ugv/perception_degraded", self._value(k.PERCEPTION_DEGRADED), flags)
        self.create_subscription(Bool, "/ugv/pose_valid", self._value(k.POSE_VALID), flags)
        self.create_subscription(String, "/ugv/localization_status", self._value(k.LOCALIZATION_STATUS), 10)
        self.create_subscription(Bool, "/ugv/nav2_heartbeat", self._value(k.NAV2_HEARTBEAT), flags)
        self.create_subscription(String, "/ugv/nav2_status", self._value(k.NAV2_STATUS), latched)
        # Volatile + reliable matches both `ros2 topic pub` (volatile) and latched publishers.
        self.create_subscription(Bool, "/ugv/e_stop", self._value(k.E_STOP), 10)
        # The arbiter publishes it latched and only on change: a volatile reader misses the one sample.
        self.create_subscription(String, "/ugv/safety_status", self._value(k.SAFETY_STATUS), latched)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, flags)

        self._pub_estop = self.create_publisher(Bool, "/ugv/e_stop", latched)
        self.create_timer(1.0 / estop_hz, self._republish_estop)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.create_timer(0.1, self._poll_tf)

        self._nav = ActionClient(self, NavigateToPose, str(p("navigate_action", "/navigate_to_pose").value))
        self._mode_clients = {
            "mapping": self.create_client(Empty, f"{self._rtabmap_ns}/set_mode_mapping"),
            "localize": self.create_client(Empty, f"{self._rtabmap_ns}/set_mode_localization"),
        }
        self.get_logger().info(f"operator gateway on http://{self.host}:{self.port}/api/v1")
        if maps is not None:
            self._start_map_inputs()  # last: nothing after it can raise and leave its context and thread behind

    def destroy_node(self) -> None:
        map_inputs = getattr(self, "_map_inputs", None)
        if map_inputs is not None:
            map_inputs.close()
        super().destroy_node()

    # ---- inputs -------------------------------------------------------------------------------
    def now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def _value(self, key: str):
        def cb(msg) -> None:
            self._store.put(key, msg.data, self.now_ns())

        return cb

    def _stamped(self, key: str):
        def cb(msg) -> None:
            self._store.put(key, None, self.now_ns(), _stamp_ns(msg.header.stamp))

        return cb

    def _on_cmd_vel(self, msg: Twist) -> None:
        lin, ang = msg.linear, msg.angular
        value = ((lin.x, lin.y, lin.z), (ang.x, ang.y, ang.z))
        self._store.put(k.CMD_VEL, value, self.now_ns())

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self._store.put(k.CAMERA_INFO, None, self.now_ns(), _stamp_ns(msg.header.stamp))  # §12 camera watch

    def _poll_tf(self) -> None:
        try:
            t = self._tf_buffer.lookup_transform(self._map, self._base, Time())
        except TransformException:
            return  # absence shows up as a missing / stale TF watch
        tr, q = t.transform.translation, t.transform.rotation
        pose = (tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w)  # what GET /map/pose reports; the §12 watch reads the stamp
        self._store.put(k.TF_MAP_BASE, pose, self.now_ns(), _stamp_ns(t.header.stamp))

    # ---- 3D map inputs ------------------------------------------------------------------------
    def _declare_map_config(self) -> MapConfig:
        """Declare every `map.*` parameter (MapConfig holds the defaults and the types) and validate the lot.

        A map-only mistake must not take the gateway down with it: it carries the e-stop publisher and the §12
        table. The error is logged, the HTTP side gets the default numbers and the map inputs stay off
        (`map_inputs_alive: false`, the reason in `map_last_reject`)."""
        try:
            values = {}
            for f in dataclasses.fields(MapConfig):
                values[f.name] = type(f.default)(self.declare_parameter(f"map.{f.name}", f.default).value)
            return MapConfig(**values)
        except Exception as exc:  # noqa: BLE001 - see the docstring
            self._map_error = f"map inputs are off, invalid map configuration: {type(exc).__name__}: {exc}"
            self.get_logger().error(self._map_error)
            return MapConfig()

    def _start_map_inputs(self) -> None:
        # the timer first: if it cannot be created nothing is running yet that would have to be undone
        self.create_timer(MAP_HEALTH_PERIOD_S, self._map_health_tick)
        if self._map_error is None:
            try:
                self._map_inputs = _MapInputs(self, self._maps, self.map_cfg, self._tf_buffer, self._map)
            except Exception as exc:  # noqa: BLE001 - _MapInputs has already taken down what it started
                self._map_error = f"map inputs are off, they failed to start: {type(exc).__name__}: {exc}"
                self.get_logger().error(self._map_error)
        self._map_health_tick()

    def _map_health_tick(self) -> None:
        """The map health timer runs on the gateway's executor, next to the e-stop republish and the §12 watches: an
        exception escaping a callback there would end that executor's spin and take them down with it, so a failure
        is logged (throttled) and never raised, like the map side's `_guarded` callbacks."""
        try:
            self._publish_map_health()
        except Exception as exc:  # noqa: BLE001 - see the docstring
            self.get_logger().error(
                f"map health update failed: {type(exc).__name__}: {exc}", throttle_duration_sec=10.0
            )

    def _publish_map_health(self) -> None:
        """The gateway's side of the map inputs' health, on the gateway's executor (see _MapInputs.publish_health);
        with the inputs off it says so and why."""
        if self._map_inputs is not None:
            self._map_inputs.publish_health()
            return
        self._maps.put_stats("map_inputs", {"map_inputs_alive": False})
        self._maps.put_stats("gateway", {"map_rejects": 0, "map_restarts": 0,
                                         "map_last_reject": _short(self._map_error or "no map inputs")})

    @property
    def map_rejects(self) -> dict[str, int]:
        """How many messages (or frames) each map input has refused or skipped, by input name."""
        return self._map_inputs.reject_counts() if self._map_inputs is not None else {}

    # ---- e-stop (§3.1 level 1) ----------------------------------------------------------------
    @property
    def estop_asserted(self) -> bool:
        with self._lock:
            return self._estop_asserted

    def set_estop(self, asserted: bool) -> None:
        """Assert: publish true now and keep re-publishing. Release: publish false once."""
        with self._lock:
            self._estop_asserted = asserted
        self._pub_estop.publish(Bool(data=asserted))
        self.get_logger().warning(f"operator e-stop {'ASSERTED' if asserted else 'released'}")

    def _republish_estop(self) -> None:
        if self.estop_asserted:
            self._pub_estop.publish(Bool(data=True))

    # ---- localization mode (§10) --------------------------------------------------------------
    def set_mode(self, mode: str, timeout_s: float = 3.0) -> None:
        client = self._mode_clients[mode]
        if not client.wait_for_service(timeout_sec=0.5):
            raise ServiceUnavailable(f"{client.srv_name} is not available (is RTAB-Map running?)")
        done = threading.Event()
        future = client.call_async(Empty.Request())
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout_s):
            future.cancel()
            raise ServiceUnavailable(f"{client.srv_name} did not answer within {timeout_s:.1f} s")
        if future.exception() is not None:
            raise ServiceUnavailable(f"{client.srv_name} failed: {future.exception()}")
        self.requested_mode = mode

    # ---- navigation goals (§10, §11) ----------------------------------------------------------
    def action_ready(self) -> bool:
        return self._nav.server_is_ready()

    def send_goal(self, rec: GoalRecord) -> None:
        if not self._nav.wait_for_server(timeout_sec=0.5):
            raise ServiceUnavailable("/navigate_to_pose action server is not available (is Nav2 running?)")
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = rec.frame_id
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = rec.x
        goal.pose.pose.position.y = rec.y
        q = goal.pose.pose.orientation
        q.x, q.y, q.z, q.w = yaw_to_quaternion(rec.yaw)

        def on_feedback(fb) -> None:
            f = fb.feedback
            fields = {"distance_remaining": float(f.distance_remaining), "recoveries": int(f.number_of_recoveries)}
            cur = self._goals.get(rec.id)
            if cur is not None and cur.state == GoalState.PENDING:  # never undo a CANCELING
                fields["state"] = GoalState.EXECUTING
            self._goals.update(rec.id, **fields)

        future = self._nav.send_goal_async(goal, feedback_callback=on_feedback)
        future.add_done_callback(lambda f: self._on_goal_response(rec.id, f))

    def _on_goal_response(self, goal_id: str, future) -> None:
        if future.exception() is not None:
            self._goals.update(goal_id, state=GoalState.FAILED, error_message=str(future.exception()))
            return
        handle = future.result()
        if not handle.accepted:
            self._goals.update(goal_id, state=GoalState.REJECTED, error_message="rejected by Nav2")
            return
        with self._lock:
            self._handles[goal_id] = handle
        self._goals.preempt_active(except_id=goal_id)
        self._goals.update(goal_id, state=GoalState.EXECUTING)
        handle.get_result_async().add_done_callback(lambda f: self._on_result(goal_id, f))

    def _on_result(self, goal_id: str, future) -> None:
        with self._lock:
            self._handles.pop(goal_id, None)
        if future.exception() is not None:
            self._goals.update(goal_id, state=GoalState.FAILED, error_message=str(future.exception()))
            return
        res = future.result()
        result = res.result
        code = int(getattr(result, "error_code", 0))
        msg = str(getattr(result, "error_msg", "")) or None
        state = state_from_status(res.status)
        if res.status == GoalStatus.STATUS_SUCCEEDED:
            code, msg = None, None
        self._goals.update(goal_id, state=state, error_code=code, error_message=msg)

    def cancel_goal(self, goal_id: str) -> bool:
        """Request cancel. False if the goal has no live Nav2 handle (unknown, pending or finished)."""
        with self._lock:
            handle = self._handles.get(goal_id)
        if handle is None:
            return False
        self._goals.update(goal_id, state=GoalState.CANCELING)
        handle.cancel_goal_async()
        return True


class _MapInputs:
    """The inputs of the 3D map viewer, kept away from the §12 watch inputs.

    Isolation. The map inputs run on a node of their own, `ugv_api_map`, in a second rclpy context (a second DDS
    participant) spun by a single-threaded executor on a thread of its own; every subscription and the 1 Hz
    demand timer share one mutually exclusive callback group on it. Callback groups and executor threads alone
    were measured not to be enough: with only a separate group on the gateway node, a 1 M point (32 MB) cloud
    at 2 Hz made the tf watch stale for about 1 to 3 s in 6 of 10 runs of 30 s (worst age 3.07 s), although no
    callback ran long. The delay sits below rclpy, in the participant that receives the big sample: reliable
    topics of the same participant (TF) wait behind it, a probe process on its own participant saw none of it.
    With the map inputs on their own participant 13 of 13 runs had a worst tf age of 0.17 to 0.39 s. The node
    shares nothing with the gateway node but the TF buffer (thread-safe), the stores and the clock reading.

    The second context is initialised with `args=[]` and its node created with `use_global_arguments=False`:
    without that it falls back to `sys.argv`, which under `ros2 launch` holds `-r __node:=ugv_api` (the map node
    would be a second /ugv_api) and the gateway's parameters (`use_sim_time`). Its timer runs on the system clock.

    The executor must stay single-threaded: callbacks (including the timer that destroys subscriptions) then
    run on the thread that builds the wait set. With a MultiThreadedExecutor a `destroy_subscription` from a
    worker thread can land between rclpy marking a subscription's QoS event handler in use and adding it to
    the wait set; the second entry raises InvalidHandle out of `spin` (reproduced within seconds with a
    toggling demand) and ends the thread. One group and one thread also mean these callbacks never run
    concurrently, so `_subs` needs no lock.

    Health. A map thread that hangs or dies cannot report that itself, so the gateway does it
    (`publish_health`, called by a timer of the gateway node): `map_inputs_alive` is false when the thread is
    gone or its demand timer has not ticked for MAP_ALIVE_S, and stale statistics are expired there too. What
    the map thread can report it does through the `gateway` statistics source: `map_rejects` (messages and
    frames refused or skipped), `map_restarts` (spin restarts by the supervisor) and `map_last_reject`.

    A callback does the minimum: copy out what the layer's source needs (ugv_api.mapsources, pure functions)
    and `MapStore.put` it. Encoding happens later, on the HTTP thread, only for a layer somebody requests.

    Demand: the heavy subscriptions exist only while `maps.wanted(now, idle_timeout_s)`, which GET /api/v1/map
    keeps true. The timer creates them when it becomes true and destroys them when it stops. The map stats
    subscription and the TF lookups are always on.

    Durability: a TRANSIENT_LOCAL subscription only matches a latched publisher, a VOLATILE one matches both
    but misses the latched sample. So each subscription takes the durability its publishers offer (latched only
    when all of them are), VOLATILE when there is none yet, and the timer re-creates a subscription whose
    publisher turned out to offer something else. Losing the publisher does not change anything: the
    subscription is kept for when it comes back.

    Frames: trajectory and grid are served as map-frame layers, so a message whose
    `frame_id` is neither the map frame nor empty (a local costmap is in `odom`) is refused and counted.
    """

    def __init__(self, gateway: GatewayNode, maps: MapStore, cfg: MapConfig, tf_buffer: Buffer,
                 map_frame: str) -> None:
        self._gw = gateway  # logger, clock (sim-time aware, the one the HTTP layer touches with)
        self._maps = maps
        self._cfg = cfg
        self._tf = tf_buffer
        self._frame = map_frame.lstrip("/")
        self._closing = threading.Event()
        # guards the counters and the stats bookkeeping below; held for dict updates and put_stats only, never
        # across a ROS call, so the gateway-side health check can always take it
        self._state_lock = threading.Lock()
        self.rejects: dict[str, int] = {}
        self.restarts = 0
        self._last_reject: str | None = None
        self._gateway_stats: dict[str, int] = {}
        self._stats_seen: dict[str, float] = {}  # source -> time.monotonic() of its last message
        self._last_tick = time.monotonic()
        self._subs: dict[str, tuple[Any, DurabilityPolicy]] = {}
        self._node: Node | None = None
        self._executor: SingleThreadedExecutor | None = None
        self._thread: threading.Thread | None = None
        self._closed = False
        self._context = Context()
        try:
            rclpy.init(context=self._context, args=[], domain_id=gateway.context.get_domain_id(),
                       signal_handler_options=SignalHandlerOptions.NO)
            self._node = rclpy.create_node("ugv_api_map", context=self._context, enable_rosout=False,
                                           start_parameter_services=False, use_global_arguments=False)
            self._executor = SingleThreadedExecutor(context=self._context)
            self._executor.add_node(self._node)
            self._group = MutuallyExclusiveCallbackGroup()

            c = cfg
            guard = self._guarded
            self._specs: dict[str, tuple[str, type, Callable[[Any], None]]] = {
                "trajectory": (c.trajectory_topic, Path, guard("trajectory", self._on_path)),
                "grid": (c.grid_topic, OccupancyGrid, guard("grid", self._on_grid)),
                "live": (c.live_cloud_topic, PointCloud2, guard("live", self._on_live)),
            }
            # reliable + volatile matches a latched publisher (/ugv/map/stats) and a plain one alike
            stats_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
            self._node.create_subscription(String, c.map_stats_topic, guard("map_stats", self._stats_callback("map")),
                                           stats_qos, callback_group=self._group)
            self._node.create_timer(1.0, guard("demand_timer", self._tick), callback_group=self._group)
            with self._state_lock:
                self._publish_gateway_locked()  # the health keys exist from the first moment
            self._thread = threading.Thread(target=self._spin, name="ugv_api_map_ros", daemon=True)
            self._last_tick = time.monotonic()
            self._thread.start()
        except BaseException:
            self._teardown()  # a context left running would hold a DDS participant until the process ends
            raise

    # ---- thread and lifetime ------------------------------------------------------------------
    def _spin(self) -> None:
        try:
            self._spin_supervised()
        except BaseException as exc:  # noqa: BLE001 - nothing may end this thread without a trace
            self._thread_ended(f"{type(exc).__name__}: {exc}")
        else:
            if not self._closing.is_set():
                self._thread_ended("the map executor stopped")

    def _thread_ended(self, why: str) -> None:
        self._reject("map_thread_exit", why)
        self._gw.get_logger().error(f"map input thread ended ({why}): the map layers stop updating")

    def _spin_supervised(self) -> None:
        while not self._closing.is_set():
            try:
                self._executor.spin()
                return  # shut down
            except (KeyboardInterrupt, ExternalShutdownException):
                return
            except Exception as exc:  # noqa: BLE001 - the map inputs are not worth a dead thread, and not a dead gateway
                with self._state_lock:
                    self.restarts += 1
                self._reject("map_executor", f"{type(exc).__name__}: {exc} (spin restarted)")
                self._closing.wait(1.0)

    def close(self) -> None:
        """Stop the map executor and take the second participant down (GatewayNode.destroy_node)."""
        self._closing.set()
        self._teardown()

    def _teardown(self) -> None:
        if self._closed:
            return
        self._closed = True
        steps = [
            lambda: self._executor.shutdown() if self._executor is not None else None,
            lambda: self._thread.join(timeout=2.0) if self._thread is not None and self._thread.is_alive() else None,
            lambda: self._node.destroy_node() if self._node is not None else None,
            self._context.try_shutdown,
        ]
        for step in steps:
            try:
                step()
            except Exception as exc:  # noqa: BLE001 - a later step must still run
                self._gw.get_logger().warning(f"map inputs teardown: {type(exc).__name__}: {exc}")

    # ---- health: evaluated by the gateway, not by the thread it is about ------------------------
    def publish_health(self) -> None:
        """Called by a timer of the gateway node. A hung or dead map thread cannot say so itself, and cannot
        expire the statistics it holds either, so both are decided here, on the gateway's executor."""
        now = time.monotonic()
        alive = self._thread is not None and self._thread.is_alive() and now - self._last_tick <= MAP_ALIVE_S
        self._maps.put_stats("map_inputs", {"map_inputs_alive": alive})
        self._expire_stats(now)

    def reject_counts(self) -> dict[str, int]:
        with self._state_lock:
            return dict(self.rejects)

    # ---- bookkeeping --------------------------------------------------------------------------
    def _publish_gateway_locked(self) -> None:
        """Put the gateway's own statistics. The caller holds `_state_lock`."""
        values: dict[str, Any] = dict(self._gateway_stats)
        values["map_rejects"] = sum(self.rejects.values())
        values["map_restarts"] = self.restarts
        values["map_last_reject"] = self._last_reject
        self._maps.put_stats("gateway", values)

    def _reject(self, name: str, why: str, n: int = 1) -> None:
        """Count a refused message or skipped frame and remember it as the last one; log the first one of each
        kind only. All of it is in the `gateway` statistics, which is how an operator sees it."""
        with self._state_lock:
            total = self.rejects.get(name, 0) + n
            self.rejects[name] = total
            self._last_reject = _short(f"{name}: {why}")
            self._publish_gateway_locked()
        if total == n:
            self._gw.get_logger().warning(f"map input '{name}': {why} (later ones are counted, not logged)")

    def _guarded(self, name: str, fn: Callable[..., None]) -> Callable[..., None]:
        """A bad message is counted and logged, never raised: an exception escaping a callback would end the
        executor's spin and take the §12 watches down with it."""

        def run(*args: Any) -> None:
            try:
                fn(*args)
            except Exception as exc:  # noqa: BLE001 - see the docstring
                self._reject(name, f"{type(exc).__name__}: {exc}")

        return run

    def _stamp_s(self, msg: Any) -> float:
        return ms.stamp_seconds(msg.header.stamp.sec, msg.header.stamp.nanosec, fallback_s=self._gw.now_ns() / 1e9)

    def _in_map_frame(self, name: str, msg: Any) -> bool:
        """False (and counted) for a message in some other frame; an empty frame_id is taken as the map frame."""
        frame = msg.header.frame_id.lstrip("/")
        if frame in ("", self._frame):
            return True
        self._reject(f"frame_{name}", f"{name} is in frame '{msg.header.frame_id}', not '{self._frame}', not shown")
        return False

    # ---- demand timer -------------------------------------------------------------------------
    def _tick(self) -> None:
        self._last_tick = time.monotonic()  # what the gateway's liveness check reads
        if self._maps.wanted(self._gw.now_ns() / 1e9, self._cfg.idle_timeout_s):
            self._ensure_subscriptions()
        else:
            self._drop_subscriptions()

    def _offered_durability(self, topic: str) -> DurabilityPolicy | None:
        """What the topic's publishers offer: None without a publisher, TRANSIENT_LOCAL when all of them latch."""
        infos = self._node.get_publishers_info_by_topic(topic)
        if not infos:
            return None
        if all(i.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL for i in infos):
            return DurabilityPolicy.TRANSIENT_LOCAL
        return DurabilityPolicy.VOLATILE

    def _ensure_subscriptions(self) -> None:
        for key, (topic, msg_type, callback) in self._specs.items():
            try:
                offered = self._offered_durability(topic)
                current = self._subs.get(key)
                if current is not None and (offered is None or offered == current[1]):
                    continue
                if current is not None:  # the publisher offers something else than we asked for
                    self._node.destroy_subscription(current[0])
                    del self._subs[key]
                durability = offered or DurabilityPolicy.VOLATILE
                qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=durability)
                sub = self._node.create_subscription(msg_type, topic, callback, qos, callback_group=self._group)
                self._subs[key] = (sub, durability)
            except Exception as exc:  # noqa: BLE001 - one bad topic must not stop the others
                self._reject(f"subscribe_{key}", f"{type(exc).__name__}: {exc}")

    def _drop_subscriptions(self) -> None:
        for sub, _durability in self._subs.values():
            self._node.destroy_subscription(sub)
        self._subs.clear()

    def _expire_stats(self, now: float) -> None:
        """Drop a stats source that has been silent for `stats_stale_s` (monotonic seconds): it must not keep
        showing old numbers."""
        with self._state_lock:
            for source, seen in list(self._stats_seen.items()):
                if now - seen > self._cfg.stats_stale_s:
                    self._maps.put_stats(source, {})
                    del self._stats_seen[source]

    # ---- statistics ---------------------------------------------------------------------------
    def _stats_callback(self, source: str) -> Callable[[String], None]:
        def on_stats(msg: String) -> None:
            values = ms.parse_stats(msg.data)
            if values is None:
                self._reject(f"{source}_stats", "not a JSON object, ignored")
                return
            with self._state_lock:
                self._maps.put_stats(source, values)
                self._stats_seen[source] = time.monotonic()

        return on_stats

    # ---- layers -------------------------------------------------------------------------------
    @staticmethod
    def _cloud_args(msg: PointCloud2) -> dict[str, Any]:
        return {
            "fields": [(f.name, f.offset, f.datatype, f.count) for f in msg.fields],
            "point_step": msg.point_step,
            "n_points": msg.width * msg.height,
            "is_bigendian": msg.is_bigendian,
            "data": msg.data,
            "width": msg.width,
            "height": msg.height,
            "row_step": msg.row_step,
        }

    def _on_path(self, msg: Path) -> None:
        if not self._in_map_frame("trajectory", msg):
            return
        rows = [(p.pose.position.x, p.pose.position.y, p.pose.position.z, p.pose.orientation.x,
                 p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w) for p in msg.poses]
        self._maps.put("trajectory", ms.trajectory_source(rows), self._stamp_s(msg))

    def _on_grid(self, msg: OccupancyGrid) -> None:
        if not self._in_map_frame("grid", msg):
            return
        info, o = msg.info, msg.info.origin
        source = ms.grid_source(
            data=msg.data, width=info.width, height=info.height, resolution=info.resolution,
            origin_x=o.position.x, origin_y=o.position.y,
            origin_q=(o.orientation.x, o.orientation.y, o.orientation.z, o.orientation.w))
        self._maps.put("grid", source, self._stamp_s(msg))

    def _on_live(self, msg: PointCloud2) -> None:
        """Dev 1's depth cloud (camera optical frame, back-projected by perception) range-gated and moved into the
        map frame. Without any TF map <- camera the frame is skipped: camera-frame points are never published as
        map-frame points."""
        transform = self._camera_to_map(msg.header.frame_id, msg.header.stamp)
        if transform is None:
            self._reject("live_no_transform", f"no TF {self._frame} <- '{msg.header.frame_id}', live layer skipped")
            return
        xyz, _ = codec.cloud_view(**ms.cloud_source(**self._cloud_args(msg)))  # a view of msg.data, no copy
        points = ms.within_range(xyz, self._cfg.live_range_min_m, self._cfg.live_range_max_m)
        self._maps.put("live", {"xyz": ms.transform_points(points, *transform)}, self._stamp_s(msg))

    def _camera_to_map(self, frame_id: str, stamp: Any) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
        """(translation, quaternion) of `frame_id` in the map frame at the image stamp, else the latest one."""
        for when in (Time.from_msg(stamp), Time()):
            try:
                t = self._tf.lookup_transform(self._frame, frame_id, when)
            except TransformException:
                continue
            tr, q = t.transform.translation, t.transform.rotation
            return (tr.x, tr.y, tr.z), (q.x, q.y, q.z, q.w)
        return None
