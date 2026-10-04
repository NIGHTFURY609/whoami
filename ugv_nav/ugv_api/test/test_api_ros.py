"""Gateway on a real ROS graph over real HTTP (uvicorn on a free port, httpx client).

Real rclpy nodes publish the contract topics; no Nav2 or RTAB-Map is running, so goals and mode
switches must answer 503 problems, and a tripped §12 watch must answer 409. Runs under colcon test
(or any shell with ROS sourced); skipped in plain Python.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import socket
import sys
import threading
import time

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
httpx = pytest.importorskip("httpx")
uvicorn = pytest.importorskip("uvicorn")

from geometry_msgs.msg import PoseStamped, TransformStamped, Twist  # noqa: E402
from nav_msgs.msg import OccupancyGrid, Path  # noqa: E402
from rclpy.context import Context  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile  # noqa: E402
from rclpy.time import Time  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField  # noqa: E402
from std_msgs.msg import Bool, String  # noqa: E402
from tf2_ros import TransformBroadcaster  # noqa: E402

import mapread  # noqa: E402
from ugv_api import mapcodec as codec  # noqa: E402
from ugv_api import ros_node  # noqa: E402
from ugv_api.app import create_app  # noqa: E402
from ugv_api.goals import GoalRegistry  # noqa: E402
from ugv_api.mapsources import MapConfig  # noqa: E402
from ugv_api.mapstore import MapStore  # noqa: E402
from ugv_api.ros_node import GatewayNode  # noqa: E402
from ugv_api.state import StateStore  # noqa: E402

PROBLEM = "application/problem+json"

# Short on purpose: the map tests wait for the demand timer (1 Hz) and for these limits to pass.
IDLE_S = 1.5
STATS_STALE_S = 1.5

CLOUD, PATH = "/rtabmap/cloud_map", "/rtabmap/mapPath"
GRID, DEPTH_CLOUD = "/global_costmap/costmap", "/perception/depth_cloud"
MAP_STATS = "/ugv/map/stats"
HEAVY_TOPICS = (PATH, GRID, DEPTH_CLOUD)
GATEWAY_STAT_KEYS = {"cloud_source_points", "map_inputs_alive", "map_rejects", "map_restarts", "map_last_reject"}
# A map thread that stopped ticking is reported not alive after MAP_ALIVE_S; the gateway checks every 0.5 s.
MAP_ALIVE_BOUND = ros_node.MAP_ALIVE_S + 3.0

# What Inputs broadcasts: map -> base_link at POSE, base_link -> camera_optical_frame at CAMERA_MOUNT, and a
# CameraInfo for an 8 x 6 camera.
POSE = {"x": 1.5, "y": -0.5, "z": 0.25, "yaw": 0.4}
CAMERA_MOUNT = (0.1, 0.0, 0.3)  # metres in base_link
OPTICAL_TO_BODY_Q = (-0.5, 0.5, -0.5, 0.5)  # optical (x right, y down, z forward) -> body (x forward, y left, z up)
K = (10.0, 0.0, 4.0, 0.0, 10.0, 2.0, 0.0, 0.0, 1.0)
CAM_W, CAM_H = 8, 6


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Inputs:
    """Real publishers for the §12 inputs, driven at 20 Hz by a timer on `node`."""

    def __init__(self, node) -> None:
        self.node = node
        self.healthy = True
        self.cam = node.create_publisher(CameraInfo, "/camera/camera_info", 10)
        self.mask = node.create_publisher(Image, "/segmentation/mask", 10)
        self.degraded = node.create_publisher(Bool, "/ugv/perception_degraded", 10)
        self.pose = node.create_publisher(Bool, "/ugv/pose_valid", 10)
        self.loc_status = node.create_publisher(String, "/ugv/localization_status", 10)
        self.hb = node.create_publisher(Bool, "/ugv/nav2_heartbeat", 10)
        self.cmd = node.create_publisher(Twist, "/cmd_vel", 10)
        self.tf = TransformBroadcaster(node)
        node.create_timer(0.05, self.tick)

    def tick(self) -> None:
        if not self.healthy:
            return
        now = self.node.get_clock().now().to_msg()
        info = CameraInfo()
        info.header.stamp, info.header.frame_id = now, "camera_optical_frame"
        info.width, info.height, info.k = CAM_W, CAM_H, list(K)
        self.cam.publish(info)
        mask = Image()
        mask.header.stamp, mask.header.frame_id, mask.encoding = now, "camera_optical_frame", "mono8"
        self.mask.publish(mask)
        self.degraded.publish(Bool(data=False))
        self.pose.publish(Bool(data=True))
        self.hb.publish(Bool(data=True))
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = now, "map", "base_link"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (
            POSE["x"], POSE["y"], POSE["z"])
        t.transform.rotation.z, t.transform.rotation.w = math.sin(POSE["yaw"] / 2), math.cos(POSE["yaw"] / 2)
        mount = TransformStamped()
        mount.header.stamp, mount.header.frame_id, mount.child_frame_id = now, "base_link", "camera_optical_frame"
        mount.transform.translation.x, mount.transform.translation.y, mount.transform.translation.z = CAMERA_MOUNT
        q = mount.transform.rotation
        q.x, q.y, q.z, q.w = OPTICAL_TO_BODY_Q
        self.tf.sendTransform([t, mount])


class MapPubs:
    """Publishers for the map inputs, with the durability each real publisher uses: the map stats are latched
    (transient local); the path, the depth cloud and the perception stats are volatile."""

    def __init__(self, node) -> None:
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        volatile = QoSProfile(depth=1)
        self.node = node
        self.path = node.create_publisher(Path, PATH, volatile)
        self.depth_cloud = node.create_publisher(PointCloud2, DEPTH_CLOUD, volatile)
        self.map_stats = node.create_publisher(String, MAP_STATS, latched)

    def now(self):
        return self.node.get_clock().now().to_msg()


@pytest.fixture(scope="module")
def graph():
    os.environ.setdefault("ROS_DOMAIN_ID", "57")
    rclpy.init()
    store, goals, maps = StateStore(), GoalRegistry(), MapStore()
    gw = GatewayNode(store, goals, maps, parameter_overrides=[
        Parameter("map.idle_timeout_s", Parameter.Type.DOUBLE, IDLE_S),
        Parameter("map.stats_stale_s", Parameter.Type.DOUBLE, STATS_STALE_S),
    ])
    pub_node = rclpy.create_node("ugv_api_test_inputs")
    inputs = Inputs(pub_node)
    pubs = MapPubs(pub_node)
    seen: list[bool] = []
    pub_node.create_subscription(Bool, "/ugv/e_stop", lambda m: seen.append(bool(m.data)), 10)
    ex = MultiThreadedExecutor()
    ex.add_node(gw)
    ex.add_node(pub_node)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()

    port = _free_port()
    app = create_app(gw, store, goals, telemetry_hz=10.0, maps=maps, **gw.map_cfg.app_kwargs())
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    serve = threading.Thread(target=server.run, daemon=True)
    serve.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn did not start"
    client = httpx.Client(base_url=f"http://127.0.0.1:{port}/api/v1", timeout=10.0)
    yield {"client": client, "gw": gw, "inputs": inputs, "pub_node": pub_node, "seen": seen, "pubs": pubs,
           "maps": maps}
    client.close()
    server.should_exit = True
    serve.join(timeout=5)
    ex.shutdown()
    spin.join(timeout=5)  # nothing may still be running on the nodes when they are destroyed
    gw.destroy_node()
    pub_node.destroy_node()
    rclpy.shutdown()


def _wait(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        v = pred()
        if v:
            return v
        time.sleep(0.05)
    return pred()


@contextlib.contextmanager
def own_participant(name):
    """A node in a context of its own: a DDS participant the gateway has not met yet. Not spun, so it can be
    destroyed again (destroying an entity under a spinning executor can raise InvalidHandle out of rclpy)."""
    context = Context()
    rclpy.init(context=context)
    node = rclpy.create_node(name, context=context)
    try:
        yield node
    finally:
        node.destroy_node()
        context.try_shutdown()


@contextlib.contextmanager
def bare_gateway(*overrides, args=(), spin=True):
    """A gateway of its own, in a context of its own, with its own store and no HTTP server: yields (gw, maps).
    `args` are the gateway's ROS arguments (what `ros2 launch` puts after --ros-args)."""
    os.environ.setdefault("ROS_DOMAIN_ID", "57")
    # Like `ros2 launch` / `ros2 run`: main() calls rclpy.init() without arguments, so they come from sys.argv, and
    # every later rclpy.init() without arguments in the process (the map side's) reads the same ones.
    saved_argv = sys.argv
    sys.argv = ["api_gateway", *args]
    context = Context()
    maps = MapStore()
    gw = executor = spinner = None
    try:
        rclpy.init(context=context)
        gw = GatewayNode(StateStore(), GoalRegistry(), maps, context=context, parameter_overrides=list(overrides))
        sys.argv = saved_argv
        if spin:
            executor = MultiThreadedExecutor(context=context)
            executor.add_node(gw)
            spinner = threading.Thread(target=executor.spin, daemon=True)
            spinner.start()
        yield gw, maps
    finally:
        if executor is not None:
            executor.shutdown()
            spinner.join(5.0)  # nothing may still be running on the node when it is destroyed
        if gw is not None:
            gw.destroy_node()
        context.try_shutdown()
        sys.argv = saved_argv


def _has_estop_publisher(gw):
    # not asserted: a latched True from a throwaway gateway would trip the e-stop watch of the module's gateway
    def listed():  # the graph learns of an entity a moment after it is created
        return "/ugv/e_stop" in {name for name, _ in gw.get_publisher_names_and_types_by_node(gw.get_name(), "/")}

    return bool(_wait(listed, timeout=5.0))


class _MapThreadGone(BaseException):
    """Not an Exception: nothing on the map side is allowed to swallow it."""


def _on_map_thread(gw, fn):
    """Run `fn` once on the gateway's map thread, in the map callback group: while it runs, that group's demand
    timer (the thread's liveness tick) cannot."""
    inputs = gw._map_inputs

    def once():
        timer.cancel()
        fn()

    timer = inputs._node.create_timer(0.05, once, callback_group=inputs._group)


def _safety(c):
    r = c.get("/safety/status")
    assert r.status_code == 200
    return r.json()


def test_health_and_openapi(graph):
    c = graph["client"]
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["rosNode"] == "ugv_api"
    spec = c.get("/openapi.json").json()
    assert "/api/v1/navigation/goals" in spec["paths"] and "/api/v1/safety/e-stop" in spec["paths"]


def test_all_watches_ok_with_live_inputs_and_arbiter_absent(graph):
    c = graph["client"]
    body = _wait(lambda: (lambda b: b if b["ok"] else None)(_safety(c)))
    assert body, _safety(c)
    assert [w["name"] for w in body["watches"]] == ["camera", "perception", "localization", "tf", "nav2", "e_stop"]
    assert body["arbiter"] == {"status": None, "ageS": None, "present": False}  # no Dev 5 arbiter yet


def test_heartbeat_that_stops_trips_watches(graph):
    c, inputs = graph["client"], graph["inputs"]
    inputs.healthy = False
    expected = {"camera", "perception", "localization", "tf", "nav2"}

    def tripped_names(body):
        return {w["name"] for w in body["watches"] if not w["ok"]}

    try:
        # Wait for all five, not for the first: the stamp-based watches (camera, perception, tf) trip a few
        # milliseconds before the receipt-based ones (localization, nav2), and a poll can land in between.
        tripped = _wait(lambda: (lambda b: b if expected <= tripped_names(b) else None)(_safety(c)), timeout=3.0)
        assert tripped, f"watches stayed ok after inputs stopped: tripped {tripped_names(_safety(c))}"
        assert tripped["ok"] is False
    finally:
        inputs.healthy = True
    assert _wait(lambda: _safety(c)["ok"])


def test_base_command_reads_final_cmd_vel(graph):
    c, inputs = graph["client"], graph["inputs"]
    assert c.get("/base/command").json()["available"] is False
    tw = Twist()
    tw.linear.x, tw.angular.z = 0.25, -0.5
    body = _wait(lambda: (inputs.cmd.publish(tw), (lambda b: b if b["available"] else None)(
        c.get("/base/command").json()))[1])
    assert body["linear"]["x"] == pytest.approx(0.25) and body["angular"]["z"] == pytest.approx(-0.5)


def test_estop_assert_republish_latch_and_release(graph):
    c, seen, pub_node = graph["client"], graph["seen"], graph["pub_node"]
    seen.clear()
    r = c.put("/safety/e-stop", json={"asserted": True})
    assert r.status_code == 200 and r.json()["asserted"] is True and r.json()["assertedByGateway"] is True
    assert _wait(lambda: seen.count(True) >= 3, timeout=3.0), f"e-stop not re-published: {seen}"

    late: list[bool] = []
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    sub = pub_node.create_subscription(Bool, "/ugv/e_stop", lambda m: late.append(bool(m.data)), latched)
    assert _wait(lambda: late and late[0] is True), "late subscriber did not get the latched e-stop"
    pub_node.destroy_subscription(sub)

    e = next(w for w in _safety(c)["watches"] if w["name"] == "e_stop")
    assert e["ok"] is False

    goal = c.post("/navigation/goals", json={"x": 1.0, "y": 0.0})
    assert goal.status_code == 409 and goal.headers["content-type"].startswith(PROBLEM)
    assert any(r.startswith("e_stop") for r in goal.json()["reasons"])

    seen.clear()
    r = c.put("/safety/e-stop", json={"asserted": False})
    assert r.status_code == 200 and r.json()["assertedByGateway"] is False
    assert _wait(lambda: False in seen)
    time.sleep(0.5)
    assert True not in seen[seen.index(False):], "e-stop re-published after release"
    assert _wait(lambda: _safety(c)["ok"])


def test_goal_without_nav2_is_503_problem(graph):
    c = graph["client"]
    assert _wait(lambda: _safety(c)["ok"])
    r = c.post("/navigation/goals", json={"x": 3.0, "y": 0.0, "yaw": 0.0, "frameId": "map"})
    assert r.status_code == 503 and r.headers["content-type"].startswith(PROBLEM)
    assert "navigate_to_pose" in r.json()["detail"]
    nav = c.get("/navigation").json()
    assert nav["actionServerReady"] is False and nav["heartbeat"] is True and nav["activeGoal"] is None


def test_mode_without_rtabmap_is_503_problem(graph):
    r = graph["client"].put("/localization/mode", json={"mode": "localize"})
    assert r.status_code == 503 and r.headers["content-type"].startswith(PROBLEM)
    assert graph["client"].get("/localization").json()["requestedMode"] is None


@pytest.mark.parametrize(
    "path,method,body",
    [
        ("/navigation/goals", "post", {"x": "north", "y": 0}),
        ("/navigation/goals", "post", {"x": 1, "y": 0, "frameId": "odom"}),  # map-frame goals only (§11)
        ("/navigation/goals", "post", {"x": 1, "y": 0, "speed": 9}),  # unknown fields rejected
        ("/localization/mode", "put", {"mode": "slam"}),
        ("/safety/e-stop", "put", {}),
    ],
)
def test_invalid_bodies_are_422_problems(graph, path, method, body):
    r = getattr(graph["client"], method)(path, json=body)
    assert r.status_code == 422 and r.headers["content-type"].startswith(PROBLEM)
    assert r.json()["errors"]


def test_unknown_goal_is_404_problem(graph):
    c = graph["client"]
    for r in (c.get("/navigation/goals/nope"), c.delete("/navigation/goals/nope")):
        assert r.status_code == 404 and r.headers["content-type"].startswith(PROBLEM)


def test_sse_stream_emits_each_resource(graph):
    c = graph["client"]
    got: dict[str, dict] = {}
    with c.stream("GET", "/telemetry/stream") as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        event = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event:
                got[event] = json.loads(line[6:])
            if len(got) == 6:
                break
    assert set(got) == {"safety", "command", "localization", "navigation", "map", "pose"}
    assert "watches" in got["safety"] and "heartbeat" in got["navigation"]


def test_gateway_never_publishes_cmd_vel(graph):
    gw = graph["gw"]
    topics = {name for name, _ in gw.get_publisher_names_and_types_by_node("ugv_api", "/")}
    assert "/ugv/e_stop" in topics
    assert not topics & {"/cmd_vel", "/cmd_vel_nav2"}, topics


# ------------------------------------------------------------------------------------------- map inputs
# The heavy subscriptions exist only while a client keeps calling GET /map; each test that needs them holds the
# heartbeat with `watching`. Volatile topics are published repeatedly until the HTTP side shows them.


@contextlib.contextmanager
def watching(client):
    """Call GET /map from a thread every 0.2 s: the map view's demand heartbeat."""
    stop = threading.Event()

    def beat() -> None:
        with httpx.Client(base_url=str(client.base_url), timeout=5.0) as own:
            while not stop.is_set():
                with contextlib.suppress(httpx.HTTPError):
                    own.get("/map")
                stop.wait(0.2)

    client.get("/map")  # the first touch lands before the thread is scheduled
    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(2.0)


def _fetch(c, path, publish=None, timeout=10.0):
    """GET `path` until it answers 200, calling `publish()` before each try."""
    deadline = time.monotonic() + timeout
    r = None
    while time.monotonic() < deadline:
        if publish is not None:
            publish()
        r = c.get(path)
        if r.status_code == 200:
            return r
        time.sleep(0.15)
    pytest.fail(f"GET {path} never answered 200; last answer {r.status_code} {r.text[:200]!r}")


def _map_status(c):
    r = c.get("/map")
    assert r.status_code == 200
    return r.json()


def _subscribed(node, topics, *, at_least=1):
    return all(node.count_subscribers(t) >= at_least for t in topics)


def _pack(columns, offsets, point_step):
    n = len(columns[0])
    buf = np.zeros((n, point_step), dtype=np.uint8)
    for col, offset in zip(columns, offsets):
        buf[:, offset : offset + 4] = np.ascontiguousarray(col, dtype="<f4").view(np.uint8).reshape(n, 4)
    return buf.tobytes()


def _fields(*names):
    return [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate(names)]


def make_xyz_cloud(xyz, stamp, frame="camera_optical_frame"):
    """Dev 1's /perception/depth_cloud layout: x y z float32, 12-byte records, unorganised, not dense."""
    msg = PointCloud2()
    msg.header.stamp, msg.header.frame_id = stamp, frame
    msg.height, msg.width, msg.point_step, msg.row_step, msg.is_dense = 1, len(xyz), 12, 12 * len(xyz), False
    msg.fields = _fields("x", "y", "z")
    msg.data = _pack([xyz[:, 0], xyz[:, 1], xyz[:, 2]], [0, 4, 8], 12)
    return msg


def make_path(rows, stamp, frame="map"):
    msg = Path()
    msg.header.stamp, msg.header.frame_id = stamp, frame
    for x, y, z, qx, qy, qz, qw in ((float(v) for v in row) for row in rows):  # message fields take floats only
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = x, y, z
        ps.pose.orientation.x, ps.pose.orientation.y = qx, qy
        ps.pose.orientation.z, ps.pose.orientation.w = qz, qw
        msg.poses.append(ps)
    return msg


def make_grid(cells, resolution, origin, stamp, yaw=0.0, frame="map"):
    msg = OccupancyGrid()
    msg.header.stamp, msg.header.frame_id = stamp, frame
    msg.info.resolution, msg.info.height, msg.info.width = resolution, cells.shape[0], cells.shape[1]
    msg.info.origin.position.x, msg.info.origin.position.y = origin
    msg.info.origin.orientation.z, msg.info.origin.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    msg.data = cells.astype(np.int8).reshape(-1).tolist()
    return msg


def test_trajectory_published_on_ros_comes_out_of_http_decoded(graph):
    c, pubs = graph["client"], graph["pubs"]
    rows = [(0, 0, 0, 0, 0, 0, 1), (3, 4, 0, 0, 0, 1, 0), (3, 4, 12, 0.5, 0.5, 0.5, 0.5)]

    with watching(c):
        traj = mapread.trajectory(
            _fetch(c, "/map/trajectory", publish=lambda: pubs.path.publish(make_path(rows, pubs.now()))).content)
        status = _map_status(c)

    assert traj["count"] == 3 and traj["length_m"] == pytest.approx(17.0)
    assert traj["poses"].tolist() == [[float(v) for v in r] for r in rows]
    assert status["seq"]["trajectory"] >= 1


def test_heavy_subscriptions_exist_only_while_a_client_is_watching(graph):
    c, node = graph["client"], graph["pub_node"]
    assert _wait(lambda: _subscribed(node, [MAP_STATS]), timeout=3.0), "the map stats subscription is always on"

    with watching(c):
        assert _wait(lambda: _subscribed(node, HEAVY_TOPICS), timeout=8.0), [
            (t, node.count_subscribers(t)) for t in HEAVY_TOPICS]
        assert node.count_subscribers(CLOUD) == 0, "the reconstruction cloud is not a viewer layer any more"
    # no heartbeat for longer than IDLE_S: the demand timer destroys them (it ticks once a second)
    assert _wait(lambda: all(node.count_subscribers(t) == 0 for t in HEAVY_TOPICS), timeout=IDLE_S + 6.0), [
        (t, node.count_subscribers(t)) for t in HEAVY_TOPICS]
    assert _subscribed(node, [MAP_STATS]), "the map stats subscription stays"

    with watching(c):  # and they come back for the next viewer
        assert _wait(lambda: _subscribed(node, HEAVY_TOPICS), timeout=8.0)


def test_pose_resource_and_event_report_the_published_transform(graph):
    c = graph["client"]

    def available():
        body = c.get("/map/pose").json()
        return body if body["available"] else None

    body = _wait(available)
    assert body, "no pose"
    expected = {"x": POSE["x"], "y": POSE["y"], "z": POSE["z"], "qx": 0.0, "qy": 0.0,
                "qz": math.sin(POSE["yaw"] / 2), "qw": math.cos(POSE["yaw"] / 2)}
    assert body["available"] is True
    for key, want in expected.items():
        assert body[key] == pytest.approx(want, abs=1e-6), key
    assert -0.5 < body["ageS"] < 2.0

    got = None
    with c.stream("GET", "/telemetry/stream") as r:
        event = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event == "pose":
                got = json.loads(line[6:])
                if got["available"]:
                    break
    assert got and got["available"] and got["x"] == pytest.approx(POSE["x"])
    assert got["qz"] == pytest.approx(math.sin(POSE["yaw"] / 2))


def test_a_latched_costmap_is_received_when_its_publisher_appears_after_the_subscription(graph):
    c, node = graph["client"], graph["pub_node"]
    cells = np.array([[-1, 0, 10], [20, 100, 50]], dtype=np.int8)
    with watching(c):
        # no publisher yet: the gateway subscribes volatile and has to notice the latched publisher later
        assert _wait(lambda: node.count_subscribers(GRID) >= 1, timeout=8.0)
        assert not node.get_publishers_info_by_topic(GRID)
        # The publisher gets a DDS participant of its own (a new rclpy context): a writer added to a participant
        # the gateway already knows is matched at once, before its first sample, and then the reader's durability
        # would not matter. A new participant is discovered only after the single latched sample was written, so
        # only a transient-local subscription receives it.
        with own_participant("ugv_api_test_costmap") as other:
            pub = other.create_publisher(OccupancyGrid, GRID,
                                         QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
            pub.publish(make_grid(cells, 0.25, (-0.5, 1.0), other.get_clock().now().to_msg(), yaw=0.5))  # once
            out = mapread.grid(_fetch(c, "/map/grid", timeout=12.0).content)
    assert out["cells"].tolist() == cells.tolist()
    assert out["resolution_m"] == pytest.approx(0.25) and (out["origin_x"], out["origin_y"]) == pytest.approx((-0.5, 1.0))
    assert out["origin_yaw"] == pytest.approx(0.5, abs=1e-6)


# Four optical-frame points 2 m ahead, then what the range gate (0.3-8 m) and the NaN check drop.
LIVE_IN_RANGE = np.array([[-0.8, -0.4, 2.0], [0.0, -0.4, 2.0], [-0.8, 0.4, 2.0], [0.0, 0.4, 2.0]], dtype=np.float32)
LIVE_DROPPED = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 9.0], [np.nan, 0.0, 2.0]], dtype=np.float32)


def _expected_live_points(optical):
    """`optical` through the optical -> base -> map chain, with plain numpy and the numbers published above."""
    r_opt = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)  # body = (z, -x, -y)
    yaw = POSE["yaw"]
    r_yaw = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    in_base = optical @ r_opt.T + np.array(CAMERA_MOUNT)
    return in_base @ r_yaw.T + np.array([POSE["x"], POSE["y"], POSE["z"]])


def test_live_layer_is_the_depth_cloud_in_the_map_frame(graph):
    c, pubs = graph["client"], graph["pubs"]
    cloud = np.concatenate([LIVE_IN_RANGE, LIVE_DROPPED])

    def publish():
        pubs.depth_cloud.publish(make_xyz_cloud(cloud, pubs.now()))

    with watching(c):
        live = mapread.cloud(_fetch(c, "/map/live", publish=publish).content)
    assert live["count"] == 4 and not live["has_rgb"]  # too near, too far and NaN are dropped
    assert np.allclose(live["xyz"], _expected_live_points(LIVE_IN_RANGE), atol=1e-4)


def test_live_is_skipped_without_a_transform(graph):
    c, pubs, gw = graph["client"], graph["pubs"], graph["gw"]
    with watching(c):
        before = _map_status(c)
        rejects_before = gw.map_rejects.get("live_no_transform", 0)
        deadline = time.monotonic() + 8.0
        # frames keep arriving (each one is counted as skipped), none of them becomes a live layer
        while time.monotonic() < deadline and gw.map_rejects.get("live_no_transform", 0) < rejects_before + 3:
            pubs.depth_cloud.publish(make_xyz_cloud(LIVE_IN_RANGE, pubs.now(), frame="no_such_frame"))
            time.sleep(0.1)
        after = _map_status(c)
        stats = _wait(lambda: (lambda s: s if s["map_rejects"] > before["stats"]["map_rejects"] else None)(
            _map_status(c)["stats"]))
    assert gw.map_rejects.get("live_no_transform", 0) >= rejects_before + 3
    assert after["seq"]["live"] == before["seq"]["live"], "camera-frame points were published as map-frame points"
    # the skipped frames are visible to the operator, not only in a test-only property: counted and named
    assert stats, _map_status(c)["stats"]
    assert "live_no_transform" in stats["map_last_reject"]


def test_map_stats_pass_through_malformed_json_is_ignored_and_silence_expires(graph):
    c, pubs, gw = graph["client"], graph["pubs"], graph["gw"]

    def seen(*keys):
        stats = _map_status(c)["stats"]
        return stats if all(k in stats for k in keys) else None

    def publish_good():
        pubs.map_stats.publish(String(data=json.dumps({"keyframes": 12, "mode": "mapping",
                                                       "calibration_placeholder": False, "nested": {"a": 1}})))

    deadline = time.monotonic() + 8.0
    stats = None
    while stats is None and time.monotonic() < deadline:
        publish_good()
        stats = _wait(lambda: seen("keyframes"), timeout=0.3)
    assert stats, _map_status(c)
    assert stats["keyframes"] == 12 and stats["mode"] == "mapping" and stats["calibration_placeholder"] is False
    assert "nested" not in stats  # nested values are not part of the flat contract

    publish_good()
    before = gw.map_rejects.get("map_stats", 0)
    pubs.map_stats.publish(String(data="{not json"))
    assert _wait(lambda: gw.map_rejects.get("map_stats", 0) == before + 1)
    assert seen("keyframes"), "malformed JSON must not clear what the source reported"

    # silence: nothing more is published; the source is dropped after STATS_STALE_S
    assert _wait(lambda: "keyframes" not in _map_status(c)["stats"], timeout=STATS_STALE_S + 6.0)
    # only the gateway's own statistics (retained layers, and the health of the map inputs) outlive the silence
    assert set(_map_status(c)["stats"]) <= GATEWAY_STAT_KEYS


# ----------------------------------------------------------------------------- the map side's own second context
# `ros2 launch` starts the gateway with `--ros-args -r __node:=ugv_api --params-file ...`. The map inputs run in a
# context of their own and must not pick any of that up: not the rename (two nodes called /ugv_api) and not the
# parameters (use_sim_time).


@pytest.mark.parametrize("renamed", ["ugv_api", "gateway_renamed"])
def test_the_map_node_does_not_inherit_the_gateways_ros_arguments(renamed):
    args = ["--ros-args", "-r", f"__node:={renamed}", "-p", "use_sim_time:=true"]
    with bare_gateway(args=args, spin=False) as (gw, _maps):
        assert gw.get_name() == renamed and gw.get_parameter("use_sim_time").value is True
        node = gw._map_inputs._node
        assert node.get_name() == "ugv_api_map", "the launch file's remap reached the map node"
        assert node.get_fully_qualified_name() != gw.get_fully_qualified_name()
        assert node.get_parameter("use_sim_time").value is False, "the gateway's parameters reached the map node"
        assert _wait(lambda: ("ugv_api_map", "/") in gw.get_node_names_and_namespaces(), timeout=8.0), (
            gw.get_node_names_and_namespaces())


# ------------------------------------------------------------------------- the map side may not take the gateway down


@pytest.mark.parametrize(
    "override",
    [
        Parameter("map.idle_timeout_s", Parameter.Type.DOUBLE, -1.0),  # fails MapConfig's validation
        Parameter("map.idle_timeout_s", Parameter.Type.INTEGER, 5),  # a YAML integer where a double is declared
    ],
    ids=["invalid-value", "wrong-type"],
)
def test_a_map_only_configuration_error_does_not_abort_the_gateway(override):
    with bare_gateway(override, spin=False) as (gw, maps):
        assert gw._map_inputs is None  # map inputs are off, the rest of the gateway is running
        assert gw.map_cfg == MapConfig()  # create_app still gets usable numbers
        stats = maps.stats()
        assert stats["map_inputs_alive"] is False
        assert "idle_timeout_s" in stats["map_last_reject"] and "map inputs are off" in stats["map_last_reject"]
        assert _has_estop_publisher(gw)  # the rest of the gateway is there


def test_a_failure_to_start_the_map_side_does_not_abort_the_gateway_and_leaks_nothing(monkeypatch):
    made = []

    class Recorded(Context):
        def __init__(self):
            super().__init__()
            made.append(self)

    def no_executor(*_a, **_k):
        raise RuntimeError("no executor today")

    def map_threads():
        return {th for th in threading.enumerate() if th.name == "ugv_api_map_ros"}

    threads_before = map_threads()  # the module's own gateway has one
    monkeypatch.setattr(ros_node, "Context", Recorded)
    monkeypatch.setattr(ros_node, "SingleThreadedExecutor", no_executor)
    with bare_gateway(spin=False) as (gw, maps):
        assert gw._map_inputs is None
        assert len(made) == 1 and made[0].ok() is False, "the second context was left running"
        stats = maps.stats()
        assert stats["map_inputs_alive"] is False and "no executor today" in stats["map_last_reject"]
        assert _has_estop_publisher(gw)
    assert map_threads() == threads_before


# --------------------------------------------------------------------------------- map input health in the stats


def test_map_input_health_is_part_of_the_status_stats(graph):
    c = graph["client"]
    stats = _wait(lambda: (lambda s: s if s.get("map_inputs_alive") is True else None)(_map_status(c)["stats"]))
    assert stats, _map_status(c)["stats"]
    assert stats["map_restarts"] == 0 and isinstance(stats["map_rejects"], int) and "map_last_reject" in stats
    assert set(stats) >= {"map_inputs_alive", "map_rejects", "map_restarts", "map_last_reject"}


def test_a_hung_map_thread_shows_as_not_alive_while_the_gateway_keeps_answering(graph):
    c, gw = graph["client"], graph["gw"]
    assert _wait(lambda: _map_status(c)["stats"].get("map_inputs_alive") is True)
    assert _wait(lambda: _safety(c)["ok"]), _safety(c)
    release = threading.Event()
    _on_map_thread(gw, lambda: release.wait(30.0))  # the map thread is stuck inside a callback
    slowest = 0.0
    try:
        deadline = time.monotonic() + MAP_ALIVE_BOUND
        alive = True
        while alive and time.monotonic() < deadline:
            started = time.monotonic()
            assert _safety(c)["ok"] is True  # the §12 table does not notice a map side that hangs
            assert c.get("/health").status_code == 200
            slowest = max(slowest, time.monotonic() - started)
            alive = _map_status(c)["stats"].get("map_inputs_alive")
            time.sleep(0.1)
        assert alive is False, f"still reported alive {MAP_ALIVE_BOUND:.0f} s after the map thread hung"
        assert slowest < 1.0, f"the other endpoints slowed down to {slowest:.2f} s"
    finally:
        release.set()
    assert _wait(lambda: _map_status(c)["stats"].get("map_inputs_alive") is True, timeout=6.0), "never recovered"


def test_a_dead_map_thread_is_reported_and_the_stats_it_held_still_expire(graph):
    pubs = graph["pubs"]
    with bare_gateway(Parameter("map.stats_stale_s", Parameter.Type.DOUBLE, STATS_STALE_S)) as (gw, maps):
        def have_stats():
            pubs.map_stats.publish(String(data=json.dumps({"keyframes": 7})))
            return maps.stats().get("keyframes") == 7

        assert _wait(have_stats, timeout=8.0), maps.stats()
        assert _wait(lambda: maps.stats().get("map_inputs_alive") is True)

        def die():
            raise _MapThreadGone("the map thread is gone")

        _on_map_thread(gw, die)
        assert _wait(lambda: maps.stats().get("map_inputs_alive") is False, timeout=MAP_ALIVE_BOUND), maps.stats()
        assert "map_thread_exit" in maps.stats()["map_last_reject"]
        assert not gw._map_inputs._thread.is_alive()
        # nothing is published any more, and the thread that used to clear the stale entries is gone
        assert _wait(lambda: "keyframes" not in maps.stats(), timeout=STATS_STALE_S + 4.0), maps.stats()
        assert maps.stats()["map_inputs_alive"] is False  # and it stays reported


def test_the_supervisor_counts_a_restart_and_names_what_failed():
    with bare_gateway() as (gw, maps):
        assert _wait(lambda: maps.stats().get("map_inputs_alive") is True)

        def fail():
            raise RuntimeError("a callback the guard does not wrap")

        _on_map_thread(gw, fail)
        stats = _wait(lambda: (lambda s: s if s.get("map_restarts", 0) >= 1 else None)(maps.stats()), timeout=6.0)
        assert stats, maps.stats()
        assert "map_executor" in stats["map_last_reject"] and "a callback the guard does not wrap" in stats["map_last_reject"]
        assert stats["map_rejects"] >= 1
        assert _wait(lambda: maps.stats().get("map_inputs_alive") is True, timeout=6.0), "did not come back after a restart"


def test_a_failing_map_health_update_never_stops_the_gateways_executor():
    """The map health timer shares the gateway's executor with the e-stop republish and the §12 watches: an exception
    in it is logged, never raised, so the executor keeps spinning (and the timer keeps firing)."""
    with bare_gateway() as (gw, maps):
        assert _wait(lambda: maps.stats().get("map_inputs_alive") is True)
        calls = []

        def broken():
            calls.append(time.monotonic())
            raise RuntimeError("map health is broken")

        gw._map_inputs.publish_health = broken
        # an unguarded timer would end the spin at the first call; guarded, it keeps being called every period
        assert _wait(lambda: len(calls) >= 4, timeout=4.0 * ros_node.MAP_HEALTH_PERIOD_S + 3.0), f"{len(calls)} calls"


# ------------------------------------------------------------------------------- what a map layer may be made of


def test_layers_in_another_frame_are_refused_and_an_empty_frame_is_accepted(graph):
    c, pubs, node, gw = graph["client"], graph["pubs"], graph["pub_node"], graph["gw"]
    cells = np.zeros((6, 8), dtype=np.int8)
    kinds = ("frame_trajectory", "frame_grid")

    def counts():
        rejects = gw.map_rejects
        return {kind: rejects.get(kind, 0) for kind in kinds}

    def served(path, decode):
        r = c.get(path)
        return decode(r.content) if r.status_code == 200 else None

    before = counts()
    with watching(c), own_participant("ugv_api_test_odom_costmap") as other:
        assert _wait(lambda: _subscribed(node, HEAVY_TOPICS), timeout=8.0)
        costmap = other.create_publisher(OccupancyGrid, GRID, QoSProfile(depth=1))

        def publish_odom_layers():
            now = pubs.now()
            pubs.path.publish(make_path([(901, 902, 903, 0, 0, 0, 1)], now, frame="odom"))
            costmap.publish(make_grid(cells, 0.2, (0.0, 0.0), now, frame="odom"))  # a local costmap, say

        refused = _wait(lambda: (publish_odom_layers(), all(counts()[k] > before[k] for k in kinds))[1], timeout=12.0)
        assert refused, f"not every layer was refused: {before} -> {counts()}"
        # nothing of it was served as a map-frame layer
        trajectory = served("/map/trajectory", mapread.trajectory)
        assert trajectory is None or 901.0 not in trajectory["poses"][:, 0]
        grid = served("/map/grid", mapread.grid)
        assert grid is None or grid["resolution_m"] != pytest.approx(0.2)

        # a message with no frame at all is taken as the map frame (some publishers leave it empty)
        def publish_unframed():
            costmap.publish(make_grid(cells, 0.125, (0.0, 0.0), other.get_clock().now().to_msg(), frame=""))
            grid = served("/map/grid", mapread.grid)
            return grid is not None and grid["resolution_m"] == pytest.approx(0.125)

        assert _wait(publish_unframed, timeout=8.0), "a grid with an empty frame_id was refused"


def test_an_organised_cloud_with_padded_rows_is_refused(graph):
    c, pubs, node, gw = graph["client"], graph["pubs"], graph["pub_node"], graph["gw"]
    msg = PointCloud2()
    msg.header.stamp, msg.header.frame_id = pubs.now(), "map"
    msg.height, msg.width, msg.point_step, msg.row_step, msg.is_dense = 2, 3, 12, 40, True  # 4 bytes of padding per row
    msg.fields = _fields("x", "y", "z")
    msg.data = bytes(80)
    before = gw.map_rejects.get("live", 0)
    with watching(c):
        assert _wait(lambda: _subscribed(node, [DEPTH_CLOUD]), timeout=8.0)
        assert _wait(lambda: (pubs.depth_cloud.publish(msg), gw.map_rejects.get("live", 0) > before)[1])
        r = c.get("/map/live")
    assert r.status_code != 200 or mapread.cloud(r.content)["source_count"] != 6


def test_a_latched_arbiter_status_published_before_discovery_is_received(graph):
    # The Dev 5 arbiter publishes /ugv/safety_status latched and only on change: a gateway that misses the one
    # sample reports the arbiter absent while it is running and holding the robot.
    c = graph["client"]
    with own_participant("ugv_api_test_arbiter") as other:
        pub = other.create_publisher(String, "/ugv/safety_status",
                                     QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        pub.publish(String(data="L1 ESTOP: e_stop"))  # once, before the gateway has discovered this participant
        body = _wait(lambda: (lambda b: b if b["arbiter"]["present"] else None)(_safety(c)), timeout=8.0)
    assert body and body["arbiter"]["status"] == "L1 ESTOP: e_stop", _safety(c)
