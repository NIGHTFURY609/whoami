#!/usr/bin/env python3
"""Opt-in load rig for the gateway's map inputs. NOT part of the test suite (pytest only collects `test/`).

What it measures. The gateway's map inputs receive a 1,000,000 point PointCloud2 (Dev 1's XYZ layout, 12 bytes a
point, 12 MB a message) at 2 Hz on /perception/depth_cloud while a viewer polls GET /api/v1/map (5 Hz) and
GET /api/v1/map/live (4 Hz). (Until 2026-10-03 it loaded RTAB-Map's /rtabmap/cloud_map, 32 MB a message, which the
viewer no longer subscribes; the numbers below were measured with that cloud.) Meanwhile the six §12 inputs (camera_info, mask, perception_degraded, pose_valid,
nav2_heartbeat, TF) arrive at 20 Hz from a separate process, and GET /api/v1/safety/status is sampled at 10 Hz.
The result is the worst age each §12 watch showed and how many samples had a watch that was not ok.

Why it exists. It is the evidence for running the map inputs on a node of their own in a second rclpy context
(ugv_api.ros_node._MapInputs): with only a separate callback group on the gateway node the tf watch went stale for
about 1 to 3 s in 6 of 10 runs of 30 s (worst age 3.07 s) although no callback ran long; with the second DDS
participant 13 of 13 runs stayed at a worst tf age of 0.17 to 0.39 s. Re-run it after touching that code.

How to run. Needs a sourced ROS 2 Lyrical environment with numpy and httpx. The cloud is big, so give it a spare ROS
domain that no robot is on; the script refuses to start without an explicit ROS_DOMAIN_ID. From the package
directory (ugv_nav/ugv_api):

    ROS_DOMAIN_ID=78 python3 scripts/map_load_probe.py                      # 30 s with the cloud
    ROS_DOMAIN_ID=78 python3 scripts/map_load_probe.py --no-load            # the same without the cloud (baseline)
    ROS_DOMAIN_ID=78 python3 scripts/map_load_probe.py --seconds 15 --points 500000

It starts the gateway (config/api.yaml, a free port) and the two publishers as child processes and stops them again.
Exit status 0: no watch was ever not ok and no watch's worst age exceeded --max-age (default 1.0 s; the §12 limits
are 0.5 s, and a stale sample is what trips them). 1: it did, or the gateway never came up. The JSON on stdout also
shows the gateway's map input health (`map_inputs_alive`, `map_rejects`, `map_restarts`, `map_last_reject`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------ publishers


def spin_quietly(node) -> None:
    """rclpy.spin that ends without a traceback when the probe stops the process (SIGTERM shuts the context down)."""
    import rclpy
    from rclpy.executors import ExternalShutdownException

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():  # a context that went away under the spin is the normal end, anything else is not
            raise


def run_inputs() -> None:
    """The six §12 inputs at 20 Hz, plus TF map -> base_link."""
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import Bool, String
    from tf2_ros import TransformBroadcaster

    rclpy.init()
    node = rclpy.create_node("map_load_probe_inputs")
    cam = node.create_publisher(CameraInfo, "/camera/camera_info", 10)
    mask = node.create_publisher(Image, "/segmentation/mask", 10)
    degraded = node.create_publisher(Bool, "/ugv/perception_degraded", 10)
    pose = node.create_publisher(Bool, "/ugv/pose_valid", 10)
    loc = node.create_publisher(String, "/ugv/localization_status", 10)
    heartbeat = node.create_publisher(Bool, "/ugv/nav2_heartbeat", 10)
    tf = TransformBroadcaster(node)

    def tick() -> None:
        now = node.get_clock().now().to_msg()
        info = CameraInfo()
        info.header.stamp, info.header.frame_id = now, "camera_optical_frame"
        info.width, info.height, info.k = 8, 6, [10.0, 0.0, 4.0, 0.0, 10.0, 2.0, 0.0, 0.0, 1.0]
        cam.publish(info)
        image = Image()
        image.header.stamp, image.header.frame_id, image.encoding = now, "camera_optical_frame", "mono8"
        mask.publish(image)
        degraded.publish(Bool(data=False))
        pose.publish(Bool(data=True))
        loc.publish(String(data="tracking"))
        heartbeat.publish(Bool(data=True))
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = now, "map", "base_link"
        t.transform.translation.x = 1.5
        t.transform.rotation.z, t.transform.rotation.w = math.sin(0.2), math.cos(0.2)
        tf.sendTransform(t)

    node.create_timer(0.05, tick)
    spin_quietly(node)


def run_cloud(points: int, hz: float) -> None:
    """An N point XYZ cloud at `hz`, volatile like Dev 1's /perception/depth_cloud. It is stamped in the map frame
    (an identity TF for the gateway) with z, the optical depth the gateway range-gates on, inside 0.5 .. 7.5 m."""
    import numpy as np
    import rclpy
    from rclpy.qos import QoSProfile
    from sensor_msgs.msg import PointCloud2, PointField

    step = 12  # Dev 1's layout: x y z float32
    rng = np.random.default_rng(1)
    xyz = np.column_stack([rng.uniform(-5, 5, points), rng.uniform(-5, 5, points),
                           rng.uniform(0.5, 7.5, points)]).astype("<f4")
    payload = xyz.view(np.uint8).reshape(-1)

    rclpy.init()
    node = rclpy.create_node("map_load_probe_cloud")
    pub = node.create_publisher(PointCloud2, "/perception/depth_cloud", QoSProfile(depth=1))
    msg = PointCloud2()
    msg.header.frame_id = "map"
    msg.height, msg.width, msg.point_step, msg.row_step, msg.is_dense = 1, points, step, step * points, False
    msg.fields = [PointField(name=c, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, c in enumerate("xyz")]
    msg.data = payload

    def publish() -> None:
        msg.header.stamp = node.get_clock().now().to_msg()
        pub.publish(msg)

    node.create_timer(1.0 / hz, publish)
    spin_quietly(node)


# ------------------------------------------------------------------------------------------------ probe


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def sample(base: str, seconds: float, with_load: bool) -> dict:
    import httpx

    stop = threading.Event()
    worst: dict[str, float] = {}
    not_ok: dict[str, int] = {}
    latency: list[float] = []
    cloud = {"ok": 0, "other": 0, "bytes": 0, "slowest_s": 0.0}
    last_map: dict = {}

    def poll_map() -> None:
        with httpx.Client(base_url=base, timeout=30.0) as c:
            while not stop.is_set():
                try:
                    last_map.update(c.get("/map").json())
                    t0 = time.monotonic()
                    r = c.get("/map/live")
                    if r.status_code == 200:
                        cloud["ok"] += 1
                        cloud["bytes"] = len(r.content)
                        cloud["slowest_s"] = max(cloud["slowest_s"], time.monotonic() - t0)
                    else:
                        cloud["other"] += 1  # 503 until the first cloud has arrived
                except httpx.HTTPError:
                    cloud["other"] += 1
                stop.wait(0.2)

    loader = threading.Thread(target=poll_map, daemon=True)
    if with_load:
        loader.start()
    samples = 0
    deadline = time.monotonic() + seconds
    with httpx.Client(base_url=base, timeout=10.0) as c:
        while time.monotonic() < deadline:
            t0 = time.monotonic()
            body = c.get("/safety/status").json()
            latency.append(time.monotonic() - t0)
            samples += 1
            for w in body["watches"]:
                if w["ageS"] is not None:
                    worst[w["name"]] = max(worst.get(w["name"], -math.inf), w["ageS"])
                if not w["ok"] and w["name"] != "e_stop":  # nothing publishes /ugv/e_stop here
                    not_ok[w["name"]] = not_ok.get(w["name"], 0) + 1
            time.sleep(0.1)
    stop.set()
    if with_load:
        loader.join(15)
    with httpx.Client(base_url=base, timeout=10.0) as c:
        last_map.update(c.get("/map").json())
    stats = last_map.get("stats", {})
    return {
        "samples": samples,
        "worst_age_s": {k: round(v, 3) for k, v in sorted(worst.items())},
        "not_ok_samples": not_ok,
        "safety_status_latency_s": {"max": round(max(latency), 3), "mean": round(sum(latency) / len(latency), 4)},
        "cloud": cloud if with_load else None,
        "map_input_health": {k: stats.get(k) for k in ("map_inputs_alive", "map_rejects", "map_restarts",
                                                      "map_last_reject")},
    }


def orchestrate(args: argparse.Namespace) -> int:
    import httpx

    port = args.port or free_port()
    base = f"http://127.0.0.1:{port}/api/v1"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(PACKAGE_DIR), os.environ.get("PYTHONPATH")])),
               PYTHONDONTWRITEBYTECODE="1")
    me = str(Path(__file__).resolve())
    children = [subprocess.Popen(
        [sys.executable, "-c", "from ugv_api.main import main; main()", "--ros-args", "--params-file",
         str(PACKAGE_DIR / "config" / "api.yaml"), "-p", f"port:={port}"],
        env=env, cwd=PACKAGE_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
    children.append(subprocess.Popen([sys.executable, me, "--role", "inputs"], env=env))
    if not args.no_load:
        children.append(subprocess.Popen([sys.executable, me, "--role", "cloud", "--points", str(args.points),
                                          "--hz", str(args.hz)], env=env))
    try:
        up = False
        for _ in range(120):  # up to 60 s: the gateway answers and every §12 watch is ok
            try:
                body = httpx.get(f"{base}/safety/status", timeout=2.0).json()
                if body["watches"] and all(w["ok"] for w in body["watches"] if w["name"] != "e_stop"):
                    up = True
                    break
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(0.5)
        if not up:
            print("the gateway never came up with every watch ok", file=sys.stderr)
            return 1
        report = sample(base, args.seconds, not args.no_load)
    finally:
        for child in children:
            child.terminate()
        for child in children:
            try:
                child.wait(10)
            except subprocess.TimeoutExpired:
                child.kill()
    report.update(seconds=args.seconds, points=None if args.no_load else args.points,
                  hz=None if args.no_load else args.hz, max_age_s=args.max_age)
    worst = max(report["worst_age_s"].values(), default=math.inf)
    report["pass"] = not report["not_ok_samples"] and worst <= args.max_age
    print(json.dumps(report, indent=1))
    return 0 if report["pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--role", choices=("probe", "inputs", "cloud"), default="probe", help=argparse.SUPPRESS)
    ap.add_argument("--seconds", type=float, default=30.0, help="how long to sample (default 30)")
    ap.add_argument("--points", type=int, default=1_000_000, help="points per cloud (default 1,000,000 = 12 MB)")
    ap.add_argument("--hz", type=float, default=2.0, help="clouds per second (default 2)")
    ap.add_argument("--no-load", action="store_true", help="do not publish the cloud (baseline)")
    ap.add_argument("--max-age", type=float, default=1.0, help="fail above this worst watch age, s (default 1.0)")
    ap.add_argument("--port", type=int, default=0, help="gateway port (default: a free one)")
    args = ap.parse_args()
    if "ROS_DOMAIN_ID" not in os.environ:
        ap.error("set ROS_DOMAIN_ID to a spare domain (for example 78): the cloud is 12 MB a message")
    if args.role == "inputs":
        run_inputs()
        return 0
    if args.role == "cloud":
        run_cloud(args.points, args.hz)
        return 0
    return orchestrate(args)


if __name__ == "__main__":
    sys.exit(main())
