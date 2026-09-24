# Dev 2 — SLAM & Localization (`ugv_localization`)

**Authority:** [`architecture.md`](../../../architecture.md) wins over [`dev.md`](../../../dev.md) on conflict.
**Decisions:** [`mindmap.md`](../../../mindmap.md) (D1–D6). **Gaps vs dev.md:** [CONFLICTS.md](CONFLICTS.md).
**Contract:** [interfaces.md](interfaces.md). **Environment:** [ENVIRONMENT.md](ENVIRONMENT.md). **Honesty:** [SENSOR_HONESTY.md](SENSOR_HONESTY.md).

## What Dev 2 is

The **pose half of the brain** (architecture §2): RTAB-Map + the TF chain + an honest "is the pose usable" signal.
Dev 2 does **not** own the camera (Dev 5), does **not** consume the perception mask (Dev 1, §5), and does **not** plan (Dev 4).

```
Dev 5 camera ─ Image + CameraInfo ─┐
Dev 5 diff-drive ─ /wheel/odom ──┐ │
                                 ▼ ▼
      odom_tf_bridge ── odom->base_link TF + /odom
                                 │
      rtabmap (mono, RGB-only) ──┴── map->odom TF, /rtabmap/info, rtabmap.db
                                 │
      pose_validity_node ─────── /ugv/pose_valid (20 Hz heartbeat, fail closed) → Dev 5
```

## Laws

| Law | Meaning here |
|---|---|
| Mono is the minimum, not the recommendation (§6, §10, kill list) | Scale from wheel odom; drift documented; never marketed as outdoor-meter-ready |
| Pose valid only if valid (§10.1) | TF present + fresh, odom sane, camera fresh, RTAB-Map alive, localized, drift budget |
| Fail closed | `/ugv/pose_valid=false` at startup, on any missing input, on clock reset |
| Never restamp | TF edges carry the odometry stamp, never "now" |
| No fake sensors | Camera YAML only from a real CameraInfo / calibration; tests use numeric tables |
| One owner per TF edge | Dev 2 owns `map->odom->base_link` (D6); Dev 5 publishes no odom TF |

## Build order and status (2026-09-23)

| # | Module | Code | Status |
|---|---|---|---|
| L1 | Camera calibration validate/load | `camera/` | **done, tested** |
| L2 | Wheel-odom gate → odom->base_link | `odom/`, `nodes/odom_tf_bridge.py` | **runs on Lyrical**: TF 20 Hz, jitter 0.2 ms |
| L3 | RTAB-Map mono config + launch | `config/rtabmap_mono.yaml`, `launch/localization.launch.py` | **launches**; param names verified; values untuned (needs camera) |
| L4 | mapping / localize modes | `modes/`, `nodes/mode_cli.py` | **runs**: fail-fast, db save, localize relaunch, runtime switch |
| L5 | Pose validity | `validity/`, `nodes/pose_validity_node.py` | **runs**: false @ 20 Hz with reasons; `true` path needs camera |
| L6 | TF rate/jitter check | `tfcheck/`, `nodes/tf_rate_check.py` | **runs** (odom->base_link); map->odom needs camera |
| L7 | Bag harness | `launch/bag_eval.launch.py`, `scripts/record_eval_bag.sh` | written; needs a recorded bag |
| L8 | Drift metrics + report | `drift/`, `tools/drift_report.py`, `nodes/drift_eval.py` | kernel + CLI **tested**; node starts/exits cleanly; needs GT + map->odom |
| L9 | Sensor honesty numbers | [SENSOR_HONESTY.md](SENSOR_HONESTY.md) | protocol written; **no numbers until sim runs** |

Runtime-verified on WSL Ubuntu 26.04 + Lyrical, 2026-09-24 ([ENVIRONMENT.md](ENVIRONMENT.md)). Remaining blocker: Dev 5 sim camera + `/wheel/odom` + ground truth.

## Run the tests (no ROS needed)

```bash
cd ugv_nav/ugv_localization
python -m pip install numpy pyyaml pytest
python -m pytest            # ROS smoke tests auto-skip without rclpy/launch_ros
```

With ROS: `colcon build --packages-select ugv_localization && colcon test --packages-select ugv_localization`.

## Run the stack (once ROS + Dev 5 sim exist)

```bash
ros2 launch ugv_localization localization.launch.py mode:=mapping profile:=sim fresh_db:=true
# drive the robot around a loop, Ctrl-C → ~/.ros/ugv/rtabmap.db saved
ros2 launch ugv_localization localization.launch.py mode:=localize profile:=sim
ros2 topic echo /ugv/pose_valid
ros2 topic echo /ugv/localization_status
ros2 run ugv_localization tf_rate_check --ros-args -p use_sim_time:=true
ros2 run tf2_ros tf2_echo map base_link
```
