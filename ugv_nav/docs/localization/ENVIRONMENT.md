# Environment (Dev 2)

**Status 2026-09-24:** team distro is **ROS 2 Lyrical Luth** (CLAUDE.md; overrides "Jazzy" in architecture.md / dev.md).
Devs work independently; exact Ubuntu version TBD (Lyrical's Tier 1 Linux is Ubuntu 26.04). See mindmap D1.
Code is distro-agnostic: plain `rclpy`, `tf2_ros`, `rtabmap_msgs/Info` (fields read defensively).

## Can Dev 2 work on Windows?

| Piece | Windows 11 native (Lyrical) | WSL2 Ubuntu 26.04 + Lyrical |
|---|---|---|
| Kernels + tests (plain Python) | **yes — runs today** | yes |
| ROS core, our rclpy nodes, launch | yes — Lyrical is **Tier 1 on Windows 11** | yes |
| `rtabmap_ros` (the SLAM itself) | **not confirmed** — no known Windows ROS 2 build; would be a source build (OpenCV, PCL, g2o…) | **binary** (`ros-lyrical-rtabmap-ros`: rtabmap_slam, rtabmap_odom, rtabmap_launch) |
| Gazebo sim camera (Dev 5 world) | weak / partial on Windows | **binary**: Gazebo **Jetty** + `ros_gz` (rendering via WSLg) |
| Talking to Linux teammates | yes — DDS is cross-OS (same distro + RMW + `ROS_DOMAIN_ID`, open firewall) | yes (`networkingMode=mirrored`) |

**Decision:** edit + unit-test on Windows; run ROS / RTAB-Map / sim in **WSL2 Ubuntu 26.04 + Lyrical** so the runtime matches the Linux teammates. Native Windows ROS is fine for lightweight nodes (e.g. echoing `/ugv/pose_valid` from a teammate's robot), not for the SLAM stack.

## WSL2 setup

WSL distro **`Ubuntu-26.04`** (26.04.1 LTS) is installed on this box (2026-09-24). The older `Ubuntu` (24.04)
is the Jazzy pairing — don't use it for ROS. Note `docker-desktop` is the WSL default, so always pass `-d Ubuntu-26.04`.

Install (once, in your own WSL terminal — needs your sudo password):

```bash
wsl -d Ubuntu-26.04
bash /mnt/n/coding/projects/whoami/ugv_nav/docs/localization/setup_wsl_lyrical.sh
```

Build + test — sources stay on N: (edited from Windows), build output lives on the Linux filesystem:

```bash
source /opt/ros/lyrical/setup.bash
mkdir -p ~/ugv_ws && cd ~/ugv_ws
colcon build --base-paths /mnt/n/coding/projects/whoami/ugv_nav --packages-select ugv_localization
colcon test  --base-paths /mnt/n/coding/projects/whoami/ugv_nav --packages-select ugv_localization
colcon test-result --verbose
source ~/ugv_ws/install/setup.bash
```

WSL caveats:
- **RAM:** `%UserProfile%\.wslconfig` sets `memory=12GB`, `swap=8GB` (host 15.6 GB). Takes effect after `wsl --shutdown`; applies to every distro including docker-desktop.
- **Gazebo rendering** (the sim camera needs it) goes through WSLg; it works but can be slow. Check the sim camera frame rate before trusting TF/latency numbers.
- **Real USB camera** later needs `usbipd-win` to attach it to WSL.
- **Multi-machine DDS** (talking to other devs' boxes) needs `networkingMode=mirrored` in `.wslconfig`.
- Keep `build/ install/ log/` on the Linux filesystem (`~/ugv_ws`, as above); only the small source tree is read over `/mnt/n`. If builds get slow, clone the repo into `~` instead.

## Phase 0 checklist — DONE 2026-09-24 on WSL `Ubuntu-26.04`

| Check | Result |
|---|---|
| Versions | ROS 2 Lyrical (`ros-lyrical-desktop` 0.13.0), rtabmap_ros **0.23.7**, ros_gz 3.0.10, Gazebo **Jetty 10.5.0**, Python 3.14.4 |
| `check_rtabmap_params` | OK — all 19 of our parameter names exist (414 available). Names exist ≠ values tuned |
| `rtabmap_msgs/Info` fields | `header`, `loop_closure_id`, `proximity_detection_id`, `landmark_id` present |
| `colcon test` | **135 tests, 0 failures, 0 skipped** (ROS smoke tests run) |
| Launch `mode:=localize` without db | fails fast: "database … does not exist — run mode:=mapping first" |
| Launch `mode:=mapping` | nodes up: `odom_tf_bridge`, `/rtabmap/rtabmap`, `pose_validity`; db saved on shutdown |
| `/ugv/pose_valid` | `false` at 20.0 Hz with reasons `tf_missing, camera_missing, slam_missing` (no camera yet) |
| `/wheel/odom` → TF `odom->base_link` | passes; `tf_rate_check`: 20.0 Hz, jitter p95 0.2 ms |
| `mode_cli localize / mapping / backup` | ok against live RTAB-Map |
| Relaunch `mode:=localize` on saved db | RTAB-Map "Localization mode (Mem/IncrementalMemory=false)" |

**Not yet verified (needs Dev 5 sim camera):** `map->odom`, loop closure / relocalization, pose_valid going `true`, drift numbers.

### Lyrical-specific findings (fixed)
- `rclpy` removed `logger.warn` → use `.warning` (crashed `pose_validity_node`; guarded by `test_s3`).
- Destroying a node after SIGINT raises → nodes use `with rclpy.init():` and catch `ExternalShutdownException`.
- RTAB-Map **services** are node-private: `/rtabmap/rtabmap/set_mode_*`, `/rtabmap/rtabmap/backup`. **Topics** stay at `/rtabmap/info`, `/rtabmap/mapData`.
- setuptools ignores `tests_require` → colcon ran 0 tests; `setup.py` uses `extras_require={"test": [...]}`.
- `/opt/ros/lyrical/setup.bash` fails under `set -u` (`AMENT_TRACE_SETUP_FILES` unbound) — don't source it in `set -u` scripts.
- `ros2 launch` logs "process has died … exit code -2" for every node (ours and RTAB-Map's C++ node) when the terminal's SIGINT hits the process group — expected, not a crash.
- WSL prints "Failed to start the systemd user session" on `wsl -d Ubuntu-26.04 -- …` — harmless for ROS.

### Driving it from Windows (Git Bash)
Git Bash rewrites `/mnt/...` paths; prefix with `MSYS_NO_PATHCONV=1`, or pipe a script: `wsl -d Ubuntu-26.04 -- bash -s < script.sh`.
