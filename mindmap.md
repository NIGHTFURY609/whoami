# Dev 2 mindmap — SLAM & Localization decisions

**Owner:** Dev 2 · **Authority:** architecture.md > dev.md (only on conflict) · **Date:** 2026-09-23

## Decisions
| # | Topic | Decision | Source / reason |
|---|---|---|---|
| D1 | Runtime | **ROS 2 Lyrical Luth** (CLAUDE.md overrides architecture/dev.md "Jazzy"). Devs work independently; no access to Dev 1's box; Ubuntu version TBD (Lyrical pairs with 26.04). Lyrical core is Tier 1 on **Windows 11**, but rtabmap_ros / Gazebo are not confirmed on Windows → Dev 2 runs ROS in **WSL2 Ubuntu 26.04 + Lyrical** (distro `Ubuntu-26.04` installed 2026-09-24; binaries confirmed by user: rtabmap_ros, Gazebo **Jetty** + ros_gz, RViz2); kernels/tests run on Windows Python. Code stays distro-agnostic | User 2026-09-23 / 2026-09-24 |
| D1a | Build scope now | Build everything that does **not** need ROS / sim / Dev 5: docs, package skeleton, configs, pure kernels (TDD, runnable on plain Python), rclpy nodes + launch written but **unverified until ROS is installed** | User |
| D2 | Sensor | **Mono camera only** | User. Architecture §6/§10: mono = minimum → drift docs + VO-lost hold mandatory |
| D3 | Test data | **Dev 5 Gazebo sim camera = main focus**; real-world later | User. Sim/bag = eval profiles (§4), not product proof |
| D4 | Layout | **Architecture §7**: `ugv_nav/ugv_localization/`, `ugv_nav/config/cameras/`, `ugv_nav/docs/` | User |
| D5 | Metric scale | **Wheel odometry from Dev 5** (`nav_msgs/Odometry`) | User. RTAB-Map has no mono VO; needs external odom |
| D6 | TF ownership | **Dev 2 owns full `map->odom->base_link`**. Dev 5 diff-drive publishes `/wheel/odom` topic only (no TF) | User. Architecture silent → dev.md §3 contract applies |

## What architecture.md says (and doesn't)
- §2/§6: brain = "RTAB-Map VO/SLAM" — assumes RTAB-Map produces VO.
- §6/§10: stereo/RGB-D recommended; **mono minimum + drift docs + VO-lost hold**.
- §1: vision is the **primary** sensor (not sole) → wheel odom for scale is allowed.
- §8.5/§10.1/§12: required TFs must exist; missing TF → invalid pose → hold. No TF owner named.
- Kill list: "mono = recommended outdoor RTAB-Map", "ORB-SLAM3 bake-off".
- **Silent on:** wheel odom, IMU, EKF, TF edge ownership.

## Gap noted (architecture vs mono reality)
Architecture assumes RTAB-Map does VO; RTAB-Map cannot do monocular VO alone.
Resolution: wheel odom = odom frame (scale); RTAB-Map mono = visual loop closure + relocalization correcting `map->odom`.
"VO-lost" for mono = no visual constraint within a drift budget (distance/time since last visual correction).

## Consequences
- Dev 5 must publish `/wheel/odom` with covariance and `publish_odom_tf=false`.
- Mono RTAB-Map produces **no occupancy `/map`** (no depth) → Dev 3 must not expect a static map layer from Dev 2.
- Sim world must be visually textured for features (Dev 5).
- Drift benchmark uses Gazebo ground-truth pose (Dev 5 to expose).

## Rejected
- Wheel+IMU EKF (no IMU planned; robot_localization on Lyrical uncertain) — revisit later.
- Depth Anything pseudo-RGB-D → creates Dev1→Dev2 dependency; architecture wants branches independent.
- Dev 5 owning `odom->base_link` → splits TF ownership vs dev.md.

## Open / to verify
- ~~Team distro~~ → Lyrical on Ubuntu 26.04, binaries exist (resolved 2026-09-24). Install via `ugv_nav/docs/localization/setup_wsl_lyrical.sh`.
- Exact RTAB-Map mono params (e.g. `Mem/StereoFromMotion`) via `rtabmap --params`.
- Dev 5 topic names: camera image/info, `/wheel/odom`, ground-truth pose.
