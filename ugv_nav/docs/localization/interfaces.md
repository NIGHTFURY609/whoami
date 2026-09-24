# Dev 2 interface contract

Topic names marked **TBD** are proposals until Dev 5 confirms; every one is a launch argument.

## Inputs (Dev 2 consumes)

| Topic / TF | Type | From | Rule |
|---|---|---|---|
| `/camera/image_raw` (**TBD**) | `sensor_msgs/Image` | Dev 5 | stamp = exposure time; `frame_id` = optical frame (`camera_optical_frame`) |
| `/camera/camera_info` (**TBD**) | `sensor_msgs/CameraInfo` | Dev 5 | same stamp/frame as image; real K (zero K → `camera_info_invalid`) |
| `/wheel/odom` (**TBD**) | `nav_msgs/Odometry` | Dev 5 diff-drive (sim plugin / motor driver) | `frame_id=odom`, `child_frame_id=base_link`, pose covariance filled, ≥ 15 Hz, **no TF** |
| `/tf_static` `base_link->camera_link->camera_optical_frame` | TF | Dev 5 URDF / robot_state_publisher | camera extrinsics |
| `/clock` | `rosgraph_msgs/Clock` | Dev 5 sim / bag play | `profile:=sim|bag` → `use_sim_time` |
| `/ground_truth/odom` (**TBD**, eval only) | `nav_msgs/Odometry` | Dev 5 sim | drift benchmark only; never used by the product path |

## Outputs (Dev 2 publishes)

| Topic / TF | Type | To | Rule |
|---|---|---|---|
| TF `odom->base_link` | TF | Dev 3, 4, 5 | stamp = wheel-odom stamp; only gated samples |
| TF `map->odom` | TF | Dev 3, 4, 5 | RTAB-Map, 20 Hz (`tf_delay 0.05`) |
| `/odom` | `nav_msgs/Odometry` | Dev 4 (controller velocity), RTAB-Map | gated copy of `/wheel/odom` |
| `/ugv/pose_valid` | `std_msgs/Bool` | **Dev 5** level-3 hold | 20 Hz always (heartbeat); `false` at startup and on any failure; `true` only after `recover_hold_s` clean |
| `/ugv/localization_status` | `std_msgs/String` | humans / eval | comma-separated reasons (`tf_stale,not_localized,…`) or `valid`; on change |
| `/rtabmap/info` | `rtabmap_msgs/Info` | eval | RTAB-Map native |
| `rtabmap.db` | file | Dev 2 localize mode | `~/.ros/ugv/rtabmap.db` by default |

**Not published:** `/map` occupancy grid (mono has no depth), `/cmd_vel*`, anything from the perception mask.

## Requests to other devs

**Dev 5 (platform/sim)**
1. Confirm camera / wheel-odom / ground-truth topic names and optical `frame_id`.
2. Diff-drive: publish `/wheel/odom` with covariance, `publish_odom_tf=false` (Gazebo `DiffDrive` plugin: do not bridge its TF; motor driver: don't broadcast).
3. Sim camera: textured world (feature-rich ground, walls, objects) — mono SLAM finds nothing on flat untextured planes.
4. Sim ground-truth pose topic for drift benchmarks.
5. Safety mux: treat missing `/ugv/pose_valid` messages (> watchdog timeout) the same as `false`.

**Dev 3 (costmaps)**
1. No `/map` occupancy from Dev 2; don't configure a static map layer against it.
2. Use `map` (global) and `odom` (local) frames from Dev 2's TF.

**Dev 4 (planning)**
1. `/odom` (gated) is the odometry topic for the controller.
2. Goals in `map` frame; valid only while `/ugv/pose_valid` is true (Dev 5 enforces).
