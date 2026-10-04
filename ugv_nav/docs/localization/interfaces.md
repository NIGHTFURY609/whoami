# Dev 2 interface contract

Topic names marked **TBD** are proposals until the owning dev confirms; every one is a launch argument.
QoS: Dev 2 subscribes to camera / CameraInfo / depth **best-effort**, so it connects to reliable or best-effort
publishers alike (verified end-to-end with a best-effort camera, `test/test_ros_stack.py`).

## Inputs (Dev 2 consumes)

| Topic / TF | Type | From | Rule |
|---|---|---|---|
| `/camera/image_raw` (**TBD**) | `sensor_msgs/Image` | Dev 5 | stamp = exposure time; `frame_id` = optical frame (`camera_optical_frame`) |
| `/camera/camera_info` (**TBD**) | `sensor_msgs/CameraInfo` | Dev 5 | **one per image, identical stamp** + frame (rgbd_sync pairs RGB + depth + CameraInfo by exact stamp; a latched-once CameraInfo never pairs — Dev 1's transient-local subscriber accepts either); real K (zero K → `camera_info_invalid`) |
| `/perception/depth/image` | `sensor_msgs/Image` 32FC1 m | **Dev 1** (DA3) | primary — see **Depth input** below |
| `/perception/depth_cloud` | `sensor_msgs/PointCloud2` | **Dev 1** (DA3) | fallback (`depth_input:=cloud`) |
| `/wheel/odom` (**TBD**) | `nav_msgs/Odometry` | Dev 5 diff-drive (sim plugin / motor driver) | `frame_id=odom`, `child_frame_id=base_link`, pose covariance filled, ≥ 15 Hz, **no TF**. **Not needed by default** (`odom_source:=visual`); only for `auto` / `wheel` |
| `/tf_static` `base_link->camera_link->camera_optical_frame` | TF | Dev 5 URDF / robot_state_publisher | camera extrinsics (rgbd_odometry + RTAB-Map need them) |
| `/clock` | `rosgraph_msgs/Clock` | Dev 5 sim / `ros2 bag play --clock` | `profile:=sim|bag` → `use_sim_time`. Never recorded into eval bags |
| `/ground_truth/odom` (**TBD**, eval only) | `nav_msgs/Odometry` | Dev 5 sim | drift benchmark only; never used by the product path |
| `/camera/depth/image_raw` (**TBD**, eval/bring-up only) | `sensor_msgs/Image` 32FC1 | Dev 5 sim depth camera | same pose, size and `frame_id` as the RGB camera; DA3 accuracy benchmark + bring-up before DA3 is live |

## Depth input (Dev 1 — `af7ebbf` "Publish a metric depth Image for Dev 2", branch `dev1-turing-perception`)

**Primary (launch default `depth_input:=image`):** `/perception/depth/image`. Checked against Dev 1's code 2026-09-29,
and his real `depth_to_image()` output passes Dev 2's depth gate (stamp preserved to the ns):

| Field | Required | Dev 1 code |
|---|---|---|
| Type / encoding | `sensor_msgs/Image`, `32FC1`, meters | ✓ `node/cloud.py:depth_to_image` |
| Holes / sky | `NaN` | ✓ `hole_safe_resize` output kept as-is |
| Size | camera `width × height` (same as the RGB) | ✓ resized to the camera size before publishing |
| `header.stamp` | the RGB image's stamp (rgbd_sync pairs RGB + depth + CameraInfo by exact stamp) | ✓ `frame.stamp_ns` |
| `header.frame_id` | the RGB image's optical frame (= `CameraInfo.frame_id`) | ✓ `frame.frame_id` |
| QoS | reliable, depth 10 (Dev 2 subscribes best-effort: compatible) | ✓ |
| Enabled | only when `turing/weights/da3metric-large.xml` exists | **weights not in the repo yet** |
| Rate | one depth image per published mask (same DA3 infer as the cloud) | latency / fps not measurable yet (static test images) |

**Fallback (`depth_input:=cloud`):** `/perception/depth_cloud` (PointCloud2) → `rtabmap_util/pointcloud_to_depthimage`
(`config/cloud_to_depth.yaml`). Exact only while the cloud is back-projected with the raw K at camera resolution.

Dev 2 checks the depth image at runtime (`depth/gate.py`): wrong encoding / frame / size, future stamp or coverage below
`min_depth_coverage` → `/ugv/pose_valid=false` (`depth_invalid`); nothing for `depth_max_age_s` → `depth_stale`.
Sim ground-truth depth camera for bring-up / DA3 benchmark: `depth_topic:=<its topic>`.

## Outputs (Dev 2 publishes)

| Topic / TF | Type | To | Rule |
|---|---|---|---|
| TF `odom->base_link` | TF | Dev 3, 4, 5 | `odom_selector` only; stamp = selected odometry stamp; continuous across source switches |
| TF `map->odom` | TF | Dev 3, 4, 5 | RTAB-Map, 20 Hz (`tf_delay 0.05`) |
| `/odom` | `nav_msgs/Odometry` | Dev 4 (controller velocity), RTAB-Map | selected source (wheel or visual), re-anchored; twist passed through |
| `/ugv/localization/odom_source` | `std_msgs/String` | humans / eval / pose_validity | `wheel` or `visual`; latched, on change |
| `/map` | `nav_msgs/OccupancyGrid` | Dev 3 (**optional**) | RTAB-Map grid from DA3 depth, 5 cm, range ≤ 5 m. Quality = DA3 quality (SENSOR_HONESTY.md) |
| `/ugv/pose_valid` | `std_msgs/Bool` | **Dev 5** level-3 hold | 20 Hz always (heartbeat); `false` at startup and on any failure; `true` only after `recover_hold_s` clean |
| `/ugv/localization_status` | `std_msgs/String` | humans / eval | comma-separated reasons (`tf_stale,depth_stale,…`) or `valid`; on change |
| `/ugv/localization/distance_travelled` | `std_msgs/Float64` | humans / eval / Dev 4 testbench | metres driven along `/odom` since start or reset, 5 Hz always. An **odometry estimate**, not surveyed: with no wheel sensor it is visual odometry, so DA3 scale bias = distance bias; motion while VO is lost is not counted; jitter < 5 cm and jumps > 3 m/s are ignored (`config/distance.yaml`) |
| `/ugv/localization/distance_basis` | `std_msgs/String` | whoever reads the distance | what the total is made of: `none`, `visual_odometry_estimate`, `wheel_odometry`, `odometry_estimate` (mixed / unattributed); latched, on change |
| `/ugv/localization/reset_distance` | `std_srvs/Empty` (service) | Dev 4 / operator | zero the total, e.g. at each new goal |
| `/ugv/map/stats` | `std_msgs/String` (one JSON object) | web viewer (gateway: the `stats` of the map status) | `map_stats` node, ~1 Hz, reliable + transient local (a late subscriber gets the latest). Scalars only: `keyframes` (int, graph poses with id > 0; tag landmarks, negative ids, are not keyframes), `loop_closures` (int, distinct unordered node pairs among the latest `/rtabmap/mapGraph` links of type global, local-space or user closure; neighbour, local-time, merged, virtual, prior, landmark and gravity links are excluded, so a parked robot on a never-repeating scene reads 0). It counts rtabmap's closure constraints, the same thing rtabmap reports, not "times the robot returned to a known place": while driving, rtabmap links almost every new node to an older, non-adjacent node with an overlapping view, so it rises roughly with the node count during normal driving (0 to 92 in 60 s with no revisit on a live graph) and does not grow while parked. `path_length_m` (float or null, x/y polyline through the graph poses in id order), `db_bytes` (int or null; null unless the database path is a regular file), `last_update_age_s` (float or null; seconds since the graph's ids or poses changed; null before the first graph and always null in `localize` mode because the map is read-only), `mode` (`mapping` or `localize`), `calibration_placeholder` (bool, the `placeholder` flag of the `calibration_file` launch argument, false when none). A non-finite float is null. Reads only `/rtabmap/mapGraph`, never `cloud_map`, `mapData` or `/rtabmap/info` (a subscriber on `cloud_map` or `mapData` makes RTAB-Map assemble and send the whole map every step) |
| `/rtabmap/info` | `rtabmap_msgs/Info` | eval | RTAB-Map native |
| `/rtabmap/cloud_map` | `sensor_msgs/PointCloud2` | none in this repo: RViz or tools (the web viewer dropped it on 2026-10-03, mindmap D27) | the whole 3D map, coloured, from `map_assembler`, once per SLAM step while subscribed. See **3D map outputs** |
| `/rtabmap/mapPath` | `nav_msgs/Path` | web viewer (gateway, on demand) | graph poses in `map`. See **3D map outputs** |
| `/rtabmap/mapData` | `rtabmap_msgs/MapData` | future elevation mapper (not built; Task 12 deferred) | each new graph node's depth, camera info and camera pose, **once**. See **`/rtabmap/mapData` consumer contract** |
| `rtabmap.db` | file | Dev 2 localize mode | `~/.ros/ugv/rtabmap.db` by default |

**Not published:** `/cmd_vel*`, anything from the perception mask.

## 3D map outputs (Task 8; rtabmap_ros 0.23.7, measured on the real stack with synthetic sensors)

`mapPath`, `mapGraph` and `mapData` come from the `rtabmap` node (namespace `/rtabmap`, `config/rtabmap_rgbd.yaml`:
`Grid/3D true`, node params `cloud_output_voxelized`, `cloud_subtract_filtering`, `map_always_update`, `latch` pinned).
**`cloud_map` comes from `rtabmap_util/map_assembler`** (`/rtabmap/assembler/map_assembler`, same YAML), which builds it
in its own process from `mapData` (Task 8 review I1: rtabmap assembles its own copy inside the SLAM step, and a viewer
opened mid-mission made that one step assemble the whole map). rtabmap's own copy is remapped to
`/rtabmap/slam/cloud_map`: **never subscribe it** (nor rtabmap's other map clouds, `/rtabmap/cloud_obstacles`,
`/rtabmap/octomap_*`, ...: any subscriber makes the SLAM step assemble them). `map_assembler` subscribes `mapData` all
the time, so rtabmap now sends `mapData` every SLAM step; `cloud_map` and `mapPath` are still built only while someone
subscribes them.

| Topic | Reliability | History | Durability | Subscribe with |
|---|---|---|---|---|
| `/rtabmap/cloud_map` | RELIABLE | KEEP_LAST 1 | TRANSIENT_LOCAL | RELIABLE + TRANSIENT_LOCAL (a late subscriber gets the last cloud) |
| `/rtabmap/mapPath` | RELIABLE | KEEP_LAST 1 | VOLATILE | RELIABLE + VOLATILE |
| `/rtabmap/mapGraph` | RELIABLE | KEEP_LAST 1 | TRANSIENT_LOCAL | RELIABLE + TRANSIENT_LOCAL (`map_stats` reads it; cheap) |
| `/rtabmap/mapData` | RELIABLE | KEEP_LAST 1 | VOLATILE | RELIABLE + VOLATILE, **deep queue** (the tests use 100): the publisher keeps 1, a busy reader loses messages |

- **`cloud_map`**: frame `map`, fields `x y z rgb` (FLOAT32 each, `point_step` 16), `height` 1, voxelised at `Grid/CellSize`
  (5 cm). It is the **whole map, republished every SLAM step** (about 2 Hz): 57k points (0.9 MB) after 190 s of motion,
  growing with the map, so a consumer needs a point budget. **Clipped at about 1 m above `base_link`** by
  `Grid/MaxObstacleHeight: "1.0"` (it filters the 3D local maps the cloud is made of): upper walls, trees and overhangs
  are missing from it. Owner decision pending; a future elevation mapper would not be affected (it would read `mapData` depth).
  **Late attach** (the gateway subscribes only while a viewer is open): `map_assembler` builds the whole cloud when the
  first subscriber attaches (1.4-1.7 s at 270-285 nodes, about 5 ms per node, growing with the map; the cloud arrives
  1.4-2.3 s after the attach) and adds only the new nodes after that. `map_cleanup: false` (assembler only) keeps its
  cache when the viewer closes, so a re-attach costs about 0.2 s and the cloud arrives 0.3-0.9 s later. None of this
  runs in the SLAM step any more: the largest `/rtabmap/info` gap in the 10 s after an attach was 0.73 s over 20 attaches,
  the same as with no viewer (docs/mapping/baseline.md "Task 8 fix round 1"). Limits: the assembler's `mapData` subscription keeps 1 message
  (hard-coded in rtabmap_util 0.23.7), so the messages that arrive while it assembles are dropped and those nodes (1-2
  at the first attach, 1-2 at start-up) stay missing from the cloud for the run. The cloud is for display only; the
  a future elevation mapper would read `mapData` itself. It also holds every node's data (about 0.62 MB per node at 640x480) plus,
  once a viewer has been opened, the grid cache (about 1.9 MB per node in all), and gives none of it back: 1.2 GB at about 530
  nodes in the synthetic 640x480 run, next to rtabmap's 1.4 GB (0.71 GB + 1.3 MB per node). `map_cleanup: true` was measured and
  does not lower the peak with a viewer open (docs/mapping/baseline.md "Final review I2"). It runs only with `map_assembler:=true`
  (launch argument, default false, mindmap D24): otherwise nothing publishes `/rtabmap/cloud_map` at all; mission-length guidance in
  docs/mapping/README.md "Memory and mission length".
- **`mapPath`**: frame `map`, one pose per graph node (optimised; it changes on loop closure).

### `/rtabmap/mapData` consumer contract (what a future elevation node would build against)

> **Elevation (not built).** The owner deferred the elevation map (plan Tasks 10-12). No node publishes `/ugv/elevation/cloud` or `/ugv/elevation/obstacles`, and the gateway and the web viewer have no elevation layer until such a node lands (mindmap D16). `/rtabmap/mapData` therefore has no consumer today except `map_assembler`; the contract below is for a future elevation mapper.

Which points a test asserts and which were measured once (`test/test_ros_stack.py`): `test_x4_rtabmap_3d_map_outputs` (`_map_data_problems`, `_depth_png_problem`) asserts points 2-4 (first entry per id with data, every id of the final graph delivered, float32-in-PNG depth at camera size) and that `left_camera_info` and `local_transform` are non-empty; unit tests `test_x4a_*` cover those checks. Points 1 and 5-7 (one node per message, pose z = 0, JPEG left image, empty camera-info `frame_id`, the `local_transform` values) were measured once and are not asserted.

1. **One node per message.** `graph` is the whole graph (`poses_id`, optimised `poses` in `map`; take node poses from the
   latest message's graph, not from the message that delivered the node: loop closures move them). `nodes` holds one
   entry: normally the node this step added (its id is the last of `graph.poses_id`). `node.stamp` is the image stamp
   (float64 seconds); `node.pose` is `base_link` in `map` at that stamp, with z = 0 and roll = pitch = 0 (`Reg/Force3DoF`).
2. **Images arrive once.** A node's images are only in the message that adds it. A processed frame that adds no node
   (the robot has not moved `RGBD/LinearUpdate` 0.1 m / `RGBD/AngularUpdate` 0.1 rad) re-sends the newest node: same id
   and stamp, `graph.poses_id` unchanged, **`left_compressed` and `right_compressed` empty**. Deduplicate by id and skip
   entries with an empty `right_compressed`.
3. **VOLATILE, so subscribe before mapping starts.** A subscriber that connects late, or loses a message, never gets
   those nodes' images from this topic (no replay). Started before mapping, every id in the final `graph.poses_id` was
   delivered once with its data (the x4 check).
4. **`data.right_compressed` is the depth, a PNG of float32 metres** (not RVL, not 16-bit millimetres: `Mem/SaveDepth16Format`
   is false, rtabmap warns once at start). Decode: `cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)` gives an `(h, w, 4)` uint8
   array; `np.ascontiguousarray(im).view('<f4')[..., 0]` is the `(h, w)` depth in metres at the camera size; NaN or 0 =
   no depth (the sky band is NaN).
5. **`data.left_compressed`** is the colour image as a JPEG (decodes to BGR with OpenCV). The raw `data.left` /
   `data.right` images are empty.
6. **`data.left_camera_info`**: one entry, `K` / `width` / `height` of the camera (e.g. K = [260, 0, 160, 0, 260, 120,
   0, 0, 1] at 320x240), but **`header.frame_id` is empty**: do not look the camera up in TF by that name.
7. **`data.local_transform`**: one entry, the **`base_link` -> camera optical frame** transform (translation + quaternion,
   `geometry_msgs/Transform`). Camera pose in `map` = node pose (from the latest graph) * `local_transform`. Measured:
   translation (0, 0, 0.5), rotation (x, y, z, w) (0.5, -0.5, 0.5, -0.5), which is the harness's static
   `base_link -> camera_optical` with the quaternion's sign flipped (the same rotation).

## TF for Dev 3 (answers 2026-09-29; measured on the real stack with synthetic sensors unless marked)

| Question | Answer |
|---|---|
| TF tree / frame names | `map → odom → base_link` (Dev 2) → `camera_link → camera_optical_frame` (Dev 5 URDF, static). `map`, `odom`, `base_link` are fixed in Dev 2 configs. Camera frame names are Dev 5's (proposal above) — **read `header.frame_id` from the mask / depth message, don't hard-code it**. `odom_visual` is a label, never a TF frame. |
| `odom → base_link` rate | = the selected odometry's rate, stamped with the odometry message stamp (never "now"). `auto`/`wheel`: Dev 5 wheel odom rate (contract ≥ 15 Hz). `visual`: the DA3 depth rate (not measured; may be < 15 Hz). Measured: arrives ~6 ms after its stamp. Continuous across source switches (no jump). |
| `map → odom` rate / behavior | RTAB-Map, **20 Hz** (`tf_delay 0.05`), measured 20.1 Hz. Stamped **~+100 ms in the future** (`tf_tolerance 0.1`) so lookups at "now" succeed. Its *value* changes only when RTAB-Map processes a frame (`Rtabmap/DetectionRate 2` Hz) and jumps on loop closure / relocalization (Dev 2 holds `/ugv/pose_valid=false` for 1 s after a jump > 1 m / 0.35 rad). Identity until the first frame; in `localize` identity until relocalized (pose invalid meanwhile). |
| TF lookup timeout | Look up `camera_optical_frame → costmap frame` **at the mask stamp**, never "latest". Wheel odom at ≥ 15 Hz: the TF for a stamp exists ≤ ~70 ms after capture, and masks arrive later than that (segmentation latency), so the lookup normally succeeds immediately — use **0.1 s** timeout. `odom_source:=visual`: TF follows DA3 latency — use up to **0.5 s** (= Dev 5 watchdog), drop the mask on timeout. Keep the TF buffer ≥ 10 s (default). Nav2 `transform_tolerance` 0.3–0.5 s. Revisit once latency is measured. |
| Stamp / latency perception ↔ TF | Every perception message (mask, depth image, cloud) carries the **RGB capture stamp**. `odom → base_link` carries wheel (or visual) odometry stamps — tf2 interpolates to the mask stamp. `map → odom` is future-stamped. Absolute latency (capture → mask, capture → depth) is **not measurable yet** (static test images); measure with a sim/bag stream. |
| `base_link` vs `base_footprint` | Dev 2 uses **`base_link`** as the robot frame (RTAB-Map `frame_id`, odom `child_frame_id`, 3-DoF ground robot); **no `base_footprint`** in the chain today. Where `base_link` sits (ground level or axle height) is Dev 5's URDF — recommend **`base_link` at ground level** (Grid heights are relative to it). If Dev 5 adds `base_footprint`, Dev 2 switches its robot frame to it (config-only). |
| `rtabmap_util` dependency | Yes — `package.xml` `exec_depend rtabmap_util` (added 2026-09-29; needed for `depth_input:=cloud`). Also `rtabmap_slam`, `rtabmap_sync`, `rtabmap_odom`, `rtabmap_msgs`. |

## Requests to other devs

**Dev 1 (perception)**
1. Merge `af7ebbf` (depth image) to `main`. Keep its stamp / frame / size = the RGB image's.
2. Fetch + export the DA3 weights (`fetch_da3metric_large.sh`, `export_da3metric_openvino.py`): without the IR nothing is published.
3. DA3 latency / fps once a camera stream exists (sim or bag; static test images can't measure it) — sets
   `sync_queue_size`, `depth_max_age_s`, and whether visual odom meets 15 Hz.
4. Optional: publish depth even when a mask is skipped (today depth only follows a published mask).

**Dev 5 (platform/sim)**
1. Confirm camera / wheel-odom / ground-truth topic names and optical `frame_id`. Publish CameraInfo **with every image, same stamp**.
2. Diff-drive: publish `/wheel/odom` with covariance, `publish_odom_tf=false` (Gazebo `DiffDrive` plugin: do not bridge its TF; motor driver: don't broadcast).
3. Sim: a **ground-truth depth camera** co-located with the RGB camera (same intrinsics / size / frame) — DA3 benchmark and bring-up.
4. Sim world: textured (feature-rich ground, walls, objects) — visual odometry and loop closure need features.
5. Sim ground-truth pose topic for drift benchmarks.
6. Safety mux: treat missing `/ugv/pose_valid` messages (> watchdog timeout) the same as `false`.

**Dev 3 (costmaps)**
1. `/map` from Dev 2 is **optional** and only as good as DA3 depth; don't make the global costmap depend on it.
2. Use `map` (global) and `odom` (local) frames from Dev 2's TF.
3. In `odom_source:=visual` the `odom->base_link` rate is the DA3 depth rate (CONFLICTS.md C9).

**Dev 4 (planning)**
1. `/odom` (selected, re-anchored) is the odometry topic for the controller.
2. Goals in `map` frame; valid only while `/ugv/pose_valid` is true (Dev 5 enforces).
