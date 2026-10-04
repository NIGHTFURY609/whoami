# 3D mapping and the web map view

What was built on branch `mapping-3d` (plan: `docs/superpowers/plans/2026-10-02-3d-mapping-elevation.md`), how to run it
and what it can and cannot tell you. Where the plan and the code disagree, this document follows the code.

**Status.** The map, live cloud, trajectory and cost grid layers, the image panels and the statistics widget are built and
tested on synthetic data and on the real stack with synthetic sensors. **Nothing has run on the UGV yet** (see "Pending
owner runs"). The **elevation map (plan Tasks 10-12) is not built yet**: the owner deferred it, and its gateway endpoint,
decoder and viewer layer were removed until a producer lands with them (mindmap D16). Nothing here feeds Nav2: the map is mapping and display only (mindmap D9, architecture §9 unchanged).

## Data flow

```
phone camera -> tunnel -> ugv_bringup camera driver -- /camera/image_raw + /camera/camera_info --> Dev 1 perception
                                   |                                                              (DA3 depth, SegFormer mask)
                                   +-- /image_raw/compressed (camera view, read by the UI)          /perception/depth/image (32FC1 m)
                                                                                                          |
        rgbd_sync (RGB + depth, exact stamps) -> rgbd_odometry -> odom_selector -> /odom                  |
                                         |                                                                |
                                         v                                                                |
        RTAB-Map (rgbd, Grid/3D true): TF map->odom, /rtabmap/info, /rtabmap/mapPath, /rtabmap/mapGraph, /rtabmap/mapData
                                         |                                   |
                                         |        map_assembler (own process, opt-in map_assembler:=true):
                                         |        /rtabmap/mapData -> /rtabmap/cloud_map
                                         |
        map_stats node (reads /rtabmap/mapGraph only) -> /ugv/map/stats
                                         |
                                         v
        ugv_api gateway (on-demand subscriptions, TF poll map->base_link) -- GET /api/v1/map/... binary v1 --> web UI map view (three.js)
```

RTAB-Map is the only fusion engine. The gateway and UI add no mapping; they reshape what RTAB-Map (through its
`map_assembler`), perception and Nav2's global costmap already publish. `rtabmap` builds `mapPath` only while something
subscribes; with `map_assembler:=true`, `mapData` is built every SLAM step because `map_assembler` subscribes it for the
whole run. `map_assembler` is off by default (D24): the 3D map view's cloud layer needs `map_assembler:=true`. It
publishes `cloud_map` only while something subscribes. The gateway subscribes to the heavy topics (those marked "on
demand" below) only while a client keeps calling `GET /api/v1/map`.

## Topics

| Topic | Type | Publisher | Consumer | Notes |
|---|---|---|---|---|
| `/rtabmap/cloud_map` | `sensor_msgs/PointCloud2` | `map_assembler` (`/rtabmap/assembler/map_assembler`) | gateway (on demand) | whole coloured 3D map, voxelised at 5 cm, built from `mapData` outside the SLAM step and republished on every `mapData` message while subscribed. Reliable, transient local. Absent with `map_assembler:=false`. rtabmap's own copy is on `/rtabmap/slam/cloud_map`: never subscribe it (it would make the SLAM step assemble the map) |
| `/rtabmap/mapPath` | `nav_msgs/Path` | `rtabmap` | gateway (on demand) | optimised graph poses in `map` |
| `/rtabmap/mapGraph` | `rtabmap_msgs/MapGraph` | `rtabmap` | `map_stats` | cheap, transient local |
| `/rtabmap/mapData` | `rtabmap_msgs/MapData` | `rtabmap` | `map_assembler`, always (while it runs) | every SLAM step: the new node's data and the graph. Also the input a future elevation mapper would use (not built); contract in `ugv_nav/docs/localization/interfaces.md` |
| `/ugv/map/stats` | `std_msgs/String` (JSON) | `map_stats` (`ugv_localization`) | gateway, always on | `keyframes`, `loop_closures` (distinct closure-type graph links; rtabmap's closure constraints, not "returns to a known place": it rises roughly with the node count while driving, even with no revisit, and does not grow while parked), `path_length_m`, `db_bytes`, `last_update_age_s` (null before the first graph and in `localize` mode), `mode`, `calibration_placeholder` |
| `/perception/depth/image` | `sensor_msgs/Image` 32FC1 | Dev 1 perception | RTAB-Map, UI camera view | published after each published mask, on the same decoded frame (D14 deferred, D21). A depth failure publishes nothing for that frame, is counted and logged by perception, and Dev 2 holds on `depth_stale` (D23) |
| `/perception/depth_cloud` | `sensor_msgs/PointCloud2` | Dev 1 perception | gateway (on demand), Nav2 VoxelLayer | the live cloud layer; the gateway does not back-project depth itself |
| `/global_costmap/costmap` | `nav_msgs/OccupancyGrid` | Nav2 | gateway (on demand) | the cost grid layer, display only |
| `/image_raw/compressed` | `sensor_msgs/CompressedImage` | camera driver | UI camera view (read only) | also drawn in the map view's camera panel; the gateway does not carry it |

Topic names are gateway parameters under `map:` in `ugv_nav/ugv_api/config/api.yaml`.

## Gateway endpoints (`ugv_api`, read only)

| Endpoint | Body | Notes |
|---|---|---|
| `GET /api/v1/map` | `MapStatus` JSON: `epoch`, `seq` for every layer, flat `stats` | also the demand heartbeat: only an explicit GET keeps the heavy subscriptions alive (`idle_timeout_s`, default 10 s) |
| `GET /api/v1/map/pose` | `Pose` JSON, `map -> base_link` from TF | `available: false` and null fields until a transform is seen |
| `GET /api/v1/map/{cloud,trajectory,grid,live}` | binary format v1 (below) | 503 `application/problem+json` until the layer has data |
| `GET /api/v1/telemetry/stream` | SSE | gains the `map` (MapStatus) and `pose` events |

`epoch` is random per gateway process; `seq[layer]` counts changes (0 = nothing yet). A client compares `epoch:seq`, not
`seq` alone, and refetches every layer when the epoch changes. `stats` passes the ROS stats keys through unchanged and adds
the gateway's own input health: `map_inputs_alive`, `map_rejects`, `map_restarts`, `map_last_reject`. JSON contract:
`docs/mapping/map-contract.md`.

### Binary format v1

Little endian; a 24-byte prelude (`magic`, `format = 1`, `header_bytes`, `epoch`, `seq`, `stamp_s`), then a layer header and
body. Decoders reject an unknown format and any length that is not exact.

| Layer | Magic | Body |
|---|---|---|
| cloud, live | `UGVC` | `f32 xyz`, then `u8 rgb` if flag bit 0; at most `cloud_point_budget` points (default 500000) picked by spatial hash at `cloud_spacing_m` (0.05). `live` is Dev 1's `/perception/depth_cloud` (already back-projected by perception, full resolution), range-gated on optical depth (0.3-8 m), transformed to `map` and cut to `live_point_budget` (20000) points the same way |
| trajectory | `UGVT` | `f32 x y z qx qy qz qw` per pose |
| grid | `UGVG` | `i8` cells (-1 unknown, 0..100); size, resolution, origin and yaw in the header |

Authoritative layout and the golden files the Python and TypeScript tests share:
`docs/mapping/map-contract.md`, `ugv_nav/ugv_api/test/fixtures/map/*.bin`.

## Run mapping and view it

Full stack with the live camera (arguments as in `ugv_nav/ugv_bringup/README.md`):

```
ros2 launch ugv_bringup bringup.launch.py profile:=live_cam mode:=mapping \
    calibration_file:=<phone yaml> device:=<stream URL> transport_latency_s:=<measured s> \
    allow_placeholder_calibration:=true \
    camera_x:=.. camera_y:=.. camera_z:=.. camera_pitch_deg:=.. perception_src:=<repo>/turing/src
cd ui && npm install && npm run dev      # http://localhost:5173, proxies /api to the gateway (UGV_API_URL)
```

`allow_placeholder_calibration:=true` is needed only while `phone_640x480.yaml` is still the placeholder: without it the
camera driver refuses the file and publishes nothing (`ugv_nav/ugv_bringup/README.md`). Never for an autonomous run.

Open the console and pick **map** in the top bar. The view shows the accumulated cloud (camera colours), the live depth scan
(height colours), the trajectory, the Nav2 cost grid halo, the robot pose, the depth and camera image panels (the camera view's feed) and the
statistics widget. Layer buttons: cloud, live, path, cost, img; the choice is kept in
`localStorage`. Banners: `NO MAP YET`, `STALE · map not updating`, `STALE · telemetry lost`, `MAP INPUTS STOPPED` (the
gateway's input thread is gone).

Drive the loop, then stop the stack: the database is saved at `~/.ros/ugv/rtabmap.db` (override with `database_path`). To
localize on it, relaunch with `mode:=localize`. The localization stack alone:
`ros2 launch ugv_localization localization.launch.py mode:=mapping|localize` (`fresh_db:=true` starts an empty database).
Map assembly runs outside the SLAM step: `rtabmap_util/map_assembler` (`/rtabmap/assembler/map_assembler`, started by `localization.launch.py`) builds the whole cloud from `/rtabmap/mapData` and publishes `/rtabmap/cloud_map`. rtabmap's own cloud goes to `/rtabmap/slam/cloud_map`, which nothing should subscribe, so opening the viewer does not load the SLAM loop. Known limits (numbers in `docs/mapping/baseline.md`, section "Task 8 fix round 1"): the first open of the map view takes about 1.4-2.9 s for the first cloud (about 5 ms per node); a few nodes (1-3 per run) can be missing from the viewer cloud (map_assembler's `mapData` queue depth is 1, upstream); map_assembler memory grows with the map and is never given back (next section). It runs only with `map_assembler:=true` (on `localization.launch.py` or `bringup.launch.py`; off by default, D24). Without it there is no `/rtabmap/cloud_map` at all, the viewer's cloud layer stays at "no map yet", and everything else (SLAM, TF, pose validity, trajectory, live scan, stats) is unchanged.

### Memory and mission length

Measured 2026-10-02 on the live profile (`timing:=laptop`, 640x480, reliable camera) with the synthetic harness
(`test_m1_measure_late_cloud_map_attach`, robot sliding at 0.3 m/s, about 1.9 graph nodes per second, viewer attached 15 s and
detached 20 s, 5 cycles, up to about 550 nodes; numbers in `docs/mapping/baseline.md`, section "Final review I2"). Resident
memory of the two mapping processes, by graph node count N (`keyframes` in the statistics widget):

| What runs | rtabmap | map_assembler | Both, about |
|---|---|---|---|
| `map_assembler:=false` | 0.71 GB + 1.3 MB/node | - | 0.7 GB + 1.3 MB/node |
| assembler, viewer never opened | 0.71 GB + 1.3 MB/node | 0.2 GB + 0.62 MB/node | 0.9 GB + 1.9 MB/node |
| assembler, viewer opened at least once (shipped, `map_cleanup: false`) | 0.71 GB + 1.3 MB/node | 0.24 GB + 1.9 MB/node | 0.95 GB + 3.2 MB/node |

The assembler's grid cache is built only while a viewer is attached and with `map_cleanup: false` (shipped) it is kept when the
viewer closes. `map_cleanup: true` was measured too and is not used: its peak while a viewer is open is the same (0.18 GB + 2.1
MB/node), it frees only part of the cache when the viewer closes (1.2 MB/node stays), a re-open then takes 1.7-3.0 s instead of
0.2-0.4 s for the first cloud (growing about 5 ms per node), and it drops about 7 more `mapData` messages per run (nodes missing
from the viewer cloud). The figure that limits a mission is the peak with the viewer open, because the operator can open it at
any time, and that is the same either way. SLAM is unaffected by either setting (map_assembler is its own process; rtabmap's map
work stayed at 3-5 ms per step after every attach).

**Mission-length guidance for the 15.6 GB laptop (WSL `memory=12GB`, swap 8 GB).** Perception (DA3 Metric Large and
SegFormer-B5 in PyTorch), Nav2, the gateway and the rest of the stack share the same 12 GB and were not measured here; keep
about 6 GB for them, which leaves about 6 GB for rtabmap and map_assembler:

- **With the map view (`map_assembler:=true`):** up to about 1500 graph nodes (about 5.7 GB for the two). RTAB-Map adds at most 2 nodes per
  second (`Rtabmap/DetectionRate`) and only while the robot moves (`RGBD/LinearUpdate` 0.1 m or `RGBD/AngularUpdate` 0.1 rad since the last node), so
  that is about **12 minutes of continuous driving** at the worst-case 2 nodes/s; slower driving or stops last longer. Watch
  `keyframes` in the statistics widget.
- **Longer missions (default, `map_assembler:=false`):** up to about 4000 nodes (about 5.9 GB), about **30 minutes of continuous driving**
  at 2 nodes/s. The map is still built and saved in the database; view it afterwards from `mode:=localize`.
- Past that the processes reach swap and slow down, and an out-of-memory kill of rtabmap ends the mission safely (SLAM stale
  -> pose invalid -> safety hold) but ends it.

These per-node figures come from a synthetic textured plane 3 m away. A real outdoor scene fills the 5 m fusion range with
more points per node, so the real cost per node may be higher: re-measure (`ps -o rss,comm -C rtabmap,map_assembler` in the
container, with `keyframes` from the widget) during the owner's lit run before relying on the limits above.

Tests: `python -m pytest ugv_nav/ugv_api/test -k map` (codec, store, endpoints); `cd ui && npm run lint && npm test &&
npm run build`; in the container `colcon test --packages-select ugv_localization ugv_bringup ugv_api`.

## Honest limits

- **Monocular scale wobble.** Depth is DA3 pseudo-depth from one camera. Its scale varies from frame to frame and with
  scene content (see `docs/mapping/baseline.md`; mono is the architecture's minimum tier). Loop closure corrects pose, not a
  depth scale that differed between keyframes: a wall seen twice can sit at two distances.
- **5 m fusion range.** The 3D map fuses depth from 0.3 to 5.0 m (`Grid/RangeMin`, `Grid/RangeMax`); far DA3 depth smears.
  The live scan shows up to 8 m, but it is not map.
- **Heights are relative to the driving plane.** `Reg/Force3DoF` is true: pose is x, y, yaw only and `base_link` z = 0 is
  the ground. Pitch and slopes are not tracked, so a ramp can read as a wall or a drop.
- **Clipped about 1 m above the robot.** `Grid/MaxObstacleHeight: "1.0"` also clips `cloud_map` (measured: z max 0.96 m, 1.42 m
  with it off). Upper walls and trees are missing. Owner decision pending.
- **Phone calibration.** `ugv_nav/config/cameras/phone_640x480.yaml` is the phone's own calibration since 2026-10-03
  (RMS 0.35 px, 60 views; its header lists the limits: the board never reached the image corners). Runs made before
  that used the laptop webcam's intrinsics as a flagged placeholder: their scale and projection are wrong, do not
  trust measurements from them.
- **Stamps are arrival time minus `transport_latency_s`** for a network camera, not exposure time (mindmap D10).
- **A mask is held up to about 0.58 s** against the 0.5 s limit (owner decision below).
- **Memory bounds the mission length.** rtabmap and map_assembler grow with every graph node and give nothing back: about
  12 minutes of continuous driving with the map view (`map_assembler:=true`), about 30 without (the default) ("Memory and mission length").
- **Display only.** Not a Nav2 input, not a safety input; elevation into Nav2 needs its own §9 decision.

## Pending owner runs

None of this has happened and no numbers exist for it. The synthetic screenshots in this folder are not UGV evidence.

1. The recorded moving run (a 2-3 minute closed loop: `DEPTH_CLOUD_TOPIC="" GT_DEPTH_TOPIC="" bash ugv_nav/ugv_localization/scripts/record_eval_bag.sh eval_bags/loop1` (`eval_bags/` is the output folder you choose; it is not in the repo), replayed through `bag_eval.launch.py`) and the four go/no-go gate numbers of Task 7: depth image rate >= 5 Hz, `/ugv/pose_valid` true while moving >= 90 %, visual odometry lost < 5 % of frames, closed-loop start-to-end error < 5 % of path length. Record them in `docs/mapping/gate-phase0.md`.
   While the phone is uncalibrated (item 2 not done), the launch that starts the camera driver for this recording needs
   `allow_placeholder_calibration:=true`, otherwise the driver refuses `phone_640x480.yaml` and nothing is recorded. The
   gate numbers from such a run are bring-up numbers only (wrong depth scale).
2. Phone camera: ~~lock focus and exposure, calibrate, replace the placeholder YAML~~ done 2026-10-03 (browser page, `phone_640x480.yaml`); still to do: measure the tunnel latency and set `transport_latency_s` (`PHONE_LATENCY_S` in `run.sh`).
3. A re-measure in a lit scene (the baseline run used a black image, so its odometry and depth-stability rows are invalid) and the tape-measured wall test at 1-5 m (depth error per distance).
4. The closed-loop end-to-end run on the UGV: mapping, save the database, restart in `localize`. Keep a screenshot of the map view and the stats values in this folder.
5. Two open owner decisions: (a) the mask freshness budget, since a mask can be held up to about 0.58 s against the 0.5 s limit; (b) whether to keep `Grid/MaxObstacleHeight 1.0`, which clips the 3D cloud at about 1 m above the robot.

## Files

| File | What |
|---|---|
| `docs/mapping/map-contract.md` | binary format v1 and the MapStatus/Pose JSON contract |
| `docs/mapping/baseline.md` | baseline measurement; before/after numbers for depth rate and odometry QoS |
| `docs/mapping/map-view-synthetic-*.jpg` | the map view on synthetic data (not UGV evidence) |
| `ugv_nav/docs/localization/interfaces.md` | topic contracts, 3D map outputs, `mapData` consumer contract |
| `ugv_nav/ugv_api/config/api.yaml` | gateway `map:` parameters |
| `ui/src/map/`, `ui/src/components/MapView.tsx` | decoders, scheduler, geometry, three.js scene, view |
