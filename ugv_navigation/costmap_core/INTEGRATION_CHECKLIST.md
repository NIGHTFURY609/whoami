# Dev 3 integration checklist — before ROS integration

What Dev 3 (costmaps & spatial geometry) needs from other developers and from the future ROS adapter, before we leave the ROS-independent Windows stage.

Every item is sourced from `PROJECT_CONTEXT.md`, `dev.md`, `architecture.md`, Dev 1's `turing/` docs/code (read-only), or this package's code. Nothing here is a new value or interface.

---

## Status labels

| Label | Meaning |
|---|---|
| **CONFIRMED** | Specified in the project docs and/or present in code. |
| **AVAILABLE BUT NOT USED** | Exists upstream; Dev 3 core does not consume it. |
| **OPTIONAL** | The core works without it. |
| **NOT YET DEFINED** | No source defines the value, format, or owner. |
| **BLOCKED** | Blocked on another team or on ROS integration. |

Source abbreviations: **PC** = `PROJECT_CONTEXT.md`, **ARCH** = `architecture.md`, **DEV** = `dev.md`, **D1** = Dev 1 `turing/`, **README** = `costmap_core/README.md`.

---

## 1. Current Dev 3 outputs

**Internal (exists today, CONFIRMED in code).** `pipeline.run_costmap_pipeline(inputs, inflation_radius=...)`

returns a `CostmapPipelineResult`:

| Field | Content |
|---|---|
| `semantic` | projected semantic costmap |
| `geometry` | geometry costmap, or `None` (no occupancy given) |
| `fused` | semantic after geometry precedence (geometry lethal always wins) |
| `final` | `fused` after inflation — the costmap intended for publishing |
| `footprint` | input footprint (optionally padded), or `None`; **not applied to cells** |

- Arrays: `numpy` `int64`, shape `(grid.height, grid.width)`, read-only.
  Row = y, col = x, origin = lower-left corner of cell (0, 0) (the `nav_msgs/OccupancyGrid` origin convention).
- Cost values (defaults, overridable, following Nav2 `costmap_2d` conventions): free `0`, lethal `254`, unknown / never observed `255`, inflated `1..253`. Semantic hazard cost must equal geometry lethal cost (enforced).
- Inflation decay is a Dev 3 **placeholder** (linear); no project formula exists.
- The result carries **no** frame id or stamp. The grid frame is `inputs.grid.frame_id`. Which stamp the published costmap carries is **NOT YET DEFINED**.

**Future ROS/Nav2 output (BLOCKED, not implemented).**

DEV §3 / PC §4: Dev 3 publishes `/global_costmap/costmap` and `/local_costmap/costmap` as `nav_msgs/msg/OccupancyGrid`, consumed by Dev 4.

ARCH §5 / DEV Dev 3 task 1 describe this as a Nav2 costmap with a semantic layer plugin (+ optional VoxelLayer). The mapping from core costs (0–255) to the published representation, and how global vs local costmaps differ (frame, extent, update behaviour), are **NOT YET DEFINED**.

---

## 2. Inputs available from Dev 1

Verified against D1 `interfaces.md`, `node/adapter_node.py`, `node/wire.py`, `port/ids.py`.

| Input | Status | Detail |
|---|---|---|
| Semantic mask | **CONFIRMED** | `/segmentation/mask`, `sensor_msgs/msg/Image`, `mono8`. Mirrored by `contracts.SemanticMaskInput`. |
| Canonical class IDs | **CONFIRMED** | `0 unknown`, `1 traversable`, `2 hazard` (ARCH §8.1, D1 `port/ids.py`). Any other value is rejected by the core. |
| Mask timestamp | **CONFIRMED** | `header.stamp` = image capture time; stamp reuse is illegal (D1 `interfaces.md`). |
| Mask `frame_id` | **CONFIRMED** | Camera optical frame; must equal the source image frame. The actual frame **name** is **NOT YET DEFINED** (PC §6 lists only temporary examples). |
| Mask resolution | **CONFIRMED** (v1) | `scale = 1.0` (mask size = source image size). The core relies on this. |
| Validity | **CONFIRMED**, format provisional | `/segmentation/port_meta`, currently `std_msgs/Float64MultiArray` `[valid, age, scale]` until Dev 1's `PortMeta.msg` is compiled. D1 publishes mask + meta only when its publish decision allows. The core refuses `valid=False`. |
| CameraInfo | **AVAILABLE**, source topic **NOT YET DEFINED** | Driver is Dev 5's (DEV §3). D1 republishes the last received CameraInfo on `/segmentation/camera_info`. Which topic Dev 3 subscribes to is not agreed. |
| Confidence | **AVAILABLE BUT NOT USED** | `/segmentation/confidence`, 32FC1, `[0,1]`. Not in the Dev 3 contract. |
| `/ugv/perception_degraded` | **AVAILABLE BUT NOT USED** | Subscriber is Dev 5 (DEV §3), not Dev 3. |
| Depth (Depth Anything, T08) | **NOT YET DEFINED / not implemented** | See §5. |

**Needs team confirmation:**

- Is the mask computed on a rectified image? D1's node defaults to `/camera/image_raw`, and the core ignores distortion (README §8).
- Which CameraInfo topic Dev 3 should use.

---

## 3. Inputs required from Dev 2

| Item | Status | Source |
|---|---|---|
| TF chain `map → odom → base_link` | **CONFIRMED requirement**, **BLOCKED** (not yet available) | PC §3/§4, DEV §3: `tf2_msgs/msg/TFMessage`, subscribers include Dev 3. |
| TF quality | **CONFIRMED** | DEV §3: continuous tree, jitter < 50 ms, publish rate ≥ 15 Hz. |
| Camera intrinsics YAML (`config/cameras/`) | **Owner not yet defined** (sources conflict) | DEV §3 assigns `config/cameras/` to Dev 2; D1 `interfaces.md` lists `config/cameras/<name>.yaml` as owned by Dev 1. No calibration exists (PC §5). |
| `/ugv/pose_valid` | **AVAILABLE BUT NOT USED** | Subscriber is Dev 5 (DEV §3). |

Dev 3 needs, via TF, the pose of the camera optical frame relative to the costmap frame at the mask stamp (ARCH §8.5: "TF tree connecting camera frame → `base_link` / costmap frame"). The TF lookup API, the lookup timeout and the allowed stamp mismatch are **NOT YET DEFINED**.

---

## 4. Inputs required from Dev 5

| Item | Status | Source |
|---|---|---|
| Camera driver (`Image` + `CameraInfo`) | **CONFIRMED** owner: Dev 5 | PC §3, DEV §3/Dev 5 task 5. Consumers listed are Dev 1 and Dev 2; Dev 3 gets the mask via Dev 1. |
| Camera extrinsics (mount height, pitch, offset) | **BLOCKED** | DEV Dev 5 task 4: URDF/xacro "with … camera extrinsics". No values exist (PC §5). Which node broadcasts `base_link → camera` TF is **NOT YET DEFINED**. |
| Robot footprint | **BLOCKED** | DEV Dev 5 task 4: "primary + secondary footprint YAMLs"; ARCH §13 item 9. No polygon exists. Which footprint Dev 3 uses for inflation/padding is **NOT YET DEFINED**. |
| Safety behaviour | **CONFIRMED**, context only | ARCH §3.1/§12: Dev 5's safety authority zeros `/cmd_vel` on stale perception, invalid pose, TF missing. DEV Dev 5 task 2: watchdog timeouts (0.5 s mask, 0.5 s TF) in `safety_timeouts.yaml`. These are Dev 5 watchdog values, **not** Dev 3 costmap thresholds. |

---

## 5. Geometry / depth side-channel (OPTIONAL)

- **Core side — CONFIRMED.** `contracts.OccupancyInput`: 2D `bool` array already rasterised onto the costmap grid, grid-shaped, same `frame_id` as the grid, with `stamp_ns`. `True` = geometrically occupied → lethal; geometry lethal always wins over semantics (ARCH §9). If omitted, the pipeline skips fusion.
- **Producer — NOT YET DEFINED / not implemented.** ARCH §6/§9: optional Depth Anything → VoxelLayer, a geometry side-channel. D1 lists Depth Anything as T08, **deferred** (D1 `README.md`), and its interface is `DepthFrame` whose `unit` may be `"relative"`, which on its own cannot give metric occupancy.
- DEV Dev 3 task 3 assigns *integration* of this side-channel to Dev 3.

Missing contract information (owner not defined in the sources unless stated):

1. Output topic and message type of the depth/geometry producer.
2. Metric vs relative depth, and how relative depth becomes metric.
3. Frame and stamp of the geometry output. D1 runs YOLOE and Depth Anything on the same `ImageFrame` (D1 `interfaces.md`), and `DepthFrame` carries `stamp_ns`/`frame_id`, but no stamp contract toward Dev 3 is stated.
4. Who converts depth / voxels into a grid-rasterised occupied mask, and with which occupancy threshold.
5. Whether this goes through Nav2's VoxelLayer or through the Dev 3 core's `OccupancyInput` path.

---

## 6. Timing / freshness

**Defined:**

- Mask stamp = image time; no silent reuse of older frames (ARCH §8.4/§8.5).
- A consumer rejects a mask if age > `perception_max_age` (ARCH §8.4). Dev 1's own value is `0.50` s in D1 `config/perception/port.yaml` — a Dev 1 gate.
- Dev 5 watchdog: 0.5 s for mask and TF (DEV Dev 5 task 2).
- TF jitter < 50 ms, rate ≥ 15 Hz (DEV §3).
- Core: `contracts.age_s` / `is_fresh(stamp, now, max_age_s)` exist, with no default.

**NOT YET DEFINED:**

- Dev 3's own `max_age_s` (whether Dev 3 reuses `perception_max_age` is not stated).
- Stamp alignment tolerance between mask, TF pose and occupancy.
- Stale/invalid mask fail-safe: ARCH §8.6 requires "front ROI lethal/max-inflate"; the ROI extent and its owner are not defined. Not implemented in the core.
- Update / publish rate of the costmaps.

---

## 7. Frame requirements

| Frame | Status | Notes |
|---|---|---|
| Camera optical frame | name **NOT YET DEFINED** | Must be identical on mask, CameraInfo and the camera pose (enforced by `CostmapCoreInputs`). PC §6 temporary example: `camera_optical_frame`. |
| Ground / grid (costmap) frame | name **NOT YET DEFINED** | Camera pose's ground frame must equal the grid frame; occupancy must be in the grid frame (enforced). Global vs local costmap frames not specified. PC §6 temporary example: `map`. |
| Robot frame | `base_link` in the TF chain (PC §4, DEV §3) | Footprint is expressed in the robot frame; PC §6 marks `base_link` as a temporary example for configuration. |

**Simplified in the core today (needs real TF/extrinsics later):**

- Camera pose = height + downward pitch only; roll = yaw = 0; camera directly above the ground-frame origin (`projection.CameraGroundGeometry`). A real TF pose with offset, roll or yaw cannot be represented without extending `projection.py`.
- Flat ground: z = 0 plane of the costmap frame.
- Pure pinhole: no distortion.
- Footprint is not related to the grid (no robot → grid transform), so it is not applied to any cell.

---

## 8. Pending ROS/Nav2 integration items

**All items below remain pending until ROS integration is explicitly authorized.**

1. **Go-ahead:** ARCH §17/§18 — "No ROS coding until owner explicitly says go."
2. **ROS 2 / Nav2 version: CONFIRMED — ROS 2 Lyrical Luth.** Older `architecture.md` references to Jazzy/Humble are documentation mistakes. The installed Lyrical-compatible APIs will be verified when ROS integration begins.
3. **Mask adapter:** `Image` + `port_meta` → `SemanticMaskInput`.
4. **CameraInfo adapter:** `CameraInfo` → `CameraIntrinsicsInput`, plus a distortion decision.
5. **TF adapter:** camera → costmap frame at mask stamp → `CameraGroundInput`, or a 6-DoF projection core.
6. **Parameter adapter:** `config/robots/` → `GridInput` + Dev 3 tuning. DEV lists `config/robots/` under Dev 3 (costmap params), Dev 4 (planner/controller params) and Dev 5.
7. **Output mapping:** core cost array → `nav_msgs/OccupancyGrid` / Nav2 costmap values.
8. **Nav2 integration form:** semantic costmap layer plugin (DEV Dev 3 task 1, ARCH §5) vs a standalone publisher.
9. **Real calibration:** intrinsics (PC §5) and camera extrinsics (Dev 5 URDF).
10. **Real footprint:** Dev 5 footprint YAMLs → `FootprintInput`, and the footprint → inflation relationship (inscribed radius etc.; README §8).
11. **Geometry producer:** §5 above.
12. **Stale-mask fail-safe:** ARCH §8.6 ROI (§6 above).
13. **Multi-resolution inflation:** DEV Dev 3 task 4; the core has a single-resolution inflation only.
14. **Inflation parameters:** radius and a Nav2-compatible decay (`cost_scaling_factor` etc.) are not specified.

---

## 9. Ownership table

| Item | Needed by Dev 3 | Current status | Owner/Dependency | Action needed |
|---|---|---|---|---|
| Semantic mask `/segmentation/mask` | yes | CONFIRMED | Dev 1 | None for the core; write mask adapter at ROS stage |
| Mask validity (`port_meta`) | yes | CONFIRMED (format provisional) | Dev 1 | Confirm final `PortMeta.msg` format |
| Confidence `/segmentation/confidence` | no | AVAILABLE BUT NOT USED | Dev 1 | None |
| CameraInfo topic for Dev 3 | yes | AVAILABLE, topic NOT YET DEFINED | Dev 5 (driver); Dev 1 republishes | Agree which topic Dev 3 consumes |
| Camera intrinsics values / YAML | yes | BLOCKED | Owner not yet defined (DEV: Dev 2; D1 docs: Dev 1) | Resolve ownership; supply calibration |
| Distortion / rectified mask | yes | NOT YET DEFINED | Owner not yet defined | Confirm whether mask is rectified |
| TF `map → odom → base_link` | yes | BLOCKED | Dev 2 | Available at ROS stage |
| Camera extrinsics (`base_link → camera`) | yes | BLOCKED | Dev 5 (URDF) | Provide values; confirm TF broadcaster |
| Robot footprint | optional | BLOCKED | Dev 5 (dual footprint YAMLs) | Provide polygon; agree which footprint Dev 3 uses |
| Geometry / occupancy producer | optional | NOT YET DEFINED | Depth Anything: Dev 1 T08 (deferred); integration: Dev 3; occupancy conversion: Owner not yet defined | Define topic, type, units, frame, rasterisation |
| Costmap frame, resolution, extent | yes | NOT YET DEFINED | Dev 3 `config/robots/` (shared per DEV) | Decide and put in config |
| Inflation radius / decay | yes | NOT YET DEFINED | Dev 3 `config/robots/` | Decide values; replace placeholder decay |
| Dev 3 max age / stamp tolerance | yes | NOT YET DEFINED | Owner not yet defined | Team decision |
| Stale-mask ROI fail-safe | yes | NOT YET DEFINED | Owner not yet defined | Define ROI extent and owner |
| Core cost → OccupancyGrid mapping | yes | NOT YET DEFINED | Dev 3 | Define mapping |
| Nav2 layer vs publisher | yes | NOT YET DEFINED | Dev 3 (with Dev 4 as consumer) | Decide integration form |
| ROS 2 / Nav2 version | yes | **CONFIRMED — Lyrical Luth** | Current team target | Verify installed Lyrical/Nav2 APIs at ROS integration stage |
| ROS coding go-ahead | yes | BLOCKED | Project owner (ARCH §17) | Explicit go |

---

## 10. Definition of ready for ROS integration

- [ ] Project owner has given the explicit go for ROS coding (ARCH §17).
- [ ] ROS 2 **Lyrical Luth** / compatible Nav2 APIs inspected before ROS implementation.
- [ ] Camera optical, robot and costmap frame names agreed and placed in configuration.
- [ ] CameraInfo source topic agreed; rectification/distortion question answered.
- [ ] Intrinsics ownership resolved and a real (or agreed simulation) calibration available.
- [ ] Camera extrinsics available via TF from Dev 5's robot description; Dev 2's `map → odom → base_link` available (sim or bag is enough).
- [ ] Decision on whether a TF pose can be reduced to the core's height + pitch model, or `projection.py` must be extended.
- [ ] Costmap resolution/extent, inflation radius and Dev 3 max age / stamp tolerance decided in `config/robots/`.
- [ ] Core-cost → OccupancyGrid / Nav2 cost mapping defined; layer-plugin vs publisher decided.
- [ ] Stale-mask fail-safe (ARCH §8.6) ROI and owner defined, or explicitly deferred.
- [ ] Footprint (Dev 5) and geometry side-channel either delivered or explicitly deferred (both are optional in the core).
- [ ] Dev 3 ROS-independent test suite passing.