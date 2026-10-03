# Project A: Production Work Breakdown & Modular Architecture


---

Development Ownership Policy

"dev.md" defines the functional domain and ownership of each developer. It helps clarify what to do in each domain, but it is not a fixed or exhaustive list of development tasks, examples, or clear ownership boundaries.

As development progresses, new tasks, changes, or conflicting requirements may arise. Any task shall be owned and implemented by the developer whose defined functional domain the task falls under, even if that task is not explicitly listed in "dev.md".

Rules

1. "arch.md" has highest authority.
   If any task, implementation decision, or developer responsibility conflicts with "arch.md", "arch.md" takes precedence.
2. Functional-domain ownership resolves task assignment.
   When a new or conflicting task arises between developers, the developer whose functional domain the task belongs to owns the task.
3. Existing task lists do not define ownership boundaries.
   They provide context for understanding responsibilities, but ownership is determined by the functional domains defined in "dev.md".
4. Cross-domain tasks require coordination.
   The developer owning the primary functional domain remains responsible for the task and coordinates with other affected developers where necessary.
5. "dev.md" defines ownership, not architecture. It cannot override or reinterpret "arch.md"
---
**File:** `dev.md`  
**Reference Document:** [`architecture.md`](file:///home/light/Documents/sih/architecture.md)  
**Scope:** Deployable Software-Only Product for Differential-Drive UGV Autonomous Outdoor Navigation  
**Middleware:** ROS 2 (Lyrical as specified in architecture §6)  
**Effort Limit:** ~30 Hours per Developer (~144 Hours Total across 5 Developers)  

---

## 1. Task Domain Grouping & Difficulty Hierarchy

Tasks are grouped strictly by **functional domain** so that related engineering problems are owned by the same developer (AI/Vision together, SLAM/Spatial together, Costmaps/Geometry together, Planning/Control logic together, and Safety/Platform together). 

No arbitrary external tech stacks or third-party libraries are imposed; all components adhere strictly to the stack defined in `architecture.md` §6 (ROS 2, RTAB-Map, Nav2, RUGD SegFormer, YOLOE selectable, tutorial ONNX eval, Depth Anything 3 Metric Large, VoxelLayer, Priority Mux / Watchdog).

The difficulty hierarchy is calibrated so that **Dev 1 (AI & Vision) is higher in technical difficulty than Dev 4 (Planning/Control Logic) and Dev 5 (Safety/Platform)**:

```
                      TECHNICAL DIFFICULTY HIERARCHY
                      
  [Dev 2: SLAM & 3D Spatial Localization] ──► Difficulty: High (4.5 / 5)
  [Dev 1: Perception, RUGD SegFormer & Depth AI]   ──► Difficulty: High (4.0 / 5)  ◄── (Higher than 4 & 5)
  [Dev 3: Costmaps & Geometry Precedence] ──► Difficulty: High (4.0 / 5)
  [Dev 4: Navigation Planning & Control]  ──► Difficulty: Medium-High (3.5 / 5)
  [Dev 5: Safety Authority & Platform]    ──► Difficulty: Medium (3.0 / 5)
```

| Dev | Functional Domain | Architecture Package Ownership | Workload | Difficulty | Core Product Deliverables |
|---|---|---|:---:|:---:|---|
| **Dev 1** | **Perception, AI & Vision** | `ugv_perception/`<br>`config/perception/`<br>`config/ontologies/` | **29h** | **High (4.0/5)**<br>*Hardware-agnostic inference abstraction, multi-model pipelines, latency bounds, confidence math* | Consume camera **data** (`Image` + `CameraInfo`; does **not** own the driver), RUGD SegFormer outdoor adapter, YOLOE selectable, Tutorial ONNX eval adapter, Depth Anything 3 Metric Large, 3-class canonical port (`0, 1, 2`), confidence normalizer, staleness fail-safe (`/ugv/perception_degraded`). |
| **Dev 2** | **SLAM & Spatial Localization** | `ugv_localization/`<br>`config/cameras/` | **29h** | **High (4.5/5)** | RTAB-Map visual SLAM (stereo/RGB-D/mono), `mapping` vs `localize` database modes, continuous TF tree (`map->odom->base_link`), pose validity monitor node (`/ugv/pose_valid`). |
| **Dev 3** | **Costmaps & Spatial Geometry** | `ugv_navigation/` (Costmap Subsystem), `config/robots/` | **29h** | **High (4.0/5)** | Semantic Costmap Layer (mask projection via CameraInfo & TF), VoxelLayer geometry integration, Geometry Lethal Precedence Engine (geometry lethal overrides traversable). |
| **Dev 4** | **Planning & Trajectory Control** | `ugv_navigation/` (Autonomy & Motion Core) | **29h** | **Med-High (3.5/5)** | Nav2 Smac2D global path planner, Regulated Pure Pursuit (RPP) trajectory tracker, dynamic hazard reactivity, recovery behaviors, `/navigate_to_pose`, candidate twist `/cmd_vel_nav2`. |
| **Dev 5** | **Safety Authority & Platform** | `ugv_safety/`<br>`ugv_robot_description/`<br>`ugv_bringup/`<br>`ugv_eval/` | **28h** | **Medium (3.0/5)** | 4-tier Command Priority Arbiter (sole base `/cmd_vel` authority), multi-topic timeout watchdog table, deceleration ramp, diff-drive URDF/xacro, **camera driver launch** (`Image` + `CameraInfo` for Dev 1 and Dev 2), dual footprint YAMLs, Gazebo sim world, launch profiles. |

---

## 2. Production System Architecture & Data Flow

The system is decoupled into 5 clear functional domains with zero circular dependencies:

```
 ┌────────────────────────────────────────────────────────┐
 │     DEV 5 BRINGUP: camera driver (shared sensor)       │
 │           Image + CameraInfo  (dual fan-out)           │
 └───────────────┬────────────────────────────────────────┴───────────────┐
                 │ consume frames                                         │ consume frames
                 ▼                                                        ▼
 ┌────────────────────────────────────────────────────────┐  ┌────────────────────────────────────────────────────────┐
 │           DEV 1: PERCEPTION & VISION SUBSYSTEM         │  │         DEV 2: SLAM & LOCALIZATION SUBSYSTEM           │
 │  Consume Image+CameraInfo -> RUGD SegFormer -> Remap   │  │   RTAB-Map Visual SLAM + TF Tree + Pose Validity       │
 │  (does not own / launch the camera)                    │  │   (does not own the camera driver)                     │
 └───────────────────────────┬────────────────────────────┘  └───────────────────────────┬────────────────────────────┘
                             │ /segmentation/mask {0, 1, 2}                              │ TF (map->odom->base)
                             │ /ugv/perception_degraded                                  │ /ugv/pose_valid
                             ▼                                                           │
 ┌────────────────────────────────────────────────────────┐                              │
 │       DEV 3: COSTMAPS & GEOMETRY SUBSYSTEM             │◄─────────────────────────────┘
 │   Camera Projection to Grid + Geometry Conflict Engine │
 └───────────────────────────┬────────────────────────────┘
                             │ /global_costmap/costmap
                             │ /local_costmap/costmap
                             ▼
                               ┌────────────────────────────────────────────────────────┐
                               │       DEV 4: NAVIGATION PLANNING & CONTROL LOGIC       │
                               │     Smac2D Planner + Regulated Pure Pursuit (RPP)      │
                               └────────────────────────────┬───────────────────────────┘
                                                            │ /cmd_vel_nav2 (Candidate Twist)
                                                            ▼
                               ┌────────────────────────────────────────────────────────┐
                               │       DEV 5: SAFETY AUTHORITY & BASE PLATFORM          │
                               │  Priority Arbiter + Watchdogs + Base Control + Bringup │
                               └────────────────────────────────────────────────────────┘
```

---

## 3. Production Interface Contract

All 5 developers integrate against the topic contracts defined in `architecture.md`:

| Topic / Service | Message Type | Publisher | Subscriber(s) | Contract Rules (§3.1, §8, §9, §10, §12) |
|---|---|---|---|---|
| Camera `Image` + `CameraInfo` | `sensor_msgs/msg/Image`<br>`sensor_msgs/msg/CameraInfo` | **Dev 5** (bringup / driver) | **Dev 1**, **Dev 2** | Shared vision sensor (architecture §5). Stamp = image time; `frame_id` matches `CameraInfo`. Dev 1 does **not** open V4L2. Intrinsics YAML: Dev 2 `config/cameras/`. |
| `/segmentation/mask` | `sensor_msgs/msg/Image` | **Dev 1** | **Dev 3**, Dev 5 | Encoding `mono8`, pixels strictly in `{0: unknown, 1: traversable, 2: hazard}`. Header timestamp matches source frame. |
| `/ugv/perception_degraded` | `std_msgs/msg/Bool` | **Dev 1** | **Dev 5** | Emits `true` if latency > `perception_max_age` or confidence gates trip. |
| `/perception/depth_cloud` | `sensor_msgs/msg/PointCloud2` | **Dev 1** | **Dev 2**, Dev 3 | x/y/z in metres. Stamp = source image stamp, frame = camera optical frame, back-projected with the raw K at camera resolution. |
| `TF (map->odom->base)` | `tf2_msgs/msg/TFMessage` | **Dev 2** | **Dev 3**, Dev 4, Dev 5 | Continuous tree, jitter $< 50\text{ ms}$, publish rate $\ge 15\text{ Hz}$. |
| `/ugv/pose_valid` | `std_msgs/msg/Bool` | **Dev 2** | **Dev 5** | Emits `false` if tracking lost, TF expires, or covariance explodes (§10.1). |
| `/global_costmap/costmap`<br>`/local_costmap/costmap` | `nav_msgs/msg/OccupancyGrid` | **Dev 3** | **Dev 4** | 2D occupancy grid combining semantic layers and geometry precedence (§9). |
| `/cmd_vel_nav2` | `geometry_msgs/msg/Twist` | **Dev 4** | **Dev 5** | Candidate velocity command from Nav2 controller. |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | **Dev 5** (Safety Arbiter) | **Base Wheels / Sim** | **Sole final authority commanding physical wheel actuators (§3.1)**. |
| `/ugv/e_stop` | `std_msgs/msg/Bool` | Dev 5 CLI / Operator | Dev 5 | Priority Level 1 hard software kill switch. |

---

## 4. Subsystem Breakdown & Assigned Tasks

---

### Dev 1: Perception, AI & Vision Specialist
* **Domain Focus:** All vision processing, neural model adapters, ontology remapping, and confidence calibration.
* **Package Ownership:** `ugv_perception/`, `config/perception/`, `config/ontologies/`
* **Workload:** 29 Hours | **Difficulty:** **High (4.0/5)** — *Hardware-agnostic inference abstraction across target compute backends, multi-model vision pipelines, strict real-time latency bounds, and confidence calibration.*

#### Production Tasks:
1. **Consume Camera Data (§5, §8.4, §8.5) (4h):** Subscribe to Dev 5’s `Image` + `CameraInfo` (do **not** own the camera driver or V4L2). Convert to perception frames with the **image** timestamp and optical `frame_id` matching `CameraInfo`. Reject missing/fake `K`. Fail closed — no black frame.
2. **RUGD SegFormer Outdoor Adapter Pipeline (§6, §8) (7h):** Implement the primary outdoor perception adapter using RUGD SegFormer-B5 to generate path and hazard segmentation masks; YOLOE remains selectable; integrate Tutorial ONNX scaffold as eval only.
3. **Depth Anything 3 Metric Large Geometry Pipeline (§6, §9) (5h):** Build inference pipeline for Depth Anything 3 Metric Large to provide depth geometry for obstacle verification.
4. **Canonical Ontology Remapping Engine (§8.2) (4h):** Build YAML-driven translation parser mapping raw model labels strictly to canonical classes:
   - `0: unknown` (never free — costmap inflates)
   - `1: traversable` (free / low cost)
   - `2: hazard` (lethal / inscribed cost)
5. **Confidence Normalization & Gating Engine (§8.3) (5h):** Normalize model scores to $[0.0, 1.0]$ and apply per-adapter threshold profiles ($\tau_{trav}, \tau_{haz}, \tau_{min}, \kappa$).
6. **Freshness & Stale-Mask Fail-Safe (§8.4, §8.6) (4h):** Track image latency ($now - stamp$). If age > `perception_max_age` or confidence fails, publish `/ugv/perception_degraded = true`.

#### Modularity & Flexibility:
- **Independent Testing:** Can test decode + adapters on recorded outdoor **bags of Image+CameraInfo**, live topics (when Dev 5 is publishing), or RUGD (eval). Does not require SLAM, planners, or owning the camera driver.
- **Modification Freedom:** Can change model weights, upgrade inference engines, or tune confidence profiles without touching downstream navigation or safety code.

#### CLI & Verification Commands:
```bash
# Perception consumes camera topics (Dev 5 must be publishing Image + CameraInfo)
ros2 run ugv_perception adapter_node --ros-args -p adapter:=rugd

# Run unit tests validating 3-class contract (fails if any pixel != 0, 1, 2)
colcon test --packages-select ugv_perception

# Inspect freshness and degradation flag
ros2 topic echo /ugv/perception_degraded
```

---

### Dev 2: SLAM & 3D Spatial Localization Specialist
* **Domain Focus:** 3D spatial transformations, visual feature tracking, graph optimization, and visual odometry.
* **Package Ownership:** `ugv_localization/`, `config/cameras/`
* **Workload:** 29 Hours | **Difficulty:** **High (4.5/5)** — *Highest mathematical complexity (epipolar geometry, bundle adjustment, visual drift tuning, 3D transform tree).*

#### Production Tasks:
1. **RTAB-Map Visual SLAM Pipeline Configuration (§6, §10) (7h):** Configure visual odometry and graph SLAM for stereo (recommended) / RGB-D / mono (minimum). Tune visual feature tracking, bundle adjustment, and keyframing.
2. **Dual Operating Modes (§10) (6h):** Build runtime parameter handling to toggle between:
   - `mapping`: Builds 3D visual graph and saves database to disk (`rtabmap.db`).
   - `localize`: Loads database read-only; performs visual localization against landmarks without modifying graph.
3. **Dynamic TF Chain Broadcaster (§8.5, §10.1) (5h):** Manage and broadcast low-jitter, high-frequency ($\ge 15\text{ Hz}$) transforms: `map -> odom -> base_link`.
4. **Pose Validity Monitor Node (§10.1) (5h):** Build monitor checking TF freshness ($< localization\_max\_age$), tracking state (VO-lost), and covariance bounds; publish `/ugv/pose_valid = false` on failure.
5. **Sensor Honesty Benchmarks & Drift Docs (§10) (3h):** Quantify visual odometry drift under outdoor lighting and document mono camera limitations vs stereo.
6. **Standalone Bag Evaluation Harness (3h):** Configure offline rosbag playback launch and transform latency verification scripts.

#### Modularity & Flexibility:
- **Independent Testing:** Operates strictly on stereo/mono **topics** (Dev 5 camera driver) or recorded sensor rosbags. Does not consume semantic masks, own the camera, or depend on Nav2.
- **Modification Freedom:** Can adjust visual feature detectors, loop-closure parameters, or bundle adjustment internally without breaking other modules.

#### CLI & Verification Commands:
```bash
# Launch visual SLAM in mapping or localize mode on recorded bag
ros2 launch ugv_localization localization.launch.py mode:=mapping
ros2 launch ugv_localization localization.launch.py mode:=localize

# Check TF tree transform rates and continuity
ros2 run tf2_ros tf2_echo map base_link
ros2 run tf2_tools view_frames

# Check pose validity output
ros2 topic echo /ugv/pose_valid
```

---

### Dev 3: Costmaps & Spatial Geometry Specialist
* **Domain Focus:** Spatial 2D/3D costmap representation, optical projection, and geometry conflict resolution.
* **Package Ownership:** `ugv_navigation/` (Costmap Subsystem), `config/robots/` (costmap params)
* **Workload:** 29 Hours | **Difficulty:** **High (4.0/5)** — *Coordinate projection mathematics from optical frame to costmap grid, and conflict priority logic.*

#### Production Tasks:
1. **Semantic Costmap Layer Plugin (§5, §8.5) (9h):** Implement costmap layer that subscribes to `/segmentation/mask`, uses TF and `CameraInfo` to project pixels into the 2D costmap grid via optical-to-ground projection.
2. **3-Class Cost Intent Mapping (§8.1) (5h):**
   - Class 0 (`unknown`) $\to$ inflate / non-free
   - Class 1 (`traversable`) $\to$ free / clear cost
   - Class 2 (`hazard`) $\to$ lethal / inscribed cost
3. **Geometry Precedence Engine (§9) (7h):** Integrate optional VoxelLayer / Depth Anything 3 Metric Large geometry side-channel and enforce Design Law: **geometry lethal always wins**; semantic traversable **never** clears geometric lethal obstacles.
4. **Multi-Resolution Inflation & Footprint Padding (§8.1) (5h):** Configure costmap inflation layers to pad obstacles based on robot footprint configurations.
5. **Costmap Validation Testbench (3h):** Build unit tests verifying costmap cell values, projection accuracy, and geometry lethal overrides.

#### Modularity & Flexibility:
- **Independent Testing:** Can test costmap projection against any recorded rosbag or simulated environment, inspecting `/local_costmap/costmap` in RViz or via ROS CLI.
- **Modification Freedom:** Can modify projection parameters, inflation radii, or geometry thresholds without touching planning algorithms or vision models.

#### CLI & Verification Commands:
```bash
# Launch costmap server independently with recorded bag
ros2 launch ugv_navigation costmap.launch.py

# Echo costmap values at hazard coordinates
ros2 topic echo /local_costmap/costmap --once

# Run unit test validating geometry lethal precedence over semantic traversable
colcon test --packages-select ugv_navigation --ctest-args -R test_geometry_precedence
```

---

### Dev 4: Planning & Trajectory Control Specialist
* **Domain Focus:** Global path search, trajectory tracking control, dynamic reactivity, and autonomous navigation logic.
* **Package Ownership:** `ugv_navigation/` (Autonomy & Motion Core), `config/robots/` (planner/controller params)
* **Workload:** 29 Hours | **Difficulty:** **Med-High (3.5/5)** — *Non-holonomic path planning math, trajectory control stability, behavior tree orchestration.*

#### Production Tasks:
1. **Smac2D Global Path Planner Configuration (§6, §11) (7h):** Configure Smac2D 2D Feasible search on costmaps to generate smooth, kinematically feasible global paths.
2. **Regulated Pure Pursuit (RPP) Controller Tuning (§6, §11) (8h):** Configure RPP trajectory tracker for differential drive: adaptive lookahead distance, curvature-based deceleration, and collision-aware speed scaling.
3. **Dynamic Hazard Reactivity (§11) (5h):** Ensure planner and controller dynamically replan in real-time as live costmaps update with newly perceived hazards.
4. **Behavior Trees & Recovery Behaviors (4h):** Implement Nav2 recovery behaviors (spin-in-place, wait, backtrack) and configure `/navigate_to_pose` action server.
5. **Candidate Velocity Command Boundary (§3.1) (2h):** Route controller output exclusively to candidate topic `/cmd_vel_nav2` (never direct to base `/cmd_vel`).
6. **Navigation CLI & Goal Testbench (3h):** Build automated goal-sending test scripts and benchmark A-to-B navigation completion times.

#### Modularity & Flexibility:
- **Independent Testing:** Can test path planning and trajectory control against pre-recorded or simulated costmaps. Focuses purely on navigation logic.
- **Modification Freedom:** Can tune lookahead parameters, cost penalties, or recovery routines without affecting safety mux logic or perception adapters.

#### CLI & Verification Commands:
```bash
# Launch Nav2 planning & control stack
ros2 launch ugv_navigation navigation.launch.py

# Send NavigateToPose goal via CLI
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 3.0, y: 0.0}}}}"

# Echo candidate velocity commands emitted by RPP controller
ros2 topic echo /cmd_vel_nav2
```

---

### Dev 5: Safety Authority, Platform & Base Specialist
* **Domain Focus:** Safety gatekeeping, system health watchdogs, physical robot kinematics, simulation physics, and motor actuation.
* **Package Ownership:** `ugv_safety/`, `ugv_robot_description/`, `ugv_bringup/`, `ugv_eval/`, `config/safety/`, `config/robots/`
* **Workload:** 28 Hours | **Difficulty:** **Medium (3.0/5)** — *Deterministic state logic, URDF modeling, simulation setup, and motor interfacing.*

#### Production Tasks:
1. **4-Tier Command Priority Arbiter (§3.1) (6h):** Implement the sole authoritative publisher to base `/cmd_vel` with strict precedence:
   - Level 1: E-stop asserted $\to$ zero twist (`0.0, 0.0`)
   - Level 2: Health watchdog timeout trip $\to$ zero twist
   - Level 3: Degraded perception OR invalid pose $\to$ zero twist (hold)
   - Level 4: Candidate `/cmd_vel_nav2` $\to$ forwarded to base
2. **System Health Watchdog Table (§12) (5h):** Build asynchronous timeout monitor tracking arrival timestamps against `safety_timeouts.yaml` for perception mask ($0.5\text{s}$), localization/TF ($0.5\text{s}$), and Nav2 heartbeat ($0.5\text{s}$).
3. **Deceleration Profiler & Motor Driver Interface (§3.1) (5h):** Implement smooth rate-limited deceleration ramp on safety stop; write differential-drive hardware motor driver node commanding wheel actuators.
4. **URDF/xacro Model & Dual Footprints (§7, §13 item 9) (4h):** Build differential-drive robot description with kinematics, camera extrinsics, and primary + secondary footprint YAMLs.
5. **Outdoor Gazebo Simulation World & Launch Profiles (§4) (5h):** Build outdoor Gazebo world with terrain, dirt tracks, and obstacles; create master launch files for runtime profiles: `profile:=live_cam|sim|bag`. **`live_cam` launches the camera driver** and publishes `Image` + `CameraInfo` for Dev 1 and Dev 2 (architecture §5 dual fan-out). Dev 1/2 do not start the camera.
6. **E-Stop CLI Utility & Safety Test Suite (3h):** Build CLI tool to toggle E-stop, publish `/ugv/safety_status`, and verify immediate zero-twist clamp.

#### Modularity & Flexibility:
- **Independent Testing:** Acts as the complete robot base platform. Can be tested by streaming test twist commands into `/cmd_vel_nav2` and verifying wheel actuation or simulation motion.
- **Modification Freedom:** Can alter motor acceleration curves, wheel dimensions, or simulation models without touching navigation or perception algorithms.

#### CLI & Verification Commands:
```bash
# Launch safety authority and base driver
ros2 launch ugv_safety safety.launch.py

# Assert E-stop via CLI
ros2 topic pub /ugv/e_stop std_msgs/msg/Bool "{data: true}" --once

# Stream test candidate twist and verify zero-twist cutoff on timeout
ros2 topic pub /cmd_vel_nav2 geometry_msgs/msg/Twist "{linear: {x: 0.5}}" -r 10
ros2 topic echo /cmd_vel

# Launch outdoor simulation profile
ros2 launch ugv_bringup bringup.launch.py profile:=sim
```
