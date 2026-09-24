# Dev 3 → Dev 4 interface (pre-ROS hand-off)

Status as of this document: Dev 3 algorithms are implemented and tested with
synthetic inputs. **ROS/Nav2 integration is not done.** Nothing here freezes
the final ROS interface.

Sources: `PROJECT_CONTEXT.md` (**PC**), `dev.md` (**DEV**), `architecture.md`
(**ARCH**), Dev 1 code under `whoami/turing/src/ugv_perception/` (**D1**,
read-only), Dev 3 code under `ugv_navigation/costmap_core/`, and
[`costmap_core/INTEGRATION_CHECKLIST.md`](costmap_core/INTEGRATION_CHECKLIST.md)
(**CHECKLIST**). No Dev 2 source code exists in this workspace; Dev 2 facts
come only from the documents above.

Labels: **CONFIRMED** (in project docs or code) · **INTENDED** (documented
target, not implemented) · **IMPLEMENTED** (Dev 3 code, tested) ·
**PENDING / TBD** (not defined yet).

---

## 1. Purpose

Defines the current Dev 3 → Dev 4 hand-off so Dev 4 can start planner and
controller development against synthetic costmaps before live ROS integration.

```
Perception (Dev 1) → Dev 3 costmap → Dev 4 planner/controller
```

## 2. Dev 3 responsibility

Per PC §3 and DEV §4 (Dev 3), Dev 3 owns the costmap subsystem in
`ugv_navigation/` and costmap parameters in `config/robots/`:

- semantic costmap processing (class → cost)
- spatial projection (camera pixel → ground → grid)
- geometry / semantic cost fusion
- geometry lethal precedence ("geometry lethal always wins")
- inflation (and footprint padding)
- the costmap pipeline
- eventually, the global and local costmap outputs

Dev 4 also works in `ugv_navigation/` (Autonomy & Motion Core) and
`config/robots/` (planner/controller params) per DEV §1/§4.

## 3. Intended Dev 3 → Dev 4 interface (INTENDED)

| Costmap | Topic | Type | Publisher | Subscriber |
|---|---|---|---|---|
| Global | `/global_costmap/costmap` | `nav_msgs/msg/OccupancyGrid` | Dev 3 | Dev 4 |
| Local | `/local_costmap/costmap` | `nav_msgs/msg/OccupancyGrid` | Dev 3 | Dev 4 |

Source: DEV §3 interface table, PC §3/§4. Contract rule in DEV §3: "2D
occupancy grid combining semantic layers and geometry precedence (§9)".

**PENDING / TBD** (no source defines them):

- frame names (PC §6 lists `map`, `base_link` only as *temporary* examples)
- resolution, width/height, origin
- update / publish frequency
- global vs local extent
- ROS publisher vs Nav2 costmap layer plugin architecture (ARCH §5 shows a
  Nav2 costmap with a semantic layer + optional VoxelLayer; DEV Dev 3 task 1
  says "Semantic Costmap Layer Plugin"; the concrete form is not decided)
- conversion rules from Dev 3 internal costs to `OccupancyGrid` data

## 4. Internal Dev 3 cost semantics (IMPLEMENTED)

These are the values in Dev 3's internal numpy costmap
(`class_to_cost.CostValues`, `geometry_costmap.GeometryCostValues`,
`inflation.py`). Defaults follow Nav2 `costmap_2d` cost conventions and are
overridable.

| Value | Meaning | Produced by |
|---|---|---|
| `0` | free / traversable | semantic class 1, or free geometry |
| `1..253` | intermediate: inflated cost near a lethal cell | `inflation.inflate_costmap` (placeholder linear decay) |
| `254` | lethal | semantic class 2 (hazard) or geometric occupancy |
| `255` | unknown | semantic class 0, **and** any cell no mask pixel reached |

Rules enforced in code:

- **Geometry lethal always wins** (ARCH §9, PC §3): a geometrically occupied
  cell is `254` whatever the semantic class. Semantic traversable never
  clears it.
- Several pixels in one cell: HAZARD > UNKNOWN > TRAVERSABLE.
- Unknown is never turned into free (ARCH §8.1: `unknown` = "never free").
- Inflation never lowers a cost and never creates a new lethal cell.
- Grid layout: `array[row, col]`, row = y, col = x, origin = lower-left corner
  of cell (0, 0) (the `OccupancyGrid` origin convention).

**These are Dev 3 internal semantics, not the `OccupancyGrid` wire format.**
No project source defines how they map onto `OccupancyGrid.data`.
Mapping is **PENDING**.

## 5. Dev 1 input contract (CONFIRMED, context only)

Dev 4 does not consume these directly. Verified against D1 code
(`port/ids.py`, `node/wire.py`, `node/adapter_node.py`) and DEV §3.

| Topic | Type | Content | Dev 3 use |
|---|---|---|---|
| `/segmentation/mask` | `sensor_msgs/msg/Image`, `mono8` | classes `0` unknown, `1` traversable, `2` hazard; stamp = image time; `frame_id` = optical frame | main input |
| `/segmentation/port_meta` | `std_msgs/Float64MultiArray` (until D1's `PortMeta.msg` is compiled) | `[valid, age, scale]`; v1 `scale = 1.0` | validity (core refuses `valid=False`) |
| `/segmentation/camera_info` | `sensor_msgs/msg/CameraInfo` | D1 republishes the last received CameraInfo (driver owned by Dev 5) | intrinsics source; topic choice for Dev 3 PENDING |
| `/segmentation/confidence` | `sensor_msgs/msg/Image`, `32FC1` | per-pixel `[0,1]` | not used |
| `/ugv/perception_degraded` | `std_msgs/msg/Bool` | perception stale/degraded | subscriber is Dev 5, not Dev 3 |

## 6. Geometry input

Dev 3 abstraction: `contracts.OccupancyInput` (IMPLEMENTED).

- 2D boolean grid, same shape and frame as the costmap grid.
- `True` = occupied → cell becomes lethal (`254`).
- Geometry lethal always wins over semantic free/traversable.
- Optional: without it the pipeline uses semantics only.

**The live geometry producer is not implemented or connected.** ARCH §6/§9
describe an optional Depth Anything → VoxelLayer side-channel; Dev 1 lists
Depth Anything as task T08, deferred. Its topic, units, frame and conversion
to occupancy are **PENDING** (CHECKLIST §5). Depth Anything is **not**
integrated with Dev 3.

## 7. What Dev 4 can assume

- Free cells represent traversable space.
- Lethal cells represent obstacles / hazards.
- Unknown cells represent unknown space. At the costmap level the project
  says unknown is "never free" (ARCH §8.1, DEV §4 Dev 3 task 2).
- Intermediate values may represent inflated cost near obstacles.
- Geometry lethal has precedence over semantic traversable.
- Global and local costmaps (`/global_costmap/costmap`,
  `/local_costmap/costmap`) are the intended outputs.
- Dev 4 consumes the costmaps (DEV §3) and outputs `/cmd_vel_nav2` (§10).
- Costmaps are refreshed continuously from live mask (+ optional voxel)
  updates. That is the v1 dynamic-obstacle mechanism (ARCH §11).

## 8. What Dev 4 must NOT assume

All of the following are **PENDING / TBD**:

- final costmap frame(s)
- final resolution
- final map dimensions
- final map origin
- final update frequency
- final internal-cost → `OccupancyGrid` conversion
- final global / local extents
- final inflation parameters (radius, decay curve; current decay is a placeholder)
- final robot footprint (Dev 5 dual footprint YAMLs; not provided yet)
- final camera extrinsics (Dev 5 robot description; not provided yet)
- final TF lookup behaviour (API, timeout, stamp tolerance)
- final stale-mask behaviour (ARCH §8.6 "front ROI lethal/max-inflate"; ROI and owner undefined)
- a live geometry producer
- final Nav2 integration architecture (layer plugin vs publisher, ROS 2 version:
  PC says Lyrical, ARCH §6 says Jazzy/Humble)

**Dev 2:** the documents specify that Dev 2 will provide the TF chain
`map → odom → base_link` (≥ 15 Hz, jitter < 50 ms) and `/ugv/pose_valid`
(consumed by Dev 5) (DEV §3, PC §3). There is no Dev 2 implementation in this
workspace; do not assume any of it is available.

## 9. Synthetic costmap testing for Dev 4

Dev 4 can start now with synthetic costmaps built directly in Dev 3's internal
semantics (§4), or generated with `costmap_core.pipeline.run_costmap_pipeline`
from synthetic inputs. All grid sizes, resolutions and positions in
such tests are test values, not project values.

Suggested scenarios (test scenarios only):

| Scenario | Content | What it tests |
|---|---|---|
| Empty / free map | all `0` | planning through open space |
| Obstacle wall with opening | a line of `254` with a gap of `0` | routing through the available opening |
| Corridor | `0` channel bounded by `254` | planning through constrained traversable space |
| Unknown region | a block of `255` | Dev 4's chosen unknown-space behaviour, tested explicitly. The project defines unknown as "never free" at the costmap level. The planner's policy for unknown cells is **not** defined here. |
| Inflated obstacle | `254` cells surrounded by `1..253` (e.g. from `inflation.inflate_costmap`) | planner behaviour around intermediate costs |
| Dynamic obstacle update | a sequence of costmaps where `254` cells appear / move | reaction to changing costmap data (DEV Dev 4 task 3) |

Minimal example (all values synthetic):

```python
import numpy as np
FREE, LETHAL, UNKNOWN = 0, 254, 255
grid = np.full((50, 50), FREE, dtype=np.int64)   # [row=y, col=x]
grid[25, :] = LETHAL                             # wall
grid[25, 20:24] = FREE                           # opening
```

## 10. Dev 4 downstream flow

```
Dev 3 costmaps → Dev 4 planner/controller (Smac2D + RPP) → /cmd_vel_nav2
```

- `/cmd_vel_nav2`: `geometry_msgs/msg/Twist`, published by Dev 4, consumed by
  Dev 5 (DEV §3).
- **Dev 5 is the final safety authority and sole owner of `/cmd_vel`**
  (ARCH §3.1, DEV §3). Dev 4 never publishes `/cmd_vel` directly.

## 11. Current Dev 3 status

### Implemented (ROS-independent, `ugv_navigation/costmap_core/`)

| Component | Module |
|---|---|
| Class → cost mapping | `class_to_cost.py` |
| Semantic costmap (mask → same-shape costmap) | `semantic_costmap.py` |
| Pixel → ground projection (pinhole, height + pitch) | `projection.py` |
| Ground → grid cell conversion | `grid.py` |
| Mask projection onto the grid | `mask_projection.py` |
| Geometry costmap abstraction | `geometry_costmap.py` |
| Semantic + geometry fusion (geometry lethal wins) | `costmap_fusion.py` |
| Inflation | `inflation.py` |
| Footprint validation / padding | `footprint.py` |
| Input contracts | `contracts.py` |
| Single pipeline entry point | `pipeline.py` |
| Tests | `ugv_navigation/tests/`: **311 passed, 0 failed** |

The algorithms are implemented and tested with **synthetic inputs**. What's
missing is live system integration.

### Pending

- ROS adapters (mask, CameraInfo, TF, parameters, output)
- internal cost → `OccupancyGrid` conversion and Nav2 integration form
- real frames, calibration, extrinsics, footprint
- TF integration (Dev 2 TF chain + Dev 5 camera extrinsics)
- live geometry producer
- stale-mask fail-safe, freshness / stamp tolerance values
- global/local costmap configuration
- ROS 2 / Nav2 version resolution and the explicit go-ahead for ROS coding (ARCH §17)

Full list: CHECKLIST §8–§10.

## 12. Dev 4 can start now

**Dev 4 can begin planner development and testing against synthetic
costmaps now.** Dev 4 does not need to wait for:

- ROS installation
- a live Dev 1 → Dev 3 ROS adapter
- live TF
- final camera calibration
- a live geometry producer
- final Nav2 integration

Synthetic testing does **not** mean the final ROS/Nav2 interface is frozen.
Frames, resolution, extents, conversion to `OccupancyGrid`, and inflation
parameters may all change.

## 13. Pending interface table

| Item | Status |
|---|---|
| Global costmap topic | Intended |
| Local costmap topic | Intended |
| OccupancyGrid type | Intended |
| Internal cost semantics | Implemented |
| OccupancyGrid conversion | Pending |
| Costmap frame | Pending |
| Resolution | Pending |
| Global extent | Pending |
| Local extent | Pending |
| Inflation parameters | Pending |
| Robot footprint | Pending |
| TF integration | Pending |
| Live geometry producer | Pending |
| ROS/Nav2 integration | Pending |
| Dev 4 synthetic testing | Ready |
