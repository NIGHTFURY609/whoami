# Dev 3 costmap core — integration boundary

`costmap_core/` is the ROS-independent mathematical core of the Dev 3 costmap
subsystem. It takes plain numpy arrays and frozen dataclasses and returns
numpy cost arrays. It does not import ROS, TF, Nav2, or any other
developer's package.

`contracts.py` defines the **input contract**: the typed, validated data a
future ROS adapter must provide. The algorithms themselves have not been
changed to consume it. An adapter builds a `CostmapCoreInputs` and passes
it to `pipeline.run_costmap_pipeline`, the single entry point that runs
every stage in order (see §G).

```
  ROS world (future adapters)            │  ROS-independent core (this package)
                                         │
  /segmentation/mask + port_meta ──┐     │
  CameraInfo ──────────────────────┤     │   contracts.CostmapCoreInputs
  TF (camera → costmap frame) ─────┼──►  │   (validated, frame/shape-consistent)
  costmap params (config/robots) ──┤     │            │
  geometry side-channel (TBD) ─────┤     │            ▼
  footprint YAML (Dev 5, TBD) ─────┘     │   mask_projection → fusion → inflation
                                         │   footprint (validate / pad)
```

---

## A. What the core accepts

A single costmap update is described by `contracts.CostmapCoreInputs`:

| Field | Type | Required | Wraps existing core type |
|---|---|---|---|
| `mask` | `SemanticMaskInput` | yes | (numpy array, used by `mask_projection`) |
| `intrinsics` | `CameraIntrinsicsInput` | yes | `projection.CameraIntrinsics` |
| `camera_ground` | `CameraGroundInput` | yes | `projection.CameraGroundGeometry` |
| `grid` | `GridInput` | yes | `grid.CostmapGridGeometry` |
| `occupancy` | `OccupancyInput` | optional | (bool array, used by `geometry_costmap`) |
| `footprint` | `FootprintInput` | optional | `footprint.validate_footprint` |

Plus two freshness helpers: `age_s(stamp_ns, now_ns)` and
`is_fresh(stamp_ns, now_ns, max_age_s)`.

All dataclasses are frozen. Array fields are stored as read-only copies. Any
violation raises `contracts.ContractError` (a `ValueError`). Nothing is
coerced silently.

**No real-world value has a default.** Intrinsics, camera height/pitch,
grid resolution/origin/size, frame names, footprint and max age must all be
supplied explicitly.

The bundle checks the inputs against each other:

- `mask.frame_id == intrinsics.frame_id == camera_ground.camera_frame_id`
- `mask` shape `== (intrinsics.image_height, intrinsics.image_width)`
- `camera_ground.ground_frame_id == grid.frame_id`
- if present: `occupancy` shape `== (grid.height, grid.width)` and
  `occupancy.frame_id == grid.frame_id`

It does **not** check stamp alignment between inputs or freshness, because
no tolerance or max age has been agreed for Dev 3. See §D and §8.

Separate from the contract: Dev 3's own tuning (`CostValues`,
`GeometryCostValues`, `lethal_cost`, `inflation_radius`, footprint
`padding`) stays as direct arguments to the algorithm functions. These are
Dev 3 configuration, not upstream inputs.

## B. What each input means

### 1. Semantic mask — `SemanticMaskInput`
- `classes`: 2D `uint8` array (mono8). Every pixel is in `{0 unknown,
  1 traversable, 2 hazard}`. `classes[row, col]` is image pixel
  `(u=col, v=row)`.
- `stamp_ns`: image capture time (mask `header.stamp`), in nanoseconds.
- `frame_id`: the camera optical frame (mask `header.frame_id`).
- `valid`: Dev 1's port validity flag. The contract carries it but does not
  act on it.

### 2. Camera intrinsics — `CameraIntrinsicsInput`
- `intrinsics`: `CameraIntrinsics(fx, fy, cx, cy)` in pixels. From CameraInfo
  `K = [fx 0 cx; 0 fy cy; 0 0 1]`.
- `image_width`, `image_height`: the image size the intrinsics refer to.
- `frame_id`: the optical frame the intrinsics belong to.
- Distortion (`D`, `distortion_model`) is **not** part of the contract. The
  projection core is pure pinhole (see §8).

### 3. Camera-to-ground geometry — `CameraGroundInput`
- `geometry`: `CameraGroundGeometry(camera_height, pitch_rad)`. This is the
  core's simplified model:
  - the ground is a flat plane at z = 0;
  - the camera sits directly above the ground-frame origin;
  - roll = yaw = 0, and the ground x axis follows the camera's forward
    heading.
- `camera_frame_id`: the frame this pose is for (must equal the mask frame).
- `ground_frame_id`: the frame the projected ground points are expressed in
  (must equal the grid frame).
- `stamp_ns`: the time the pose refers to.

### 4. Costmap grid geometry — `GridInput`
- `geometry`: `CostmapGridGeometry(resolution, origin_x, origin_y, width,
  height)`. It uses the `nav_msgs/OccupancyGrid` origin convention: the
  origin is the lower-left corner of cell `(row 0, col 0)`, and rows
  increase with y.
- `frame_id`: the frame the grid is expressed in.

### 5. Geometry / occupancy (optional) — `OccupancyInput`
- `occupied`: 2D `bool` array, already rasterised onto the grid.
  `True` means geometrically occupied (lethal). Integer 0/1 arrays are
  rejected.
- `stamp_ns`, `frame_id`: observation time and grid frame.
- When `occupancy` is `None`, `run_costmap_pipeline` skips geometry and
  fusion (identical to fusing an all-`False` mask). **Geometry lethal always
  wins** in `costmap_fusion`.

### 6. Robot footprint (optional) — `FootprintInput`
- `vertices`: a convex polygon of `(x, y)` points in metres, validated by
  `footprint.validate_footprint`.
- `frame_id`: the robot frame the polygon is expressed in.

### 7. Validity / freshness
- Per-input `stamp_ns` on the mask, camera pose and occupancy.
- `mask.valid` (Dev 1 port flag).
- `age_s` / `is_fresh(stamp_ns, now_ns, max_age_s)`. A future-stamped input
  is never fresh. `max_age_s` has **no default**.

## C. Inputs available now

| Input | Status |
|---|---|
| Semantic mask | **Available from Dev 1 (code)**: `/segmentation/mask`, `sensor_msgs/Image`, mono8, `{0,1,2}`, header stamp = image time, header frame_id = optical frame. The contract fields mirror this. |
| Mask validity | **Available from Dev 1 (code)**: `valid` / `age` / `scale` on `/segmentation/port_meta`. Dev 1 publishes this as `Float64MultiArray` until its `PortMeta.msg` is compiled. Dev 1 only publishes a mask when it is valid. |
| Mask resolution | Dev 1 v1 port enforces `scale == 1.0` (mask size equals source image size). The contract's mask-size-equals-intrinsics-size check relies on this. |
| CameraInfo topic | **Available as a pipe, not as values**: Dev 1 republishes the last received CameraInfo on `/segmentation/camera_info`. Dev 1 checks only that its `frame_id` matches the mask and that `K` is not a placeholder. |
| Costmap cost semantics | Available: Dev 3 `class_to_cost`, `geometry_costmap`, `costmap_fusion`. |

## D. Temporary or synthetic inputs

These exist only as **test values** in `tests/`, chosen to make expected cells
easy to work out by hand:

- camera intrinsics (e.g. `fx = fy = 10`)
- camera height and pitch (e.g. 1 m, pitched straight down)
- grid resolution, origin and size
- occupancy masks
- footprints and inflation radius
- all frame names (`test_camera_frame`, `test_ground_frame`, …)
- stamps and max ages

PROJECT_CONTEXT.md §6 lists **temporary** frame names (`base_link`,
`camera_optical_frame`, `map`). They are not confirmed and are not
hard-coded here. They belong in configuration once the ROS adapter exists.

## E. Inputs that still depend on Dev 2 / Dev 5 (or are unresolved)

| Needed by Dev 3 | Owner | What is missing |
|---|---|---|
| Real intrinsics (fx, fy, cx, cy, image size) | Dev 5 camera driver; intrinsics YAML in Dev 2 `config/cameras/` | No calibration exists yet. |
| Camera → base_link extrinsic (mount height, pitch, offset) | Dev 5 robot description (URDF camera extrinsics) | No values exist yet. |
| base_link ↔ odom ↔ map transforms | Dev 2 TF chain | No TF source exists yet. |
| Robot footprint polygon(s) | Dev 5 (primary + secondary footprint YAMLs) | No footprint exists yet. |
| Geometric occupancy | Depth Anything is listed under Dev 1. The VoxelLayer / occupancy producer and its format are **undefined**. | No producer, message type or frame. Dev 1's interface doc describes a `DepthFrame` whose unit may be `"relative"`, which cannot produce metric occupancy by itself. |
| Costmap frame, resolution, extent | Dev 3 config (`config/robots/`, shared with Dev 4) | Not decided. |
| Max age / stamp tolerance for Dev 3 | Team decision | Dev 1's `perception_max_age` is Dev 1's own gate. No Dev 3 value is agreed. |

## F. What must eventually be ROS adapters

Each adapter translates ROS data into one contract dataclass. None of them
exist yet, and none should be written until the upstream interfaces above
are confirmed and the installed ROS 2 / Nav2 APIs have been checked
(PROJECT_CONTEXT.md §11).

1. **Mask adapter**: `/segmentation/mask` (`Image`, mono8) +
   `/segmentation/port_meta` → `SemanticMaskInput`. It decodes `data`/`step`,
   converts the header stamp to ns and takes `frame_id` and `valid`.
2. **CameraInfo adapter**: `CameraInfo` → `CameraIntrinsicsInput`. It takes
   K[0], K[4], K[2], K[5], width, height and frame_id, and must decide how to
   handle `D` (§8).
3. **TF adapter**: a TF lookup from the camera optical frame to the costmap
   frame at the mask stamp → `CameraGroundInput`. This needs either a
   reduction of the full transform to height + pitch (valid only when
   roll/yaw ≈ 0 and the camera is over the ground-frame origin) or a
   6-DoF-capable projection core.
4. **Grid/parameter adapter**: ROS parameters / `config/robots/` →
   `GridInput`, plus the Dev 3 tuning values.
5. **Geometry adapter** (optional): the future depth/voxel output →
   grid-rasterised `OccupancyInput`.
6. **Footprint adapter**: Dev 5 footprint YAML / Nav2 footprint parameter →
   `FootprintInput`.
7. **Output adapter**: final cost array + grid → `nav_msgs/OccupancyGrid` on
   `/local_costmap/costmap` / `/global_costmap/costmap`, or a Nav2 costmap
   layer plugin. This requires a mapping from core costs (0–255, Nav2
   costmap_2d convention) to the output representation. That mapping is not
   defined yet.

## G. Data flow through the current pipeline

```
SemanticMaskInput.classes ─┐
CameraIntrinsicsInput ─────┤  mask_projection.project_mask_to_costmap
CameraGroundInput ─────────┤    per pixel: projection.pixel_to_camera_ray
GridInput ─────────────────┘             → projection.camera_ray_to_ground_point
                                         → grid.world_to_grid_cell
                                         → class_to_cost (per-cell precedence
                                           HAZARD > UNKNOWN > TRAVERSABLE;
                                           unseen cells = unknown cost)
                                  = semantic costmap (grid-shaped)
                                              │
OccupancyInput.occupied ──► geometry_costmap.build_geometry_costmap
                                  = geometry costmap (grid-shaped)
                                              │
                            costmap_fusion.fuse_costmaps
                              (geometry lethal ALWAYS wins)
                                              │
                            inflation.inflate_costmap(resolution, radius)
                                              │
                                     final cost array

FootprintInput.vertices ──► footprint.validate_footprint / pad_footprint
                            (standalone; not yet wired into inflation)
```

`semantic_costmap.build_semantic_costmap` (mask → same-shape costmap, with no
projection) is still available but is not on the projected path.

### Entry point — `pipeline.run_costmap_pipeline`

All of the above runs in a single call. `pipeline.py` only wires the existing
stage functions together and contains no stage logic of its own. All values
come from the caller; none are defaults:

```python
inputs = CostmapCoreInputs(mask=..., intrinsics=..., camera_ground=..., grid=...,
                           occupancy=...,   # optional
                           footprint=...)   # optional
result = run_costmap_pipeline(inputs,
                              inflation_radius=...,     # required, metres, >= 0
                              footprint_padding=...)    # optional, needs a footprint
result.final  # costmap to publish; also .semantic, .geometry, .fused, .footprint
```

- `occupancy=None`: then `result.geometry is None` and `result.fused` is
  `result.semantic`.
- `footprint`: returned in `result.footprint`, padded when
  `footprint_padding` is given. It is **not** applied to any cell.
- `geometry_cost_values.lethal_cost` is the lethal value used by both fusion
  and inflation.
- All result arrays are read-only (`int64`, grid-shaped). Inputs are never
  mutated.
- It raises `PipelineError` if `mask.valid` is `False`. The §8.6 fail-safe
  is left to the caller. It also raises if `footprint_padding` is given
  without a footprint, or if `inputs` is not a `CostmapCoreInputs`.
  Radius and padding errors propagate as `InflationError` / `FootprintError`.
- It does **not** check freshness or stamp alignment (no agreed max age or
  tolerance). Call `is_fresh` before it.

## 8. Known limitations and open questions

- **Distortion ignored.** The projection is pure pinhole. Dev 1's node
  subscribes to `/camera/image_raw` by default, so the mask may be on an
  unrectified image. Open question: is the mask rectified, or must Dev 3
  undistort pixel coordinates?
- **Simplified camera pose.** The core assumes height + pitch only, with the
  camera over the ground-frame origin. A real TF-derived pose with lateral or
  forward offset, roll or yaw cannot be represented without extending
  `projection.py`.
- **Flat ground.** Ground is assumed to be the z = 0 plane of the costmap
  frame.
- **Stale / invalid masks.** Architecture §8.6 calls for a "front ROI
  lethal / max-inflate" fail-safe on stale or invalid masks. This is **not**
  implemented. The contract only exposes `valid` and `is_fresh`. The ROI
  extent is undefined.
- **Stamp alignment** between mask, camera pose and occupancy is not
  enforced, because no tolerance is agreed.
- **Footprint** is validated and padded but not yet connected to inflation
  (no inscribed/circumscribed radius logic).
- **`/ugv/pose_valid` and `/ugv/perception_degraded`** are specified as Dev 5
  inputs, not Dev 3 inputs. The contract therefore does not include them.
  Dev 3 relies on `mask.valid`, stamp freshness and TF lookup success.
- **Dev 1 confidence** (`/segmentation/confidence`, 32FC1) is published but is
  not used by the core and is not part of the contract.
