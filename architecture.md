# Project A — UGV Camera-Primary Nav Architecture
**Status:** PASS (hard) + v1 safety/port contract tighten · **Scope:** software-only **deployable product**

> This document is **stack + architecture** for a full working product (live outdoor camera → A→B).  
> Sim worlds, tutorial models, and datasets are **eval / bring-up profiles**, not the product’s fixed data.  
> Hour splits across people are owned outside this file. Soft effort: **medium limit ~30h per person** (aim to hit; stretch only for completeness) — not a hard no-go kill; **not** a whole-team total. Owner-locked 2026-09-17.

## 1. Product goal
Deliver an installable ROS 2 system that, on a differential-drive UGV with a calibrated vision sensor, navigates **Point A → Point B** in GPS-denied outdoor settings using vision as the primary sensor: path vs hazard perception, visual localization, and collision-aware planning to `/cmd_vel`.

| Challenge | Product capability |
|---|---|
| Path detection | Pluggable perception **source** → stable mask port → costmap (+ fail-safe) |
| Visual localization | RTAB-Map VO/SLAM, mapping vs localize-on-map modes |
| Collision avoidance → goal | Nav2 plan + control → `/cmd_vel` (via safety authority) |

**Not the product:** a Baylands-only demo, hard-coded TB4, tutorial-as-only-ontology, bag-only playback, mono USB marketed as outdoor-meter-ready, or a stack whose **brain** is a single CV model (e.g. YOLOE).

## 2. One-liner
`live camera ─┬─► adapters → Perception Port → Nav2 costmaps`  
`             └─► RTAB-Map ──────────────────────► Nav2`  
`                                              ↓`  
`                                    Safety authority → /cmd_vel`

**Brain** = RTAB-Map + Nav2. **Adapters** = sources behind the port. **Final `/cmd_vel` authority** = safety boundary (§3.1), not the planner alone.

## 3. Source vs brain vs safety (design law)
| Role | What | Examples |
|---|---|---|
| **Brain** | Pose, map, planning, control *candidates* | RTAB-Map, Nav2 |
| **Perception Port** | Canonical mask + port-normalized confidence + freshness | `/segmentation/mask` (+ meta) |
| **Adapters** | Implement the port | YOLOE, tutorial ONNX, future seg |
| **Safety authority** | **Final** gate on what reaches the base | Priority mux / watchdog → `/cmd_vel` |

### 3.1 `/cmd_vel` authority (explicit)
**Final authority over `/cmd_vel` is the safety authority** (priority mux or equivalent), not Nav2’s controller by itself.

Precedence (highest wins):
1. **E-stop / operator kill** → zero twist  
2. **System-health fail** (camera / perception / localization / TF / Nav2 timeouts — §12) → zero twist  
3. **Perception-degraded or VO-lost / invalid pose** → zero twist (hold)  
4. **Nav2 controller output** → commanded twist  

Nav2 may *compute* motion; it does **not** bypass the safety authority. One publisher owns the base `/cmd_vel` topic (or a documented mux with the order above).

## 4. Runtime profiles
| Profile | Purpose |
|---|---|
| **`live_cam` (default)** | Real outdoor deploy |
| `sim` | Integration / CI |
| `bag` / `rugd` | Offline outdoor checks |

Baylands / RUGD / tutorial ONNX = eval/scaffold only.

## 5. Context diagram
```
                    [Vision sensor]
              recommended: stereo/RGB-D · minimum: mono
                           │
              ┌────────────┴────────────┐
              │ dual fan-out (parallel) │
              ▼                         ▼
   ┌────────────────────┐     ┌────────────────────┐
   │ Adapters (sources) │     │ RTAB-Map           │
   │ RUGD SegFormer     │     │ pose + map         │
   │ remap→canonical    │     │ mapping|localize   │
   │ normalize conf     │     │ (does NOT use mask)│
   └─────────┬──────────┘     └─────────┬──────────┘
             ▼                          │
  ┌─────────────────────┐               │
  │  PERCEPTION PORT    │               │
  │  classes + conf     │               │
  │  stamp/frame/age    │               │
  └──────────┬──────────┘               │
             ▼                          │
  ┌────────────────────┐                │
  │ Nav2 costmaps      │◄───────────────┘
  │ SemanticLayer +    │
  │ optional VoxelLayer│◄── optional Depth Anything (geometry)
  └─────────┬──────────┘
            ▼
  ┌────────────────────┐
  │ Nav2 Smac2D + RPP  │──► cmd candidate
  └─────────┬──────────┘
            ▼
  ┌────────────────────┐
  │ SAFETY AUTHORITY   │──► /cmd_vel → any diff-drive
  │ (final cmd gate)   │
  └────────────────────┘
```

## 6. Tech stack
| Layer | Choice | Notes |
|---|---|---|
| Middleware | ROS 2 Jazzy (Humble OK) | |
| Vision recommended / minimum | Stereo·RGB-D / mono | §10 |
| **Brain** | RTAB-Map + Nav2 (Smac2D + RPP) | |
| **Perception Port** | Canonical mask + conf + freshness + frame | §8 |
| Adapter default outdoor | RUGD SegFormer-B5 | OpenVINO `device=GPU`. 25 RUGD classes remapped to {0,1,2}. YOLOE stays selectable, not live. |
| Adapter scaffold | Tutorial ONNX | Remapped |
| **Safety authority** | Priority mux / watchdog | §3.1 · §12 |
| Optional geometry | Depth Anything → VoxelLayer | Geometry side-channel only; conflict rules = §9 (geometry lethal wins; semantic never clears it) |

## 7. Package layout
```
ugv_nav/
  ugv_perception/     # port + adapters + remap + conf_normalize
  ugv_localization/   # RTAB-Map + pose validity signal
  ugv_navigation/     # Nav2
  ugv_safety/         # cmd mux + health timeouts → hold
  ugv_bringup/
  ugv_robot_description/
  ugv_eval/
  config/{cameras,robots,ontologies,perception,safety}/
  docs/
```

## 8. Perception Port contract

### 8.1 Canonical classes (**v1 — no optional IDs**)
| ID | Name | Cost intent |
|---|---|---|
| 0 | `unknown` | never free — inflate |
| 1 | `traversable` | free / low |
| 2 | `hazard` | lethal / inscribed |

**v1 drops `cautious`.** Ambiguous soft cases map to `unknown` (inflate) until a later revision promotes a fourth class. Brain/costmap bind **only** to these three IDs.

### 8.2 Adapter → canonical remapping (mandatory)
Each adapter ships remap YAML (YOLOE prompts / ONNX sidewalk|grass|background → {0,1,2}). No remap → must not publish on the port.

### 8.3 Confidence normalization (mandatory)
Gates (τ_trav, τ_haz, τ_min, κ) apply in **port-normalized** `[0,1]` via per-adapter profiles. Raw multi-model scores are not shared.

### 8.4 Freshness / validity (mandatory)
Every port message includes at least:
- `header.stamp` (image time)  
- `header.frame_id` (optical / camera frame of the mask)  
- **age** = now − stamp (or equivalent); consumer rejects if age > `perception_max_age`  
- **valid** flag (false if the adapter failed or the sample is stale). Per-pixel gates map unsure pixels to class `0`; a mostly- or all-unknown mask is still a legal **current** sample.

**Stale mask must not be treated as current.** Exceeding max age → perception-degraded → safety hold (§3.1).

### 8.5 Spatial contract (v1 minimum)
Costmap consumers require:
- Mask `frame_id` + matching `CameraInfo` for that camera  
- TF tree connecting camera frame → `base_link` / costmap frame  
- Documented mask resolution (same as source image or stated scale)  
- Stamp aligned to the image used (no silent reuse of an older frame)

Detailed projection math / multi-camera sync refinements = **deferred** (§14) if TF + CameraInfo + stamp/age already hold.

### 8.6 Port fail-safe
Stale or adapter failure → `/ugv/perception_degraded` + front ROI lethal/max-inflate + safety hold. Per-pixel gate miss is class `0` on that pixel, not a dropped mask. **Unknown ≠ free** (inflate `0`; never treat it as class `1`).

## 9. Semantic vs geometry precedence
Optional Depth Anything → VoxelLayer is a **geometry side-channel**, not a second perception port.

**Conflict rule:** if geometry says occupied/lethal and semantic says `traversable`, **geometry wins** (lethal stays). Semantic “safe” must **never** clear a geometric obstacle. Semantic hazards may add cost on top of free geometry.

## 10. Localization & autonomy
**Sensor honesty:** stereo/RGB-D recommended; mono minimum + drift docs + VO-lost hold.  
**Modes:** `mapping` (build/save) vs `localize` (load + NavigateToPose).

### 10.1 Pose validity (not only VO-lost)
Nav2 may use pose only if **valid**:
- Required TFs present  
- Pose age < `localization_max_age`  
- RTAB-Map (or wrapper) status not lost / not exploded  

Else → invalid pose → safety hold. VO-lost is one way to become invalid; it is not the only way.

## 11. Planning & avoidance (v1)
Smac2D + RPP; map-frame goals; degraded perception or invalid pose → no motion through safety authority.  
**v1 dynamic obstacles:** moving hazards are handled by **continuous live mask (+ optional voxel) updates** into the costmap; planner/controller react through normal Nav2 costmap refresh. A richer dynamic-obstacle track/predict contract is **deferred** (§14).

## 12. System health → safe stop
Not a giant Health Monitor — a **minimum timeout table**. Any trip → safety authority zeros `/cmd_vel`:

| Watch | Fail when | Action |
|---|---|---|
| Camera | no image / age > limit | hold |
| Perception port | no valid fresh mask / degraded | hold |
| Localization | invalid pose / VO-lost / TF missing | hold |
| TF | required frames missing | hold |
| Nav2 | crash / no controller heartbeat | hold |
| E-stop | asserted | hold |

## 13. Definition of Done
1. `live_cam` end-to-end documented  
2. Path/hazard via **3-class canonical port** + freshness/frame meta  
3. Adapter swap via remap + conf profile only  
4. Safety authority proven: estop / stale perception / invalid pose → zero `/cmd_vel`  
5. Geometry overrides semantic traversable when in conflict (if voxel enabled)  
6. Mapping save + localize A→B  
7. Sensor honesty documented  
8. Outdoor run or high-fidelity bag through full stack  
9. Second footprint YAML  
10. Eval profiles regression-only  

## 14. Deferred (valid, not required for initial product)
Parked because they can be added later without rewriting the brain/port, and would slow v1:

| Item | Why later |
|---|---|
| Rich **dynamic-obstacle** track/predict/ID contract | v1 live mask→costmap refresh is enough to ship |
| Deep multi-camera **projection/sync** specification | v1: CameraInfo + TF + stamp/age |
| Canonical **`cautious` class** | Reintroduce when product needs soft costs; v1 uses `unknown` |
| Full **Health Monitor** subsystem / UI | v1: timeout table + safety mux |
| Learned adapter conf calibration tooling | v1: manual profiles in YAML |

## 15. Risks → mitigations
| Risk | Mitigation |
|---|---|
| Nav2 treated as final cmd authority | §3.1 safety precedence |
| Stale masks acted on | §8.4 freshness |
| YOLOE as brain | Port + adapters |
| Semantic clears real obstacles | §9 geometry wins |
| Weak adapter outdoors | Honesty + fail-safe |
| Medium limit ~30h/person vs completeness | Completeness can stretch past 30h; prefer staying near the limit |

## 16. Kill list
CARLA-as-product · Baylands-only · TB4-hardcoded topics · YOLOE-as-brain · adapter publish without remap · shared τ on raw multi-model conf · unknown-as-free · stale-mask-as-current · semantic override of lethal geometry · GPS missions · motor firmware · ORB-SLAM3 bake-off · mono = recommended outdoor RTAB-Map

## 17. Assumptions
Software-only; `/cmd_vel` base; calibrated vision; medium limit **~30h per person**. **No ROS coding until owner explicitly says go.**

## 18. Next
1. Snow confirm this v1 contract tighten (safety / freshness / spatial min / no cautious / geometry precedence / health table)  
2. Flame brief realign if Snow confirms  
3. Refresh Notion + Gmail after confirm  
4. **No ROS coding** until owner says go  

Authority: `/workspace/project_A/architecture.md`
