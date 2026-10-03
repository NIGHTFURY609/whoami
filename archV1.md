# Project A — UGV Camera-Primary Nav Architecture (V1 add1)
**Status:** addition on [`architecture.md`](architecture.md) · **Scope:** software-only **deployable product** + localized destination  
**Reload this path:** `/home/light/sih/archV1.md` — if the first heading is still “V1 add1 — Localized goal” with Dev1/Dev2 paste, that tab is a stale buffer.

> Base contract is [`architecture.md`](architecture.md). Port `{0,1,2}`, freshness, geometry precedence, `/cmd_vel` mux, kill list, and §6 stack are unchanged.  
> This file is the **same architecture with the localized-goal add-on drawn in**. It is not a second product and not a replacement brain.

## 1. Product goal
Deliver an installable ROS 2 system that, on a differential-drive UGV with a calibrated vision sensor, navigates **Point A → Point B** in GPS-denied outdoor settings using vision as the primary sensor: path vs hazard perception, visual localization, and collision-aware planning to `/cmd_vel`.

**Add-on:** Point B may be given as a **localized destination** relative to a mission start (example: **50 m at 20° from the starting pose**). B does not have to be in the camera, on a saved map, or already in the costmap. The robot maps and updates cost on the way.

| Challenge | Product capability |
|---|---|
| Path detection | Pluggable perception **source** → stable mask port → costmap (+ fail-safe) |
| Visual localization | RTAB-Map VO/SLAM; mission **reference pose** in `map`; live TF |
| Collision avoidance → goal | Nav2 plan + **replan** as costmaps grow → `/cmd_vel` (via safety authority) |
| Unseen / unmapped B | Relative goal → `map`-frame pose → Nav2; unknown cells stay inflated |

**Not the product:** GPS waypoint missions, unknown-as-free shortcuts to a far pose, Dev 1 picking B, Nav2 as final `/cmd_vel`, YOLOE-as-brain.

## 2. One-liner
```
mission {range, bearing} from start
        │
        ▼
live camera ─┬─► adapters → Perception Port → Nav2 costmaps ─┐
             └─► RTAB-Map (reference pose + live TF) ─► Nav2 ─┤
                                                              ▼
                                                    Safety authority → /cmd_vel
```

**Brain** = RTAB-Map + Nav2. **Adapters** = sources behind the port. **Localized goal** = Dev 4 feature on a Dev 2 `map`-frame pose. **Final `/cmd_vel` authority** = safety boundary (`architecture.md` §3.1).

## 3. Source vs brain vs safety (design law)
| Role | What | This add-on |
|---|---|---|
| **Brain** | Pose, map, planning, control *candidates* | RTAB-Map holds the reference pose; Nav2 holds `/navigate_to_pose` |
| **Perception Port** | Canonical mask + conf + freshness | Unchanged. Feeds costmaps while the robot moves |
| **Adapters** | Implement the port | Live outdoor: RUGD SegFormer-B5. YOLOE selectable. Tutorial ONNX eval |
| **Safety authority** | **Final** gate on what reaches the base | Unchanged precedence |

### 3.1 `/cmd_vel` authority (explicit)
Same as `architecture.md` §3.1. Highest wins: e-stop → health fail → perception-degraded / invalid pose → Nav2 candidate. Nav2 does **not** publish base `/cmd_vel`.

## 4. Runtime profiles
| Profile | Purpose |
|---|---|
| **`live_cam` (default)** | Real outdoor deploy, including localized-goal runs |
| `sim` | Integration / CI |
| `bag` / `rugd` | Offline outdoor checks |

Baylands / RUGD / tutorial ONNX = eval/scaffold only.

## 5. Context diagram (base + add-on)

```
                    [Vision sensor]
              recommended: stereo/RGB-D · minimum: mono
                           │
              ┌────────────┴────────────┐
              │ dual fan-out (parallel) │
              ▼                         ▼
   ┌────────────────────┐     ┌────────────────────┐
   │ Adapters (sources) │     │ RTAB-Map           │◄── [Mission]
   │ RUGD SegFormer-B5  │     │ pose + map         │    {range, bearing}
   │ remap→canonical    │     │ reference pose     │    from start
   │ normalize conf     │     │ mapping-while-go   │
   └─────────┬──────────┘     │ (does NOT use mask)│
             ▼                └─────────┬──────────┘
  ┌─────────────────────┐               │
  │  PERCEPTION PORT    │               │ live TF / pose_valid
  │  {0,1,2} + conf     │               │ + map-frame PoseStamped goal
  │  stamp/frame/age    │               │
  └──────────┬──────────┘               │
             ▼                          │
  ┌────────────────────┐                │
  │ Nav2 costmaps      │◄───────────────┤
  │ SemanticLayer +    │   grow as new terrain is seen
  │ optional VoxelLayer│◄── Depth Anything 3 Metric Large (also feeds RTAB-Map depth)
  └─────────┬──────────┘                │
            ▼                           │
  ┌────────────────────┐                │
  │ Nav2 Smac2D + RPP  │◄───────────────┘
  │ replan as cost     │   map-frame goal (may start off-map)
  │ maps fill          │──► cmd candidate
  └─────────┬──────────┘
            ▼
  ┌────────────────────┐
  │ SAFETY AUTHORITY   │──► /cmd_vel → any diff-drive
  │ (final cmd gate)   │
  └────────────────────┘
```

One RTAB-Map. Camera still dual-fans to adapters and RTAB-Map. Mission `{range, bearing}` enters RTAB-Map; the converted `map`-frame goal enters **Nav2 Smac2D + RPP**, not the camera.

### 5.1 Who owns the new boxes
```
Mission ──► Dev 2 (reference pose + convert to map)
                │
                ▼
            Dev 4 (Nav2 goal + replan) ◄── Dev 3 (live costmaps)
                                              ▲
                                              │
                                          Dev 1 (port + DA3)
                │
                ▼
            Dev 5 (safety → /cmd_vel)
```

Dev 1, Dev 3, and Dev 5 keep their existing jobs. Dev 2 adds the reference-frame conversion. Dev 4 owns the localized-goal feature.

## 6. Tech stack
Same as `architecture.md` §6. This add-on does not swap adapters or engines.

| Layer | Choice | Notes |
|---|---|---|
| Middleware | ROS 2 Lyrical | |
| Vision recommended / minimum | Stereo·RGB-D / mono | `architecture.md` §10. Current hardware: mono + DA3 pseudo-depth (large tier) |
| **Brain** | RTAB-Map + Nav2 (Smac2D + RPP) | Goal may be off the current map |
| **Perception Port** | Canonical mask + conf + freshness + frame | `architecture.md` §8 |
| Adapter default outdoor | RUGD SegFormer-B5 | Intel OpenVINO GPU, CPU fallback. NVIDIA CUDA PyTorch (no OpenVINO). YOLOE selectable, not live |
| Adapter scaffold | Tutorial ONNX | Eval only |
| **Safety authority** | Priority mux / watchdog | `architecture.md` §3.1 · §12 |
| Depth / geometry | Depth Anything 3 Metric Large (Dev 1) → RTAB-Map RGB-D (required); → VoxelLayer (optional) | §9: geometry lethal wins |

## 7. Package layout
```
ugv_nav/
  ugv_perception/     # port + adapters + remap + conf_normalize   (Dev 1)
  ugv_localization/   # RTAB-Map + pose validity + reference pose  (Dev 2)
  ugv_navigation/     # costmaps (Dev 3) + Nav2 localized goal (Dev 4)
  ugv_safety/         # cmd mux + health timeouts → hold           (Dev 5)
  ugv_bringup/
  ugv_robot_description/
  ugv_eval/
  config/{cameras,robots,ontologies,perception,safety}/
  docs/
```

## 8. Perception Port (unchanged)

Canonical IDs stay `{0 unknown, 1 traversable, 2 hazard}`. Unknown ≠ free. Stale mask → `/ugv/perception_degraded` → hold. DA3 is a geometry side-channel for costmaps (VoxelLayer optional) and also feeds RTAB-Map depth (`architecture.md` §10); semantic never clears lethal geometry.

While the robot drives toward B, Dev 1 keeps publishing the same port (and optional depth). That is how previously unknown ground becomes cost. Dev 1 does **not** own B.

Full contract: `architecture.md` §8–§9.

## 9. Localization & the reference pose

**Sensor honesty** and pose validity stay `architecture.md` §10 / §10.1.

**Modes:**

| Mode | Use |
|---|---|
| `mapping` | Build/save a map |
| `localize` | Load a map + NavigateToPose on known space |
| **`mapping` + localized goal (this add)** | Freeze start pose in `map`, convert `{range, bearing}` → `PoseStamped`, drive while the map and costmaps grow |

At mission start Dev 2 records the **reference pose** in `map`. Example: 50 m at 20° from that pose becomes one `map`-frame goal. Robot pose and goal stay in the same frame. TF `map→odom→base_link` and `/ugv/pose_valid` keep updating.

Dev 2 does **not** consume the mask and does **not** call Nav2. Invalid pose → no conversion / no Nav2 use of that pose → Dev 5 hold.

## 10. Costmaps in unknown territory

Dev 3 keeps SemanticLayer + optional VoxelLayer. New mask (and optional voxels) enter the existing pipeline as the robot moves.

- Class `0` inflates. Never treated as free to “reach” a far B.
- Geometry lethal still wins over semantic traversable.
- Dev 3 does **not** set `/navigate_to_pose`.

## 11. Planning (localized goal)

Smac2D + RPP. Dev 4 **owns** the feature: take the `map`-frame pose, send `/navigate_to_pose` (or equivalent), replan as Dev 3’s costmaps fill.

The goal may start outside the currently observed / mapped costmap. That does **not** license unknown-as-free. Degraded perception or invalid pose → no motion through the safety authority.

Candidate twist remains `/cmd_vel_nav2`.

## 12. System health → safe stop

Same timeout table as `architecture.md` §12. Six watches; any trip zeros `/cmd_vel`:

| Watch | Fail when | Action |
|---|---|---|
| Camera | no image / age > limit | hold |
| Perception port | no valid fresh mask / degraded | hold |
| Localization | invalid pose / VO-lost / TF missing | hold |
| TF | required frames missing | hold |
| Nav2 | crash / no controller heartbeat | hold |
| E-stop | asserted | hold |

Localized-goal travel does not add a seventh watch and does not weaken these six.

## 13. Definition of Done (this add, on top of `architecture.md` §13)
1. Relative `{range, bearing}` from a recorded start pose becomes a `map`-frame goal  
2. Nav2 accepts that goal when B is not yet in the camera or the costmap  
3. Costmaps ingest new port (and optional voxel) data on the way; unknown stays inflated  
4. Replan occurs as the map/costmap grow  
5. Safety holds still fire (stale mask, invalid pose, TF miss, estop)  
6. No GPS; no Dev 1 / Dev 3 goal ownership  

## 14. Risks → mitigations
| Risk | Mitigation |
|---|---|
| Treat unseen B as free | Unknown inflates; geometry lethal wins |
| Dev 1 or Dev 3 owns B | Dev 4 owns the Nav2 goal; Dev 2 only converts frame |
| Nav2 as final cmd | `architecture.md` §3.1 |
| Drift while mapping-to-B | Pose validity + hold |
| GPS sneak-in | Kill list |

## 15. Kill list
Everything in `architecture.md` §16, plus: GPS localized goals · unknown-as-free to reach B · Dev 1 generating B · Dev 3 calling Nav2 · skipping `/ugv/pose_valid` for the conversion

## 16. Assumptions
Software-only; `/cmd_vel` base; calibrated vision. Engine/adapter choice remains `architecture.md` §6. **No ROS coding until owner explicitly says go.**
