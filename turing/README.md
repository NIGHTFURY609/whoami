# Turing — Dev 1 workspace

**Owner:** Dev 1 (Perception, AI & Vision)  
**Authority:** [`architecture.md`](../architecture.md) wins over [`dev.md`](../dev.md) on any conflict.  
**Product path:** live outdoor camera → Perception Port. Not a demo, not a bag player, not a model.

This folder splits Dev 1 into **independently buildable, independently testable modules**. Build in order. Do not skip a module by stubbing it with fake vision.

## What Dev 1 is

Dev 1 is **not** the brain. The brain is RTAB-Map + Nav2 (Dev 2 / Dev 4).

Dev 1 owns:

1. **The Perception Port** — the only semantic contract the rest of the product is allowed to see.
2. **Adapters** that implement that port (YOLOE outdoor default; tutorial ONNX is eval-only).
3. **Optional geometry side-channel** (Depth Anything) that is **not** the port.

Downstream (Dev 3 costmaps, Dev 5 safety) bind only to port outputs. They never import YOLOE classes, ONNX labels, or raw model scores.

## Build order

```
T01 types + validators     pure contract, no camera
T02 consume Image+CameraInfo  decode + ros_bridge **shipped**; outdoor stream still Dev 5
T03 remap engine           YAML, no model
T04 confidence gates       YAML, no model
T05 freshness / degraded   time + flags, no model
T12 backend seam           OpenVINO GPU on Arc B580; CUDA PyTorch later (low VRAM)
T06 YOLOE adapter          first real masks  ← IR on disk; uses T12
T07 port node              compose + publish canonical mask + degraded
T10 contract tests         T01–T07 wired: encoding, {0,1,2}, stamp, frame, degraded
T11 performance / latency  capture→mask, FPS, bounded queue  ← prefer real camera
T08 Depth Anything         optional geometry, parallel to port  ← needs weights
T09 tutorial ONNX          eval scaffold only, not product
```

**Implemented (2026-09-18):** T01–T05 kernels, T02 decode+ros_bridge, T06 pack, T07 `compose_tick` + Lyrical `adapter_node`, T12 seam.  
**On disk:** YOLOE-26s OpenVINO IR (`weights/yoloe-26s-seg.xml`). RUGD SegFormer-B5 OpenVINO IR (`weights/rugd-segformer.xml`, 25 RUGD classes, Arc `device=GPU`). Live adapter remains YOLOE. **No DummySource.** Outdoor product `infer` still needs a Dev 5 Image+CameraInfo stream.  
T10 uses header/label fixtures. T11 policy (latest-only queue + starve watchdog) **shipped**; live p95 waits on Dev 5. Outdoor camera is Dev 5.

Runtime: Intel Arc B580 **now** (OpenVINO 2026.4.0, `device=GPU`). ROS 2 **Lyrical** on this RHEL 10 box. Later NVIDIA, less VRAM (CUDA + PyTorch). See [HARDWARE.md](HARDWARE.md).

### T07 port node (explicit)

- Compose ingestion + YOLOE + remap + confidence + freshness
- Publish the canonical product mask
- Publish `perception_degraded`
- Preserve original sensor timestamp
- Preserve optical `frame_id`
- No model-specific logic downstream (no YOLOE/OpenVINO/TensorRT in this node)

## Module graph

```
                    [T02 Source]
                    ImageFrame (real)
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   [T06 YOLOE adapter]     [T08 Depth Anything]
   RawSemOutput            DepthFrame
          │                 (NOT the port)
          ▼
   [T03 Remap] ── YAML ontologies
          │
          ▼
   [T04 Conf gates] ── YAML profile
          │
          ▼
   [T05 Freshness]
          │
          ▼
   [T07 Perception Port]          [T01 validators on every publish]
          │                       [T10 contract tests on the wire]
          │                       [T11 latency / backpressure]
          ├─ /segmentation/mask            {0,1,2} mono8
          ├─ /segmentation/confidence      [0,1] 32FC1
          ├─ /segmentation/port_meta       valid, age (informational), scale
          └─ /ugv/perception_degraded      Bool → Dev 5
```

T06 calls `InferenceBackend` (T12). T07 never does.  
T09 plugs in at the same adapter seam as T06. It must not ship as the live_cam default.

## Laws that every task inherits

Copied from architecture; not optional:

| Law | Meaning for Dev 1 |
|---|---|
| Adapters are sources, not the brain | YOLOE never talks to Nav2 or `/cmd_vel` |
| 3-class port only | pixels ∈ `{0,1,2}` — no `cautious` |
| No remap → no publish | adapter without YAML is a hard error |
| Normalize then gate | τ_* apply in `[0,1]`, never on raw multi-model scores |
| Stale ≠ current | age > `perception_max_age` → degraded, do not present mask as live |
| Unknown ≠ free | class 0 inflates; never “fill unknown with traversable” |
| Geometry is a side-channel | Depth Anything does not write `/segmentation/mask` |
| Front ROI lethal on fail | Dev 1 publishes degraded; **Dev 3** inflates the costmap. We do not write costmaps. |

## Data policy (product)

**Forbidden:** synthetic images, simulator cameras as product perception, random masks, constant “all traversable” stubs, fake `CameraInfo`, recycled stamps, downloaded demo clips presented as this UGV’s camera.

**Allowed:** live calibrated camera, outdoor recordings from that (or the target) camera, pretrained weights, YAML we author, numeric tables for remap/gate unit tests (label→id, score→class). Numeric tables are contract tests, not perception data.

If a task cannot be tested without a fake camera, the task is not done — get a real source (T02) first.

## Files

| Path | What |
|---|---|
| [00-role-and-laws.md](00-role-and-laws.md) | Dev 1’s place in the product |
| [interfaces.md](interfaces.md) | Internal APIs between modules |
| [DATASETS.md](DATASETS.md) | Weights / datasets — **read this, action required** |
| [HARDWARE.md](HARDWARE.md) | Arc B580 OpenVINO GPU; ROS Lyrical; YOLOE-26s; no camera device |
| [CONFLICTS.md](CONFLICTS.md) | Where `dev.md` is ignored |
| [tasks/](tasks/) | T01–T12 build checklists |
| [subarchs/subarch1.md](subarchs/subarch1.md) | T01 port-kernel architecture (wins over the T01 checklist) |
| [subarchs/subarch2.md](subarchs/subarch2.md) | T02 consume-camera-data architecture (not the device; wins over V4L2 T02) |
| [subarchs/subarch3.md](subarchs/subarch3.md) | T03 remap-kernel architecture (wins over the T03 checklist) |
| [subarchs/subarch4.md](subarchs/subarch4.md) | T04 confidence-gate architecture (wins over the T04 checklist) |
| [subarchs/subarch5.md](subarchs/subarch5.md) | T05 freshness/degraded-policy architecture (wins over the T05 checklist) |
| [subarchs/subarch6.md](subarchs/subarch6.md) | T06 YOLOE adapter architecture (wins over the T06 checklist) |
| [subarchs/subarch7.md](subarchs/subarch7.md) | T07 port-composition architecture (wins over the T07 checklist) |
| [subarchs/subarch10.md](subarchs/subarch10.md) | T10 wired port-contract tests (wins over the T10 checklist; T08 still deferred) |
| [subarchs/subarch11.md](subarchs/subarch11.md) | T11 latency / latest-only queue (wins over the T11 checklist; T08 still deferred) |
| [subarchs/subarch12.md](subarchs/subarch12.md) | T12 OpenVINO GPU seam (matches shipped T06 Protocol; wins over old T12 checklist) |

Code for each task lands under `turing/src/` when we implement. Do not start T06/T08 until weights are on disk (see DATASETS.md).
