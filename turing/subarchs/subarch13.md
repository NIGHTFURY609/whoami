# Sub-architecture 13 — Transition: YOLOE → GA-Nav (terrain adapter)

**Status:** decode + ontology **shipped** (scripted logits, no weights). PyTorch eval and OpenVINO export **not started** — you download the checkpoint. Default live adapter stays YOLOE.

**Authority:** [`architecture.md`](../../architecture.md) §3, §6, §8 (port unchanged). [`HARDWARE.md`](../HARDWARE.md) OpenVINO GPU, Arc B580, later NVIDIA **less** VRAM.  
**Does not replace:** T01–T05, T07, T10, T11. Those stay.  
**Replaces only:** T06 outdoor **source** (YOLOE instance adapter) with a **terrain segmenter** adapter that still emits `RawSemOutput`. T12 still only **compiles and runs** the IR (raw tensors). `GanavAdapter` owns GA-Nav tensor decode.  
**T08** stays deferred.

If this file and `architecture.md` disagree on topics, encodings, `{0,1,2}`, or `/cmd_vel`, **architecture.md wins**.

---

## Can we run it here?

| Constraint | Answer |
|---|---|
| **ROS 2 Lyrical** | **Yes — our node, not theirs.** Upstream `ros_support/` is ROS 1 (2021). We do **not** run their node. `PerceptionAdapterNode` already injects `adapter=`. |
| **Arc B580 + OpenVINO 2026.4.0 `device=GPU`** | **Product path only after ONNX→IR succeeds.** MixTransformer + group-wise attention is custom mmseg (2022). Export is a **gate**, not a promise. |
| **≤10–12 GB VRAM** | **Expected to fit.** Peak memory and latency are **measured after** ONNX→IR→GPU compile. Do not freeze “&lt;2 GB” as a guarantee. If compile/run exceeds the box, **stop**. |
| **PyTorch CUDA / XPU as product** | **No.** Eval/export only. Same HARDWARE rule as YOLOE. |

**If OpenVINO export fails:** stop. Do not make PyTorch-on-Arc the live_cam path. Pick another segmenter that exports (e.g. SegFormer) or stay on YOLOE for objects.

---

## How to transport YOLOE → GA-Nav without ruining the rest

The brain never sees the model. Swap **only** the adapter + ontology + weights pin.

```
UNCHANGED
  T02 Image+CameraInfo → ImageFrame
  T07 compose_tick (remap → gates → freshness → make_mask → wire)
  T10 / T11 node, KEEP_LAST 1, watchdog, now_ns
  topics: /segmentation/*  /ugv/perception_degraded
  pixels {0,1,2}  mono8  32FC1  stamp = image time

CHANGED
  adapter: YoloeAdapter  →  GanavAdapter
  ontology YAML: yoloe.yaml → ganav.yaml  (group names → 0/1/2)
  weights pin: yoloe-26s-seg.xml → ganav-rugd-group6.xml (after export)
  T12: compile/run IR → raw tensors only (no GA-Nav argmax/softmax)
  GanavAdapter: decode those tensors → RawSemOutput
```

**Do not** dual-run YOLOE and GA-Nav on GPU in v1 (T11 one frame in flight). Person-as-hazard can come back later as a **second pass** only if T11 p95 still holds. v1 transition is **replace**, not fuse.

**Do not** use GA-Nav’s ROS1 publisher. **Do not** publish their 6-group colour map as the port.

---

## GA-Nav output → our port

Upstream **group-6** (RUGD relabel):

| Group | Meaning (their paper) | Our `{0,1,2}` |
|---|---|---|
| 0 | background (sky, sign, void) | **0** unknown |
| 1 | smooth navigable (concrete, asphalt) | **1** traversable |
| 2 | rough navigable (gravel, grass, dirt, sand, mulch) | **1** traversable |
| 3 | bumpy (rock, rock-bed) | **2** hazard |
| 4 | forbidden water | **2** hazard |
| 5 | obstacle (tree, pole, person, fence, bush, log, vehicle, building, …) | **2** hazard |

That map lives in **`config/ontologies/ganav.yaml`**, not in Python. T03 remap is unchanged: names must be keys in that file.

**Scores:** adapter computes per-pixel **softmax probabilities** from decoded logits. The selected-class score is the probability of the **argmax** class. Values are in `[0,1]`. They are **not** assumed calibrated. T04 `identity` still applies as **initial compatibility**. Recalibrating `tau_trav` / `tau_haz` from measured GA-Nav distributions is a **separate T04/profile change**, not T13. Do not invent a new τ from stills in this file.

Dense logits → `RawSemOutput` is **owned by `GanavAdapter`**, not T12. `hw` is always **camera / rgb HW**. **No instance pack.** `pack()` is YOLOE-only.

### Decode contract (`GanavAdapter` owns this)

Copy **exactly** the preprocess in the published checkpoint config (`trained_models/rugd_group6/ganav_rugd_6.py` or equivalent). Do not invent a second 640 policy. Freeze these fields at implement (write them into `config/adapters/ganav.yaml`):

| Field | Rule |
|---|---|
| Model input size | From that config (`img_scale` / crop). Not “whatever 640” |
| Color | RGB vs BGR as that config |
| Normalization | mean/std as that config |
| Resize to model | keep_ratio / pad / stretch as that config |
| IR output | expected `(1, C, h, w)` logits, `C=6` for group-6. Other layout → `AdapterError` |
| Logits → rgb HW | bilinear upsample of **logits** to `rgb.shape[:2]` |
| Corner align | `align_corners` as that config (default **false** if unspecified) |
| Softmax | **after** upsample, on the camera grid |
| Argmax | on softmax, axis = class |
| `raw_scores` | softmax probability of the argmax class, same HW |
| `id_to_name` | group index → name (`background`, `smooth`, `rough`, `bumpy`, `water`, `obstacle`) matching `ganav.yaml` keys |
| Stamp / frame | copy from `ImageFrame` (same law as T06) |

Two implementers must get the same mask from the same IR + rgb. If the IR spatial size already equals rgb HW, skip upsample.

T12: `run()` returns the raw OpenVINO tensors. It must **not** argmax, softmax, or NMS for GA-Nav. (YOLO-seg NMS stays inside today’s YOLOE path only.)

---

## Files later (only after freeze + “go”)

| File | Role |
|---|---|
| `config/ontologies/ganav.yaml` | group name → `{0,1,2}` |
| `config/adapters/ganav.yaml` | `backend: openvino_gpu`, `weights: weights/ganav-rugd-group6.xml` |
| `adapter/ganav.py` | `GanavAdapter.infer`: T12 raw tensors → upsample logits → softmax → argmax → `RawSemOutput` |
| T12 | compile/run only; no GA-Nav-specific decode |
| Node: `adapter:=ganav` **or** config pin. Default stays YOLOE until eval on `ti4` / trail stills **and** export IR exists |
| Tests: scripted dense logits (no GPU); IR smoke skip if weights missing |

**Do not edit** `compose/tick.py`, `port/`, `freshness/` publish law, T11 watchdog, or Dev 3 topics.

Weights stay **gitignored** (`turing/weights/*`).

---

## Sequence (do not skip)

1. Freeze this file.  
2. **Eval-only PyTorch** (not live_cam): load published `ganav_rugd_6.pth`, run on `ti4.avif` + trail stills. **Required evidence** (not “looks like path”):
   - predicted **group histogram** (counts for groups 0–5)
   - **nonzero pixel coverage** for rough-navigable (group 2: dirt/gravel/grass) **and** obstacle (group 5: tree/bush/…) on at least one still that contains those surfaces
   - qualitative overlay PNGs (gitignored)
   - wall time for one forward
   - output tensor nonempty (no empty-output regression)
   **Stop** if group 2 and group 5 coverage are both ~0 on those stills.  
3. If that evidence exists: ONNX export → OpenVINO 2026.4.0 GPU compile. **Stop** if compile/run fails. Measure peak VRAM and latency here.  
4. `GanavAdapter` + ontology + `adapter:=ganav` on the **existing** node.  
5. T10-style wire tests with a **FixtureAdapter** already cover the port; add dense-logit pack tests.  
6. Same ROS eval-2 stills. Product outdoor still needs Dev 5.

Upstream stack they used: Python 3.7, PyTorch 1.6–1.10, mmcv 1.3.16. That is **export-time**, not the Lyrical runtime.

---

## Aptness

| Clause | Fit |
|---|---|
| §6 adapter not brain | **High** — new adapter, same remap |
| §8.1 `{0,1,2}` | **High** — YAML map from groups |
| §8.2 no remap → no publish | **High** — `ganav.yaml` required |
| T11 one infer / KEEP_LAST 1 | **High** — same node |
| HARDWARE OpenVINO GPU | **Gated** on export |
| Their ROS1 node | **Out** |

---

## Non-goals

Fine-tune YOLOE on RUGD in this transition. Dual GPU YOLOE+GA-Nav. Their ROS1 node. PyTorch XPU as live_cam. Changing `tau_trav`. T08. `/cmd_vel`. DummySource.

## Done when (this subarch)

Written and frozen. Implementation starts only when you say go. **Product “GA-Nav is live_cam”** additionally requires: OpenVINO IR on disk, T10 still green, step-2 histogram evidence (nonzero group 2 and 5 on a still that has dirt/trees), T11 one-infer still holds.
