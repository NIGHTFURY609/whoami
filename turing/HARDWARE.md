# Hardware and runtime (Dev 1)

**This file owns only where tensors run** (device, engine, VRAM).  
**It does not own the product architecture.** [`architecture.md`](../architecture.md) still owns the Perception Port, adapters-vs-brain, safety, and `/cmd_vel`. T12 is the only code switch. T07 and the port never import a vendor runtime.

If this file and `architecture.md` disagree on the port or topics, `architecture.md` wins. If they disagree on GPU/engine, this file wins.

## Two different Intel stacks (do not mix)

`OpenVINO` and `XPU` are **not** the same layer. Do not write “OpenVINO + XPU” as one required stack.

```
Production on this Arc (v1):

  YOLOE weights → OpenVINO 2026.4.0 → device="GPU" → Arc B580

Separate optional path (not the product runtime):

  YOLOE / PyTorch → PyTorch XPU → Arc B580
```

| Path | What it is | v1 product? |
|---|---|---|
| **OpenVINO GPU** | OpenVINO runtime, Intel GPU plugin, `device="GPU"` | **Yes — this desktop** |
| **PyTorch XPU** | Intel PyTorch `xpu` device | No. Export/debug only if we use it. Never required *with* OpenVINO. |

Someone implementing T12 must pick **one** backend class, not both in one process as the product path.

## Pin (this project)

- **OpenVINO 2026.4.0** (current stable as of 2026-09-16: `pip install openvino==2026.4.0`).
- Compile / load IR for **GPU**. Do not default to CPU to “make it run.”
- Do not require `intel-extension-for-pytorch` for the product OpenVINO path.

## Machines

| When | GPU | Runtime | Goal |
|---|---|---|---|
| **Now** (this desktop) | Intel Arc **B580** | **OpenVINO 2026.4.0, `device=GPU`** | Efficient Arc inference |
| **Later** (after optimize) | NVIDIA, **less VRAM** than B580 | **CUDA + PyTorch** | Must still fit; do not design to B580 headroom |

```
if Intel Arc (this box):  OpenVINO GPU   (backend id: openvino_gpu)
else NVIDIA:              CUDA PyTorch   (backend id: cuda_pytorch)
```

Ultralytics is **not** the product runtime. It may export ONNX/IR **into** OpenVINO. Do not run Ultralytics/CUDA on this box as `live_cam`.

## VRAM budget (design for the later NVIDIA)

B580 is the development card, not the floor. Assume the NVIDIA has **less** VRAM:

- One frame in flight (latest-only queue; T11).
- YOLOE and Depth Anything **sequential on the same frame** if both run.
- Default adapter weights: **RUGD SegFormer-B5** OpenVINO IR (`rugd-segformer.xml`). YOLOE-26s stays on disk, not live.
- No extra GPU copies; no keeping RGB + mask + depth + two models resident if it blows the small card.
- INT8 / extra compression is later, not a dummy-mask shortcut.

## ROS on this desktop (2026-09-18)

- Distro: **ROS 2 Lyrical** (RHEL 10 Tier-2 pairing). `architecture.md` still lists Jazzy/Humble as the product stack target — this box runs **Lyrical**.
- Installed: `rclpy`, `sensor_msgs`, `std_msgs`, `rmw-dds-common-runtime`.
- Node tests: `source /opt/ros/lyrical/setup.bash`, keep `LD_LIBRARY_PATH` / `AMENT_PREFIX_PATH`; do **not** export ROS `PYTHONPATH` into pytest (breaks collection via `launch_testing`). `conftest.py` adds ROS site-packages only if `/opt/ros/lyrical` exists.
- `pytest.importorskip` on ROS test modules so pure kernels run without ROS.

## Outdoor / camera (now)

- Work is on a **desktop**. Outdoor live training/testing is **later**.
- **No camera device.** Dev 1 **consumes** `Image` + `CameraInfo` (T02 `decode_frame` + `ros_bridge`; T07 node subscribes). Dev 5 owns the driver.
- T02/T07 ROS path is **tested** with fixture messages (shared executor). That is not a dummy camera driver.
- **No training.** Default weights pin: **RUGD SegFormer-B5** IR — **on disk** at `weights/rugd-segformer.xml`.
- T12 live `run` uses that IR on `device=GPU`. T06 outdoor product `infer` and T11 still wait on a Dev 5 stream.
