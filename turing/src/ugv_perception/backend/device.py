"""Pick Intel OpenVINO GPU, CUDA PyTorch, or OpenVINO CPU. Lazy vendor imports."""

from __future__ import annotations

from pathlib import Path


def intel_openvino_gpu_available() -> bool:
    try:
        import openvino as ov
    except ImportError:
        return False
    try:
        devices = [str(d) for d in ov.Core().available_devices]
    except Exception:
        return False
    return any(item == "GPU" or item.startswith("GPU.") for item in devices)


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def is_gpu_device(device: object) -> bool:
    """True for Intel OpenVINO GPU* and NVIDIA CUDA. CPU stays off this path."""
    name = str(device or "").strip().upper()
    if name.startswith("CPU"):
        return False
    return name.startswith("GPU") or name.startswith("CUDA") or name.startswith("NVIDIA")


def can_overlap_gpu(adapter: object, depth: object) -> bool:
    """Overlap RUGD and DA3 only when both compiled devices are GPUs."""
    if depth is None:
        return False
    mask_backend = getattr(getattr(adapter, "_inner", adapter), "_backend", None)
    depth_backend = getattr(depth, "_backend", None)
    if mask_backend is None or depth_backend is None:
        return False
    return is_gpu_device(getattr(mask_backend, "device", "")) and is_gpu_device(
        getattr(depth_backend, "device", "")
    )


def _load_openvino_ir(xml: Path) -> object:
    from ugv_perception.backend.openvino_gpu import OpenVinoGpuTensorBackend

    backend = OpenVinoGpuTensorBackend()
    backend.load(str(xml), input_hw=None)
    return backend


def _load_cuda_safetensors(folder: Path, kind: str) -> object:
    from ugv_perception.backend.cuda_pytorch import CudaPytorchTensorBackend

    backend = CudaPytorchTensorBackend()
    backend.load(str(folder), kind=kind)
    return backend


def pick_tensor_backend(
    *,
    ir_xml: Path,
    safetensors_dir: Path,
    kind: str,
) -> object | None:
    """Intel GPU IR, then CUDA safetensors, then OpenVINO CPU IR. None if nothing fits."""
    xml = Path(ir_xml)
    folder = Path(safetensors_dir)
    if intel_openvino_gpu_available() and xml.is_file():
        return _load_openvino_ir(xml)
    if cuda_available() and (folder / "model.safetensors").is_file():
        return _load_cuda_safetensors(folder, kind)
    if xml.is_file():
        return _load_openvino_ir(xml)
    return None
