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
