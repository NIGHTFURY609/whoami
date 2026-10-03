"""Hardware picker for live RUGD/DA3. No CUDA required. No downloads."""

from __future__ import annotations

from pathlib import Path

import pytest

from ugv_perception.adapter.output import AdapterError
from ugv_perception.adapter.rugd import load_rugd_config
from ugv_perception.backend import device
from ugv_perception.backend.cuda_pytorch import CudaPytorchTensorBackend
from ugv_perception.backend.device import (
    can_overlap_gpu,
    cuda_available,
    intel_openvino_gpu_available,
    is_gpu_device,
    pick_tensor_backend,
)


def _weights(tmp_path: Path, *, xml: bool, safetensors: bool) -> tuple[Path, Path]:
    ir = tmp_path / "rugd-segformer.xml"
    folder = tmp_path / "rugd-segformer"
    folder.mkdir()
    if xml:
        ir.write_bytes(b"ir")
    if safetensors:
        (folder / "model.safetensors").write_bytes(b"st")
    return ir, folder


def test_load_rugd_config_accepts_cuda_pytorch(tmp_path: Path) -> None:
    path = tmp_path / "rugd.yaml"
    path.write_text(
        "adapter_id: rugd\n"
        "backend: cuda_pytorch\n"
        "weights: weights/rugd-segformer\n"
        "input_hw: [640, 640]\n"
        "mean: [0.485, 0.456, 0.406]\n"
        "std: [0.229, 0.224, 0.225]\n",
        encoding="utf-8",
    )
    cfg = load_rugd_config(path)
    assert cfg["backend"] == "cuda_pytorch"


def test_device_py_has_no_module_level_vendor_imports() -> None:
    text = Path(device.__file__).read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("import ") or line.startswith("from "):
            assert "openvino" not in line
            assert "torch" not in line
            assert "transformers" not in line


def test_can_overlap_gpu_requires_both_on_gpu() -> None:
    class _Gpu:
        device = "GPU"

    class _Cpu:
        device = "CPU"

    class _Cuda:
        device = "cuda"

    class _Adapter:
        def __init__(self, backend: object) -> None:
            self._backend = backend

    assert is_gpu_device("GPU") and is_gpu_device("GPU.0") and is_gpu_device("cuda")
    assert not is_gpu_device("CPU") and not is_gpu_device("")
    assert can_overlap_gpu(_Adapter(_Gpu()), type("D", (), {"_backend": _Gpu()})())
    assert can_overlap_gpu(_Adapter(_Cuda()), type("D", (), {"_backend": _Cuda()})())
    assert not can_overlap_gpu(_Adapter(_Cpu()), type("D", (), {"_backend": _Cpu()})())
    assert not can_overlap_gpu(_Adapter(_Gpu()), type("D", (), {"_backend": _Cpu()})())
    assert not can_overlap_gpu(_Adapter(_Gpu()), None)


def test_adapters_stay_vendor_neutral() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in (
        "adapter/rugd.py",
        "depth/geometry.py",
        "backend/depth_live.py",
    ):
        text = (root / rel).read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                assert "openvino" not in stripped
                assert "torch" not in stripped


def test_pick_prefers_intel_gpu_ir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    xml, folder = _weights(tmp_path, xml=True, safetensors=True)
    monkeypatch.setattr(device, "intel_openvino_gpu_available", lambda: True)
    monkeypatch.setattr(device, "cuda_available", lambda: True)
    monkeypatch.setattr(device, "_load_openvino_ir", lambda path: ("ov", path))
    monkeypatch.setattr(device, "_load_cuda_safetensors", lambda path, kind: ("cuda", path, kind))
    assert pick_tensor_backend(ir_xml=xml, safetensors_dir=folder, kind="rugd") == ("ov", xml)


def test_pick_cuda_when_intel_gpu_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    xml, folder = _weights(tmp_path, xml=True, safetensors=True)
    monkeypatch.setattr(device, "intel_openvino_gpu_available", lambda: False)
    monkeypatch.setattr(device, "cuda_available", lambda: True)
    monkeypatch.setattr(device, "_load_openvino_ir", lambda path: ("ov", path))
    monkeypatch.setattr(device, "_load_cuda_safetensors", lambda path, kind: ("cuda", path, kind))
    assert pick_tensor_backend(ir_xml=xml, safetensors_dir=folder, kind="da3") == (
        "cuda",
        folder,
        "da3",
    )


def test_pick_cuda_without_ir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    xml, folder = _weights(tmp_path, xml=False, safetensors=True)
    monkeypatch.setattr(device, "intel_openvino_gpu_available", lambda: False)
    monkeypatch.setattr(device, "cuda_available", lambda: True)
    monkeypatch.setattr(device, "_load_cuda_safetensors", lambda path, kind: ("cuda", path, kind))
    assert pick_tensor_backend(ir_xml=xml, safetensors_dir=folder, kind="rugd") == (
        "cuda",
        folder,
        "rugd",
    )


def test_pick_openvino_cpu_when_no_intel_gpu_no_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xml, folder = _weights(tmp_path, xml=True, safetensors=False)
    monkeypatch.setattr(device, "intel_openvino_gpu_available", lambda: False)
    monkeypatch.setattr(device, "cuda_available", lambda: False)
    monkeypatch.setattr(device, "_load_openvino_ir", lambda path: ("ov-cpu", path))
    assert pick_tensor_backend(ir_xml=xml, safetensors_dir=folder, kind="rugd") == (
        "ov-cpu",
        xml,
    )


def test_pick_none_when_no_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    xml, folder = _weights(tmp_path, xml=False, safetensors=False)
    monkeypatch.setattr(device, "intel_openvino_gpu_available", lambda: True)
    monkeypatch.setattr(device, "cuda_available", lambda: True)
    assert pick_tensor_backend(ir_xml=xml, safetensors_dir=folder, kind="rugd") is None


def test_openvino_main_net_compile_has_no_runtime_precision_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPU pre/post stay f32. The live net compile is the plugin-default used for the 16/24 fps path."""
    from ugv_perception.backend import openvino_gpu

    seen: list[object] = []
    monkeypatch.setattr(
        openvino_gpu,
        "_compile_gpu_then_cpu",
        lambda core, model, config=None: seen.append(config) or (None, "CPU"),
    )

    class _Model:
        def input(self, i):
            raise AssertionError("not reached")

    backend = openvino_gpu.OpenVinoGpuTensorBackend()
    backend._core, backend._model = object(), _Model()
    with pytest.raises(AssertionError, match="not reached"):
        backend._compile()
    assert seen == [None]


def test_cuda_tensor_load_raises_without_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "da3metric-large"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"st")
    try:
        import torch
    except ImportError:
        backend = CudaPytorchTensorBackend()
        with pytest.raises(AdapterError, match="torch is not installed"):
            backend.load(str(folder), kind="da3")
        return
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    backend = CudaPytorchTensorBackend()
    with pytest.raises(AdapterError, match="CUDA is not available"):
        backend.load(str(folder), kind="da3")


@pytest.mark.skipif(cuda_available(), reason="CUDA present")
def test_cuda_unavailable_on_this_box() -> None:
    assert cuda_available() is False


def test_intel_gpu_available_on_this_box() -> None:
    ov = pytest.importorskip("openvino")
    devices = [str(d) for d in ov.Core().available_devices]
    if not any(item == "GPU" or item.startswith("GPU.") for item in devices):
        pytest.skip("OpenVINO GPU missing")
    assert intel_openvino_gpu_available() is True


def test_live_adapter_uses_openvino_gpu_on_this_box() -> None:
    from ugv_perception.backend.rugd_live import build_live_adapter

    root = Path(__file__).resolve().parents[3]
    if not (root / "weights" / "rugd-segformer.xml").is_file():
        pytest.skip("RUGD IR missing")
    if not intel_openvino_gpu_available():
        pytest.skip("OpenVINO GPU missing")
    adapter = build_live_adapter(root)
    assert str(adapter._backend.device).startswith("GPU")


def test_depth_channel_uses_openvino_gpu_on_this_box() -> None:
    from ugv_perception.backend.depth_live import build_depth_channel

    root = Path(__file__).resolve().parents[3]
    if not (root / "weights" / "da3metric-large.xml").is_file():
        pytest.skip("DA3 IR missing")
    if not intel_openvino_gpu_available():
        pytest.skip("OpenVINO GPU missing")
    channel = build_depth_channel(root)
    assert channel is not None
    assert str(channel._backend.device).startswith("GPU")
