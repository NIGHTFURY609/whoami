"""Wire the live RUGD adapter to the GPU tensor backend."""

from __future__ import annotations

from pathlib import Path

from ugv_perception.adapter.rugd import RugdSegformerAdapter, load_rugd_config
from ugv_perception.backend.openvino_gpu import OpenVinoGpuTensorBackend


def build_live_adapter(root: Path) -> RugdSegformerAdapter:
    cfg = load_rugd_config(root / "config" / "adapters" / "rugd.yaml")
    backend = OpenVinoGpuTensorBackend()
    backend.load(str(root / str(cfg["weights"])))
    hw = cfg["input_hw"]
    mean = cfg["mean"]
    std = cfg["std"]
    if type(hw) is not tuple or type(mean) is not tuple or type(std) is not tuple:
        raise TypeError("rugd config fields have the wrong type")
    return RugdSegformerAdapter(backend, input_hw=hw, mean=mean, std=std)
