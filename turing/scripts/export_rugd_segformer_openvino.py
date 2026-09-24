#!/usr/bin/env python3
"""Export the local RUGD SegFormer checkpoint to OpenVINO IR and run it on GPU.

Transformers loads the safetensors for this conversion only. Inference is
OpenVINO 2026.4.0, device="GPU" (Arc B580). No CPU compile.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "weights" / "rugd-segformer"
IR_XML = ROOT / "weights" / "rugd-segformer.xml"
INPUT_HW = (640, 640)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def export_ir() -> None:
    import torch
    from transformers import SegformerForSemanticSegmentation
    import openvino as ov

    if not (SRC / "model.safetensors").is_file():
        raise SystemExit(f"missing checkpoint: {SRC / 'model.safetensors'}")

    class _Logits(torch.nn.Module):
        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return self.model(pixel_values=pixel_values).logits

    model = SegformerForSemanticSegmentation.from_pretrained(SRC)
    model.eval()
    wrapper = _Logits(model)
    example = torch.zeros(1, 3, INPUT_HW[0], INPUT_HW[1], dtype=torch.float32)
    with torch.inference_mode():
        ov_model = ov.convert_model(wrapper, example_input=example)
    IR_XML.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, IR_XML, compress_to_fp16=False)
    print(f"wrote {IR_XML}")


def _compiled():
    import openvino as ov

    if not IR_XML.is_file():
        raise SystemExit(f"missing IR: {IR_XML}")
    core = ov.Core()
    devices = list(core.available_devices)
    if not any(str(d).startswith("GPU") for d in devices):
        raise SystemExit(f"OpenVINO GPU not available; devices={devices}")
    model = core.read_model(IR_XML)
    try:
        compiled = core.compile_model(model, "GPU")
    except Exception as exc:
        raise SystemExit(f"OpenVINO compile on GPU failed: {exc}") from exc
    return core, compiled


def _preprocess(rgb: np.ndarray) -> np.ndarray:
    from PIL import Image

    image = Image.fromarray(rgb, mode="RGB").resize((INPUT_HW[1], INPUT_HW[0]), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return np.transpose(arr, (2, 0, 1))[None].astype(np.float32)


def _upsample_argmax(logits: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 25:
        raise SystemExit(f"expected logits (1, 25, h, w), got {logits.shape}")
    if not np.isfinite(logits).all():
        raise SystemExit("logits are not finite")
    h, w = hw
    planes = []
    for c in range(25):
        plane = logits[0, c]
        img = Image.fromarray(plane.astype(np.float32), mode="F")
        planes.append(np.asarray(img.resize((w, h), Image.BILINEAR), dtype=np.float32))
    stacked = np.stack(planes, axis=0)
    return np.argmax(stacked, axis=0).astype(np.int32)


def run_gpu(images: list[Path]) -> None:
    import json

    core, compiled = _compiled()
    stats = core.get_property("GPU", "GPU_MEMORY_STATISTICS")
    print("compiled device=GPU")
    print("GPU_MEMORY_STATISTICS", stats)
    labels = json.loads((SRC / "config.json").read_text())["id2label"]
    dummy = np.zeros((1, 3, INPUT_HW[0], INPUT_HW[1]), dtype=np.float32)
    result = compiled([dummy])
    logits = np.asarray(result[compiled.output(0)])
    print("dummy logits", logits.shape, logits.dtype)
    if images:
        from PIL import Image

    for path in images:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        blob = _preprocess(rgb)
        out = compiled([blob])
        pred = _upsample_argmax(np.asarray(out[compiled.output(0)]), rgb.shape[:2])
        total = pred.size
        counts = np.bincount(pred.ravel(), minlength=25)
        order = np.argsort(counts)[::-1]
        print(f"\n{path.name} {rgb.shape[1]}x{rgb.shape[0]}")
        for idx in order[:8]:
            if counts[idx] == 0:
                break
            name = labels[str(int(idx))]
            print(f"  {name:24} {100.0 * counts[idx] / total:6.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("images", nargs="*")
    args = parser.parse_args()
    if args.export or not IR_XML.is_file():
        export_ir()
    if args.run or args.images:
        run_gpu([Path(p) for p in args.images])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
