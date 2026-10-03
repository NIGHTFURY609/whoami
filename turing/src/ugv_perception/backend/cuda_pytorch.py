"""CUDA PyTorch backends. Tensor path is live RUGD/DA3 on NVIDIA. YOLOE stays a stub."""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ugv_perception.adapter.output import AdapterError, Instance
from ugv_perception.depth.geometry import METRIC_SCALE


class CudaPytorchBackend:
    id = "cuda_pytorch"

    def load(self, weights_path: str, **engine_args: object) -> None:
        raise AdapterError("cuda_pytorch YOLOE is not wired; live path is RUGD")

    def run(self, rgb: NDArray[np.uint8]) -> tuple[Instance, ...]:
        raise AdapterError("cuda_pytorch YOLOE is not wired; live path is RUGD")


class CudaPytorchTensorBackend:
    """SegFormer logits or DA3 depth_raw+sky on CUDA. Same blob contract as OpenVINO."""

    id = "cuda_pytorch"

    def __init__(self) -> None:
        self._model = None
        self._kind: str | None = None
        self._hw: tuple[int, int] | None = None
        self._fallback_logits: np.ndarray | None = None
        self.seg_post_disabled = False
        self.rgb_pre_disabled = False
        # Every tensor this backend creates, and the model it loads, goes to this device.
        self.device = "cuda"
        self._stream = None
        self._pending: list | None = None
        self._pending_depth = None
        self._primed_tensor = None

    def load(self, weights_path: str, kind: str = "rugd") -> None:
        path = Path(weights_path)
        if not (path / "model.safetensors").is_file():
            raise FileNotFoundError(f"safetensors missing: {path / 'model.safetensors'}")
        if kind not in ("rugd", "da3"):
            raise ValueError("kind must be rugd or da3")
        try:
            import torch
        except ImportError as exc:
            raise AdapterError("torch is not installed") from exc
        if torch.device(self.device).type == "cuda" and not torch.cuda.is_available():
            raise AdapterError("CUDA is not available")
        try:
            if kind == "rugd":
                self._model = _load_rugd(path, self.device)
            else:
                self._model = _load_da3(path, self.device)
        except AdapterError:
            raise
        except ImportError as exc:
            raise AdapterError("CUDA backend dependency missing") from exc
        except Exception as exc:
            raise AdapterError("CUDA load failed") from exc
        self._kind = kind
        if torch.cuda.is_available():
            self._stream = torch.cuda.Stream()

    def _cuda_stream(self):
        """Bind this net's stream. A second backend (RUGD vs DA3) has its own stream, so a
        host copy on the pool thread waits only for that net and the other can keep running.
        """
        import torch

        if self._stream is None:
            return contextlib.nullcontext()
        torch.cuda.set_device(self._stream.device_index)
        return torch.cuda.stream(self._stream)

    def ensure_hw(self, height: int, width: int) -> None:
        if self._model is None:
            raise AdapterError("CudaPytorchTensorBackend.load() was not called")
        self._hw = (int(height), int(width))

    def run(self, blob: NDArray[np.float32]) -> np.ndarray:
        if self._fallback_logits is not None:
            out = self._fallback_logits
            self._fallback_logits = None
            return out
        return self.run_all(blob)[0]

    def run_seg(
        self, blob: NDArray[np.float32], out_hw: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate + softmax on CUDA. Labels/scores copied out once."""
        tensor = self._blob_to_cuda(blob)
        return self._run_seg_tensor(tensor, out_hw)

    def run_seg_from_rgb(
        self,
        rgb: NDArray[np.uint8],
        out_hw: tuple[int, int],
        input_hw: tuple[int, int],
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._primed_tensor is not None:
            tensor = self._primed_tensor
            self._primed_tensor = None
        else:
            tensor = self._rgb_pre_cuda(rgb, (input_hw,), mean, std, clip_255=False)
        return self._run_seg_tensor(tensor, out_hw)

    def prime_seg_rgb(
        self,
        rgb: NDArray[np.uint8],
        input_hw: tuple[int, int],
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        """Fill the RUGD input on this stream before either net is started."""
        self._primed_tensor = self._rgb_pre_cuda(rgb, (input_hw,), mean, std, clip_255=False)

    def begin_run_all_from_rgb(
        self,
        rgb: NDArray[np.uint8],
        sizes: tuple[tuple[int, int], ...],
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        tensor = self._rgb_pre_cuda(rgb, sizes, mean, std, clip_255=True)
        self._begin_forward(tensor)

    def begin_run_all(self, blob: NDArray[np.float32]) -> None:
        self._begin_forward(self._blob_to_cuda(blob))

    def wait_run_all(self) -> list[np.ndarray]:
        if self._pending is None:
            raise AdapterError("CUDA run was not started")
        try:
            with self._cuda_stream():
                return [tensor.detach().cpu().numpy() for tensor in self._pending]
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError("CUDA run failed") from exc
        finally:
            self._pending = None

    def run_all(self, blob: NDArray[np.float32]) -> list[np.ndarray]:
        tensor = self._blob_to_cuda(blob)
        return self._run_all_tensor(tensor)

    def run_all_from_rgb(
        self,
        rgb: NDArray[np.uint8],
        sizes: tuple[tuple[int, int], ...],
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> list[np.ndarray]:
        tensor = self._rgb_pre_cuda(rgb, sizes, mean, std, clip_255=True)
        return self._run_all_tensor(tensor)

    def _blob_to_cuda(self, blob: NDArray[np.float32]) -> object:
        if self._model is None or self._kind is None:
            raise AdapterError("CudaPytorchTensorBackend.load() was not called")
        if not isinstance(blob, np.ndarray) or blob.dtype != np.float32 or blob.ndim != 4:
            raise TypeError("blob must be float32 NCHW")
        try:
            import torch
        except ImportError as exc:
            raise AdapterError("torch is not installed") from exc
        with self._cuda_stream():
            return torch.from_numpy(blob).to(self.device)

    def _rgb_pre_cuda(
        self,
        rgb: np.ndarray,
        sizes: tuple[tuple[int, int], ...],
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        clip_255: bool,
    ) -> object:
        if self.rgb_pre_disabled:
            raise AdapterError("CUDA rgb pre is disabled")
        if self._model is None or self._kind is None:
            raise AdapterError("CudaPytorchTensorBackend.load() was not called")
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise TypeError("rgb must be uint8 HWC")
        if not sizes:
            raise ValueError("rgb pre needs at least one target size")
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:
            raise AdapterError("torch is not installed") from exc
        try:
            with self._cuda_stream(), torch.inference_mode():
                tensor = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
                tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
                for oh, ow in sizes:
                    th, tw = int(oh), int(ow)
                    if (int(tensor.shape[-2]), int(tensor.shape[-1])) != (th, tw):
                        tensor = F.interpolate(
                            tensor, size=(th, tw), mode="bilinear", align_corners=False
                        )
                        if clip_255:
                            tensor = tensor.clamp(0.0, 255.0)
                mean_t = torch.tensor(mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
                std_t = torch.tensor(std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
                return (tensor / 255.0 - mean_t) / std_t
        except AdapterError:
            self.rgb_pre_disabled = True
            raise
        except Exception as exc:
            self.rgb_pre_disabled = True
            raise AdapterError("CUDA rgb pre failed") from exc

    def _run_seg_tensor(
        self, tensor: object, out_hw: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._model is None or self._kind != "rugd":
            raise AdapterError("CUDA seg decode requires a loaded RUGD net")
        oh, ow = int(out_hw[0]), int(out_hw[1])
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:
            raise AdapterError("torch is not installed") from exc
        logits = None
        try:
            with self._cuda_stream(), torch.inference_mode():
                logits = self._model(pixel_values=tensor).logits
                if not torch.isfinite(logits).all():
                    raise AdapterError("RUGD logits are not finite")
                up = F.interpolate(
                    logits, size=(oh, ow), mode="bilinear", align_corners=False
                )
                prob = torch.softmax(up, dim=1)
                labels = torch.argmax(prob, dim=1)
                scores = torch.gather(prob, 1, labels.unsqueeze(1)).squeeze(1)
                return (
                    labels[0].detach().cpu().numpy().astype(np.int32, copy=False),
                    scores[0].detach().cpu().numpy().astype(np.float32, copy=False),
                )
        except AdapterError as exc:
            if "not finite" in str(exc):
                raise
            self.seg_post_disabled = True
            raise
        except Exception as exc:
            self.seg_post_disabled = True
            if logits is not None:
                self._fallback_logits = logits.detach().cpu().numpy()
            raise AdapterError("CUDA seg decode failed") from exc

    def _run_all_tensor(self, tensor: object) -> list[np.ndarray]:
        if self._model is None or self._kind is None:
            raise AdapterError("CudaPytorchTensorBackend.load() was not called")
        try:
            import torch
        except ImportError as exc:
            raise AdapterError("torch is not installed") from exc
        self._begin_forward(tensor)
        return self.wait_run_all()

    def _begin_forward(self, tensor: object) -> None:
        if self._model is None or self._kind is None:
            raise AdapterError("CudaPytorchTensorBackend.load() was not called")
        if self._pending is not None:
            raise AdapterError("CUDA run already in flight")
        try:
            import torch
        except ImportError as exc:
            raise AdapterError("torch is not installed") from exc
        try:
            with self._cuda_stream(), torch.inference_mode():
                if self._kind == "rugd":
                    self._pending = [self._model(pixel_values=tensor).logits]
                else:
                    depth, sky = self._model(tensor)
                    self._pending = [depth, sky]
        except AdapterError:
            self._pending = None
            raise
        except Exception as exc:
            self._pending = None
            raise AdapterError("CUDA run failed") from exc

    def run_depth_metres(
        self,
        rgb: NDArray[np.uint8],
        focal: float,
        model_size: tuple[int, int],
        out_hw: tuple[int, int],
    ) -> NDArray[np.float32]:
        """DA3 on CUDA with the pre/post-processing on the GPU, in FP32: camera-sized depth in metres, NaN holes.

        Same maths as geometry.preprocess_nchw, meters_from_raw and hole_safe_resize (geometry_gpu), so only
        the camera frame goes up and only the camera-sized depth comes back. No forced synchronize: the copy
        back to the host is the only wait.
        """
        if self._model is None or self._kind != "da3":
            raise AdapterError("run_depth_metres needs a loaded DA3 model")
        try:
            import torch

            from ugv_perception.depth import geometry_gpu
        except ImportError as exc:
            raise AdapterError("torch is not installed") from exc
        try:
            self.begin_depth_metres(rgb, focal, model_size, out_hw)
            return self.wait_depth_metres()
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError("CUDA run failed") from exc

    def begin_depth_metres(
        self,
        rgb: NDArray[np.uint8],
        focal: float,
        model_size: tuple[int, int],
        out_hw: tuple[int, int],
    ) -> None:
        """Queue DA3 pre, the net, and hole-safe on this stream. wait_depth_metres() copies."""
        if self._model is None or self._kind != "da3":
            raise AdapterError("run_depth_metres needs a loaded DA3 model")
        if self._pending_depth is not None:
            raise AdapterError("DA3 CUDA run already in flight")
        try:
            import torch

            from ugv_perception.depth import geometry_gpu
        except ImportError as exc:
            raise AdapterError("torch is not installed") from exc
        try:
            self.ensure_hw(*model_size)
            with self._cuda_stream(), torch.inference_mode():
                blob, sized = geometry_gpu.preprocess_nchw_gpu(rgb, self.device)
                if sized != tuple(model_size):
                    raise AdapterError("preprocess size disagrees with K_model")
                depth, sky = self._model(blob)
                metres = depth[0].float() * (float(focal) / METRIC_SCALE)
                self._pending_depth = geometry_gpu.hole_safe_resize_gpu(
                    metres, sky[0], tuple(out_hw)
                )
        except AdapterError:
            self._pending_depth = None
            raise
        except Exception as exc:
            self._pending_depth = None
            raise AdapterError("CUDA run failed") from exc

    def wait_depth_metres(self) -> NDArray[np.float32]:
        if self._pending_depth is None:
            raise AdapterError("CUDA depth was not started")
        try:
            with self._cuda_stream():
                return self._pending_depth.cpu().numpy()
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError("CUDA run failed") from exc
        finally:
            self._pending_depth = None


def _load_rugd(path: Path, device: str) -> object:
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(
        str(path), local_files_only=True
    )
    model.eval()
    return model.to(device)


def _load_da3(path: Path, device: str) -> object:
    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.registry import MODEL_REGISTRY
    from safetensors.torch import load_file

    net = create_object(load_config(MODEL_REGISTRY["da3metric-large"]))
    state = load_file(str(path / "model.safetensors"))
    if any(key.startswith("model.") for key in state):
        state = {key.removeprefix("model."): value for key, value in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise AdapterError(
            "DA3 checkpoint does not match the net "
            f"(missing={len(missing)} unexpected={len(unexpected)})"
        )
    net.eval()
    return _Da3Head(net).to(device)


class _Da3Head:
    """depth_raw and sky from the depth head, before sky-fill. Matches the IR export."""

    def __init__(self, net: object) -> None:
        self.net = net

    def __call__(self, pixel_values: object) -> tuple[object, object]:
        image = pixel_values.unsqueeze(1)
        feats, _aux = self.net.backbone(
            image,
            cam_token=None,
            export_feat_layers=[],
            ref_view_strategy="saddle_balanced",
        )
        height, width = int(image.shape[-2]), int(image.shape[-1])
        output = self.net._process_depth_head(feats, height, width)
        return output.depth[:, 0], output.sky[:, 0]

    def to(self, device: str) -> "_Da3Head":
        self.net = self.net.to(device)
        return self

    def eval(self) -> "_Da3Head":
        self.net.eval()
        return self
