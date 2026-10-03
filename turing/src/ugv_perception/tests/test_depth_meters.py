"""T08 meter chain. No weights, no GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ugv_perception.depth.geometry import (
    SKY_THRESHOLD,
    backproject,
    focal_model,
    geometry_is_fresh,
    hole_safe_resize,
    k_model,
    meters_from_raw,
    model_hw,
    preprocess_nchw,
)


def _k(h: int, w: int, fx: float = 200.0, fy: float = 200.0) -> np.ndarray:
    return np.array(
        [[fx, 0.0, (w - 1) / 2.0], [0.0, fy, (h - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def test_plane_center_is_two_meters() -> None:
    h, w = 28, 28
    k_cam = _k(h, w)
    km, (mh, mw) = k_model(k_cam, (h, w))
    focal = focal_model(km)
    raw_z = 2.0 * 300.0 / focal
    raw = np.full((mh, mw), raw_z, dtype=np.float64)
    sky = np.zeros((mh, mw), dtype=np.float64)
    depth = hole_safe_resize(meters_from_raw(raw, focal), sky, (h, w))
    xyz = backproject(depth, k_cam)
    center = xyz[np.argmin(np.abs(xyz[:, 0]) + np.abs(xyz[:, 1]))]
    half_pixel = 0.5 * 2.0 / float(k_cam[0, 0])
    assert abs(float(center[0])) <= half_pixel + 1e-4
    assert abs(float(center[1])) <= half_pixel + 1e-4
    assert abs(float(center[2]) - 2.0) < 1e-3


def test_bilinear_through_hole_stays_a_hole() -> None:
    depth = np.array([[2.0, np.nan, 3.0]], dtype=np.float64)
    sky = np.zeros((1, 3), dtype=np.float64)
    out = hole_safe_resize(depth, sky, (1, 3))
    assert np.isnan(out[0, 1])
    assert abs(float(out[0, 0]) - 2.0) < 1e-6
    assert abs(float(out[0, 2]) - 3.0) < 1e-6


def test_sky_threshold_boundary() -> None:
    depth = np.ones((1, 2), dtype=np.float64)
    sky = np.array([[SKY_THRESHOLD, SKY_THRESHOLD - 0.01]], dtype=np.float64)
    out = hole_safe_resize(depth, sky, (1, 2))
    assert np.isnan(out[0, 0])
    assert np.isfinite(out[0, 1])


def test_nan_raw_is_absent() -> None:
    depth = np.array([[np.nan]], dtype=np.float64)
    sky = np.zeros((1, 1), dtype=np.float64)
    xyz = backproject(hole_safe_resize(depth, sky, (1, 1)), _k(1, 1))
    assert xyz.shape == (0, 3)


def test_stale_cloud_is_not_fresh() -> None:
    stamp = 1_000_000_000
    assert geometry_is_fresh(stamp + 400_000_000, stamp) is True
    assert geometry_is_fresh(stamp + 600_000_000, stamp) is False
    assert geometry_is_fresh(stamp - 1, stamp) is False


def test_preprocess_longest_side_and_patch() -> None:
    mh, mw = model_hw(100, 200)
    assert max(mh, mw) <= 504 + 14
    assert mh % 14 == 0 and mw % 14 == 0


def test_trail_still_maps_to_landscape_336x504() -> None:
    assert model_hw(408, 612) == (336, 504)
    rgb = np.zeros((408, 612, 3), dtype=np.uint8)
    blob, sized = preprocess_nchw(rgb)
    assert sized == (336, 504)
    assert blob.shape == (1, 3, 336, 504)


def test_focal_model_scales_meters() -> None:
    raw = np.array([[1.0]])
    k_lo = _k(408, 612, fx=300.0, fy=300.0)
    k_hi = _k(408, 612, fx=600.0, fy=600.0)
    f_lo = focal_model(k_model(k_lo, (408, 612))[0])
    f_hi = focal_model(k_model(k_hi, (408, 612))[0])
    z_lo = float(meters_from_raw(raw, f_lo)[0, 0])
    z_hi = float(meters_from_raw(raw, f_hi)[0, 0])
    assert abs(z_hi / z_lo - 2.0) < 1e-6


def test_depth_image_msg_is_32fc1_meters() -> None:
    sensor_msgs = pytest.importorskip("sensor_msgs")
    from ugv_perception.node.cloud import depth_to_image

    depth = np.array([[2.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    msg = depth_to_image(depth, 1_000_000_000, "camera_optical")
    assert msg.encoding == "32FC1"
    assert msg.height == 2 and msg.width == 2
    assert msg.header.frame_id == "camera_optical"
    packed = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(2, 2)
    assert packed[0, 0] == 2.0
    assert np.isnan(packed[0, 1])
    assert packed[1, 1] == 4.0


def test_maps_falls_back_when_rgb_pre_fails() -> None:
    from ugv_perception.adapter.output import AdapterError
    from ugv_perception.backend.depth_live import DepthChannel

    class _Backend:
        def __init__(self) -> None:
            self.from_rgb = 0
            self.runs = 0
            self.rgb_pre_disabled = False

        def ensure_hw(self, height: int, width: int) -> None:
            self.hw = (height, width)

        def run_all_from_rgb(self, rgb, sizes, mean, std):
            self.from_rgb += 1
            self.rgb_pre_disabled = True
            raise AdapterError("rgb pre failed")

        def run_all(self, blob):
            self.runs += 1
            mh, mw = int(blob.shape[2]), int(blob.shape[3])
            return [
                np.ones((mh, mw), dtype=np.float32),
                np.zeros((mh, mw), dtype=np.float32),
            ]

    backend = _Backend()
    ch = DepthChannel(backend)
    rgb = np.zeros((28, 42, 3), dtype=np.uint8)
    depth, xyz = ch.maps(rgb, _k(28, 42))
    assert backend.from_rgb == 1
    assert backend.runs == 1
    assert depth.shape == (28, 42)
    assert xyz.shape[1] == 3
    ch.maps(rgb, _k(28, 42))
    assert backend.from_rgb == 1
    assert backend.runs == 2


def test_maps_does_not_retry_infer_after_gpu_run_fails() -> None:
    from ugv_perception.adapter.output import AdapterError
    from ugv_perception.backend.depth_live import DepthChannel

    class _Backend:
        rgb_pre_disabled = False

        def ensure_hw(self, height: int, width: int) -> None:
            return None

        def run_all_from_rgb(self, rgb, sizes, mean, std):
            raise AdapterError("OpenVINO GPU run failed")

        def run_all(self, blob):
            raise AssertionError("must not retry infer after GPU run failed")

    ch = DepthChannel(_Backend())
    try:
        ch.maps(np.zeros((28, 42, 3), dtype=np.uint8), _k(28, 42))
    except AdapterError as exc:
        assert "run failed" in str(exc)
        return
    raise AssertionError("GPU infer failure must raise")


def test_maps_prefers_cuda_metres_then_falls_back_to_gpu_preprocess_for_good() -> None:
    """CUDA: run_depth_metres first. Once it fails, the channel keeps the base GPU preprocess path (never loses it);
    a backend without run_depth_metres (OpenVINO on Arc) goes straight to run_all_from_rgb."""
    from ugv_perception.adapter.output import AdapterError
    from ugv_perception.backend.depth_live import DepthChannel

    class _Backend:
        rgb_pre_disabled = False

        def __init__(self, metres_fails: bool) -> None:
            self.calls: list[str] = []
            self.metres_fails = metres_fails

        def ensure_hw(self, height: int, width: int) -> None:
            return None

        def run_depth_metres(self, rgb, focal, model_hw_, out_hw):
            self.calls.append("metres")
            if self.metres_fails:
                raise AdapterError("CUDA run failed")
            return np.full(out_hw, 2.0, dtype=np.float32)

        def run_all_from_rgb(self, rgb, sizes, mean, std):
            self.calls.append("from_rgb")
            mh, mw = sizes[-1]
            return [np.ones((mh, mw), dtype=np.float32), np.zeros((mh, mw), dtype=np.float32)]

    rgb, k = np.zeros((28, 42, 3), dtype=np.uint8), _k(28, 42)
    ok = _Backend(metres_fails=False)
    DepthChannel(ok).maps(rgb, k)
    assert ok.calls == ["metres"]
    broken = _Backend(metres_fails=True)
    ch = DepthChannel(broken)
    depth, _ = ch.maps(rgb, k)
    ch.maps(rgb, k)
    assert broken.calls == ["metres", "from_rgb", "from_rgb"]
    assert depth.shape == (28, 42)

    class _Arc(_Backend):
        run_depth_metres = None

    arc = _Arc(metres_fails=False)
    DepthChannel(arc).maps(rgb, k)
    assert arc.calls == ["from_rgb"]


def test_depth_live_keeps_hole_safe_on_cpu() -> None:
    text = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "backend"
        / "depth_live.py"
    ).read_text()
    assert "hole_safe_resize" in text
    assert "openvino" not in text
    assert "torch" not in text


def test_export_wrapper_stops_before_sky_fill() -> None:
    text = (
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts"
        / "export_da3metric_openvino.py"
    ).read_text()
    assert "class Da3HeadExport" in text
    assert "_process_depth_head" in text
    assert "_process_mono_sky_estimation(" not in text.split("class Da3HeadExport", 1)[1]


# --- GPU twins of the pre/post-processing (torch). Same maths as the numpy reference above. ---------------


def _torch_devices() -> list[str]:
    from ugv_perception.backend.device import cuda_available

    return ["cpu"] + (["cuda"] if cuda_available() else [])


@pytest.fixture(params=_torch_devices())
def device(request: pytest.FixtureRequest) -> str:
    pytest.importorskip("torch")
    return request.param


def _numpy_depth_for(raw: np.ndarray, sky: np.ndarray, focal: float, out_hw: tuple[int, int]) -> np.ndarray:
    return hole_safe_resize(meters_from_raw(raw, focal), sky, out_hw).astype(np.float32)


def test_gpu_plane_matches_numpy_to_under_a_millimetre(device: str) -> None:
    import torch

    from ugv_perception.depth.geometry_gpu import hole_safe_resize_gpu

    h, w = 480, 640
    k_cam = _k(h, w, fx=594.58, fy=596.22)
    km, (mh, mw) = k_model(k_cam, (h, w))
    focal = focal_model(km)
    raw = np.full((mh, mw), 2.0 * 300.0 / focal, dtype=np.float32)
    sky = np.zeros((mh, mw), dtype=np.float32)
    want = _numpy_depth_for(raw, sky, focal, (h, w))
    metres = torch.from_numpy(raw).to(device) * (focal / 300.0)
    got = hole_safe_resize_gpu(metres, torch.from_numpy(sky).to(device), (h, w)).cpu().numpy()
    assert got.dtype == np.float32 and got.shape == (h, w)
    assert np.array_equal(np.isnan(got), np.isnan(want))
    assert float(np.max(np.abs(got - want))) < 1e-3
    xyz = backproject(got, k_cam)
    center = xyz[np.argmin(np.abs(xyz[:, 0]) + np.abs(xyz[:, 1]))]
    assert abs(float(center[2]) - 2.0) < 1e-3


def test_gpu_resize_matches_numpy_with_holes_and_sky(device: str) -> None:
    import torch

    from ugv_perception.depth.geometry_gpu import hole_safe_resize_gpu

    rng = np.random.default_rng(3)
    mh, mw = 378, 504
    metres = rng.uniform(0.5, 9.0, (mh, mw)).astype(np.float32)
    metres[40:60, 100:140] = np.nan
    sky = np.zeros((mh, mw), dtype=np.float32)
    sky[0:90, :] = 0.8
    sky[200, 200] = SKY_THRESHOLD
    want = hole_safe_resize(metres, sky, (480, 640)).astype(np.float32)
    got = hole_safe_resize_gpu(
        torch.from_numpy(metres).to(device), torch.from_numpy(sky).to(device), (480, 640)
    ).cpu().numpy()
    assert np.isnan(want).any() and np.isfinite(want).any()
    assert np.array_equal(np.isnan(got), np.isnan(want))
    ok = np.isfinite(want)
    assert float(np.max(np.abs(got[ok] - want[ok]))) < 1e-3


@pytest.mark.parametrize("hw", [(480, 640), (408, 612), (480, 600), (28, 28)])
def test_gpu_preprocess_matches_numpy(device: str, hw: tuple[int, int]) -> None:
    from ugv_perception.depth.geometry_gpu import preprocess_nchw_gpu

    rgb = np.random.default_rng(5).integers(0, 256, (hw[0], hw[1], 3), dtype=np.uint8)
    want, want_hw = preprocess_nchw(rgb)
    got_t, got_hw = preprocess_nchw_gpu(rgb, device)
    got = got_t.cpu().numpy()
    assert got_hw == want_hw
    assert got.shape == want.shape and got.dtype == np.float32
    diff = np.abs(got - want)
    # The reference truncates to 8 bits. Where the exact result is an integer, float64 noise (126.99999999999999)
    # truncates it down and the float32 path (127.0) does not: one 8-bit step, 1/255/0.225 = 0.0175, on well
    # under 1 % of the values of a white-noise image (the worst case; a photo has far fewer exact integers).
    assert float(diff.max()) <= 0.02
    assert float(np.mean(diff > 1e-4)) < 5e-3


def test_depth_channel_uses_the_backends_gpu_path() -> None:
    from ugv_perception.backend.depth_live import DepthChannel

    class _GpuBackend:
        def __init__(self) -> None:
            self.args: tuple | None = None

        def ensure_hw(self, h: int, w: int) -> None:
            raise AssertionError("the GPU path sizes itself")

        def run_all(self, blob):
            raise AssertionError("the GPU path must not use the numpy run_all")

        def run_depth_metres(self, rgb, focal, model_hw_, out_hw):
            self.args = (rgb.shape, round(focal, 3), model_hw_, out_hw)
            return np.full(out_hw, 2.0, dtype=np.float32)

    rgb = np.zeros((28, 28, 3), dtype=np.uint8)
    k = (200.0, 0.0, 13.5, 0.0, 200.0, 13.5, 0.0, 0.0, 1.0)
    backend = _GpuBackend()
    depth_m, points = DepthChannel(backend).maps(rgb, k)
    km, mhw = k_model(np.asarray(k).reshape(3, 3), (28, 28))
    assert backend.args == ((28, 28, 3), round(focal_model(km), 3), mhw, (28, 28))
    assert depth_m.shape == (28, 28) and depth_m.dtype == np.float32
    assert points.shape == (28 * 28, 3) and np.allclose(points[:, 2], 2.0)


# --- Real DA3 weights on CUDA. Skipped when either is missing. ---------------------------------------------

_DA3_DIR = Path(__file__).resolve().parents[3] / "weights" / "da3metric-large"
_K_WEBCAM = np.array(
    [[594.58444, 0.0, 311.15451], [0.0, 596.2187, 232.11396], [0.0, 0.0, 1.0]], dtype=np.float64
)


def _synthetic_frames() -> list[np.ndarray]:
    """Three 640x480 frames with different structure: blocks over noise, a ramp with boxes, smooth colour."""
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:480, 0:640]

    def upsampled(grid: np.ndarray) -> np.ndarray:
        ys = np.linspace(0, grid.shape[0] - 1, 480)
        xs = np.linspace(0, grid.shape[1] - 1, 640)
        y0 = np.minimum(ys.astype(int), grid.shape[0] - 2)
        x0 = np.minimum(xs.astype(int), grid.shape[1] - 2)
        wy = (ys - y0)[:, None, None]
        wx = (xs - x0)[None, :, None]
        rows = grid[y0] * (1 - wy) + grid[y0 + 1] * wy
        return rows[:, x0] * (1 - wx) + rows[:, x0 + 1] * wx

    blocks = 0.55 * upsampled(rng.random((10, 14, 3))) + 0.45 * np.linspace(0, 1, 480)[:, None, None]
    for _ in range(14):
        r, c = int(rng.integers(0, 420)), int(rng.integers(0, 560))
        blocks[r : r + int(rng.integers(20, 60)), c : c + int(rng.integers(30, 80))] = rng.random(3)
    blocks = np.clip((blocks + rng.normal(0, 0.03, blocks.shape)) * 255.0, 0, 255)

    ramp = np.stack([xx / 640 * 255, yy / 480 * 255, (xx + yy) / 1120 * 255], axis=2)
    for _ in range(12):
        r, c = int(rng.integers(0, 400)), int(rng.integers(0, 540))
        ramp[r : r + int(rng.integers(30, 80)), c : c + int(rng.integers(40, 100))] = rng.random(3) * 255
    ramp = np.clip(ramp + rng.normal(0, 4, ramp.shape), 0, 255)

    smooth = upsampled(rng.random((6, 8, 3))) * 255.0
    return [f.astype(np.uint8) for f in (blocks, ramp, smooth)]


@pytest.fixture(scope="module")
def da3_cuda():
    from ugv_perception.backend.device import cuda_available

    if not cuda_available():
        pytest.skip("CUDA missing")
    if not (_DA3_DIR / "model.safetensors").is_file():
        pytest.skip("DA3 safetensors missing")
    from ugv_perception.backend.cuda_pytorch import CudaPytorchTensorBackend

    backend = CudaPytorchTensorBackend()
    backend.load(str(_DA3_DIR), kind="da3")
    return backend


class _NumpyPath:
    """Hides run_depth_metres so DepthChannel takes the numpy pre/post-processing."""

    def __init__(self, backend: object) -> None:
        self._backend = backend

    def ensure_hw(self, h: int, w: int) -> None:
        self._backend.ensure_hw(h, w)

    def run_all(self, blob):
        return self._backend.run_all(blob)


@pytest.mark.filterwarnings("ignore:.*torch.jit.script.*")  # raised by the DA3 package at import, not by us
def test_gpu_path_matches_the_numpy_path_on_real_weights(da3_cuda) -> None:
    from ugv_perception.backend.depth_live import DepthChannel

    for rgb in _synthetic_frames():
        want, want_pts = DepthChannel(_NumpyPath(da3_cuda)).maps(rgb, _K_WEBCAM)
        got, got_pts = DepthChannel(da3_cuda).maps(rgb, _K_WEBCAM)
        assert got.dtype == np.float32 and got.shape == want.shape == (480, 640)
        assert np.array_equal(np.isnan(got), np.isnan(want))
        ok = np.isfinite(want)
        assert ok.sum() > 0.9 * ok.size
        # Measured on these frames: 0.2 to 0.7 mm. The difference is the 8-bit truncation flips in the
        # resize, which the model turns into sub-millimetre depth noise. 2 mm and 1 % are the bounds.
        assert float(np.max(np.abs(got[ok] - want[ok]))) < 2e-3
        assert float(np.max(np.abs(got[ok] - want[ok]) / want[ok])) < 1e-2
        assert got_pts.shape == want_pts.shape


# --- FP32 only: the CUDA backend has no half-precision path (PR #40 review). -------------------------------


def test_the_cuda_backend_has_no_half_precision_path() -> None:
    from ugv_perception.backend import cuda_pytorch

    source = Path(cuda_pytorch.__file__).read_text(encoding="utf-8")
    for word in ("autocast", "float16", "half(", "da3_half", "synchronize("):
        assert word not in source, word


def test_cuda_hot_paths_use_a_private_stream() -> None:
    """RUGD and DA3 each own a stream so a host copy on one thread does not wait for the other net."""
    from ugv_perception.backend import cuda_pytorch

    source = Path(cuda_pytorch.__file__).read_text(encoding="utf-8")
    assert "torch.cuda.Stream()" in source
    for name in ("_rgb_pre_cuda", "_run_seg_tensor", "begin_depth_metres", "wait_depth_metres", "_blob_to_cuda"):
        assert name in source
    assert source.count("_cuda_stream()") >= 5


def test_cuda_rugd_and_da3_backends_get_distinct_streams() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from ugv_perception.backend.cuda_pytorch import CudaPytorchTensorBackend

    rugd, da3 = CudaPytorchTensorBackend(), CudaPytorchTensorBackend()
    rugd._stream = torch.cuda.Stream()
    da3._stream = torch.cuda.Stream()
    assert rugd._stream is not da3._stream
    with rugd._cuda_stream():
        assert torch.cuda.current_stream() == rugd._stream
    with da3._cuda_stream():
        assert torch.cuda.current_stream() == da3._stream


class _StubOut:
    def __init__(self, logits) -> None:
        self.logits = logits


# --- One definition of the DA3 preprocess constants and the half-pixel positions. --------------------------


def test_half_pixel_positions_are_the_bilinear_sample_positions_with_edge_clamp() -> None:
    from ugv_perception.depth.geometry import half_pixel_positions

    assert half_pixel_positions(4, 2).tolist() == [0.5, 2.5]
    assert half_pixel_positions(2, 4).tolist() == [0.0, 0.25, 0.75, 1.0]  # clamped to [0, n_src - 1]
    assert half_pixel_positions(3, 3).tolist() == [0.0, 1.0, 2.0]


def test_the_gpu_module_uses_the_numpy_reference_definitions_not_copies() -> None:
    pytest.importorskip("torch")
    from ugv_perception.depth import geometry, geometry_gpu

    assert geometry_gpu.half_pixel_positions is geometry.half_pixel_positions
    assert geometry_gpu.IMAGENET_MEAN is geometry.IMAGENET_MEAN
    assert geometry_gpu.IMAGENET_STD is geometry.IMAGENET_STD
    assert not hasattr(geometry_gpu, "_MEAN") and not hasattr(geometry_gpu, "_STD")
    assert geometry.IMAGENET_MEAN == (0.485, 0.456, 0.406)
    assert geometry.IMAGENET_STD == (0.229, 0.224, 0.225)


# --- The CUDA backend places everything on its own device, not on a hard-coded "cuda". ---------------------


class _DeviceProbe:
    """A model stub that records where its input was put."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.devices: list[str] = []

    def __call__(self, pixel_values):
        import torch

        self.devices.append(pixel_values.device.type)
        n, _, h, w = pixel_values.shape
        if self.kind == "rugd":
            return _StubOut(torch.zeros((n, 25, h // 4, w // 4), device=pixel_values.device))
        return (
            torch.ones((n, h, w), device=pixel_values.device),
            torch.zeros((n, h, w), device=pixel_values.device),
        )


def _cpu_backend(kind: str):
    pytest.importorskip("torch")
    from ugv_perception.backend.cuda_pytorch import CudaPytorchTensorBackend

    backend = CudaPytorchTensorBackend()
    backend.device = "cpu"  # the backend's own setting; every placement must follow it
    backend._model = _DeviceProbe(kind)
    backend._kind = kind
    return backend


def test_rugd_paths_put_the_input_on_the_backend_device() -> None:
    backend = _cpu_backend("rugd")
    blob = np.zeros((1, 3, 64, 64), dtype=np.float32)
    backend.run_seg(blob, (64, 64))
    backend.run_all(blob)
    assert backend._model.devices == ["cpu", "cpu"]


def test_da3_paths_put_the_input_on_the_backend_device_and_need_no_cuda_to_run() -> None:
    backend = _cpu_backend("da3")
    backend.run_all(np.zeros((1, 3, 28, 28), dtype=np.float32))
    out = backend.run_depth_metres(
        np.zeros((28, 28, 3), dtype=np.uint8), 100.0, (504, 504), (28, 28)
    )
    assert backend._model.devices == ["cpu", "cpu"]
    assert out.shape == (28, 28) and out.dtype == np.float32
