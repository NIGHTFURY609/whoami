"""webcam_stream.py --video helpers: output size, pacing, and the generated placeholder calibration.

scripts/ is not a package (the script runs on the Windows host), so it is loaded by path.
"""

from __future__ import annotations

import base64
import importlib.util
import math
import struct
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from ugv_localization.camera import load_calibration

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "webcam_stream.py"
_spec = importlib.util.spec_from_file_location("webcam_stream", _SCRIPT)
ws = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ws)


@pytest.mark.parametrize(
    ("src", "width", "want"),
    [
        ((1920, 1080), 640, (640, 360)),  # 16:9 keeps its aspect, nothing cropped
        ((1280, 960), 640, (640, 480)),
        ((1000, 567), 640, (640, 362)),  # 362.88 -> even 362
        ((1366, 768), 640, (640, 360)),  # 359.8 -> 360
        ((1920, 1080), 0, (1920, 1080)),  # 0 keeps the native size
        ((480, 360), 640, (480, 360)),  # never upscaled
    ],
)
def test_output_size_keeps_aspect(src, width, want) -> None:
    assert ws.output_size(*src, width) == want


def test_output_size_rejects_an_empty_video() -> None:
    with pytest.raises(ValueError):
        ws.output_size(0, 480, 640)


def test_pacing_follows_the_file_rate_times_speed() -> None:
    assert ws.frame_due_s(30, 30.0, 1.0) == pytest.approx(1.0)
    assert ws.frame_due_s(30, 30.0, 0.5) == pytest.approx(2.0)  # slow motion: twice as long
    assert ws.frame_due_s(0, 25.0, 1.0) == 0.0


@pytest.mark.parametrize("reported", [0.0, math.nan, -1.0, math.inf])
def test_unusable_reported_fps_falls_back(reported) -> None:
    assert ws.source_fps(reported, 30.0) == 30.0


def test_reported_fps_is_used() -> None:
    assert ws.source_fps(59.94, 30.0) == 59.94


@pytest.mark.parametrize(("size", "hfov", "fx"), [((640, 360), 90.0, 320.0), ((640, 480), 120.0, 320 / math.sqrt(3))])
def test_generated_calibration_is_a_loadable_placeholder(tmp_path: Path, size, hfov, fx) -> None:
    path = tmp_path / "rc.yaml"
    path.write_text(ws.placeholder_calibration(*size, hfov, "clip.mp4, 1920x1080"), encoding="utf-8")
    cal = load_calibration(path)
    assert cal.placeholder is True
    assert (cal.width, cal.height) == size
    assert cal.k[0] == pytest.approx(fx, rel=1e-6) and cal.k[4] == pytest.approx(fx, rel=1e-6)
    assert (cal.k[2], cal.k[5]) == (size[0] / 2, size[1] / 2)
    assert cal.d == (0.0,) * 5


@pytest.mark.parametrize("hfov", [0.0, 180.0, -10.0])
def test_generated_calibration_refuses_an_impossible_fov(hfov) -> None:
    with pytest.raises(ValueError):
        ws.placeholder_calibration(640, 360, hfov, "clip.mp4")


class FakeVideo:
    """n frames (their index as the payload), a rewind, and an encoder that passes them through."""

    def __init__(self, n: int) -> None:
        self.n, self.pos = n, 0

    def read(self):
        if self.pos >= self.n:
            return False, None
        self.pos += 1
        return True, self.pos - 1

    def rewind(self) -> None:
        self.pos = 0


def ready_link(cache=None, clock=time.time, offset: float = 0.0) -> "ws.PerceptionLink":
    """A link that is connected, with the container clock `offset` s ahead of the host's."""
    link = ws.PerceptionLink("ws://unused", cache, clock=clock)
    link.connected = True
    t = clock()
    link.note_offset(t, t, t + offset)
    return link


def image_msg(topic: str, stamp_s: float, encoding: str, data: bytes, w: int, h: int, step: int | None = None) -> dict:
    ns = round(stamp_s * 1e9)
    return {"op": "publish", "topic": topic, "msg": {
        "header": {"stamp": {"sec": ns // 1_000_000_000, "nanosec": ns % 1_000_000_000}, "frame_id": "cam"},
        "width": w, "height": h, "step": step or len(data) // h, "encoding": encoding, "is_bigendian": 0,
        "data": base64.b64encode(data).decode()}}


def mask_for_last_sent(link, offset: float = 0.0, value: int = 1) -> dict:
    t_sent, _ = link._sent._log[-1]
    return image_msg(link.MASK, t_sent + 0.005 + offset, "mono8", bytes([value]) * 8, 4, 2)


def _player(n: int, **kw):
    video, latest, stop = FakeVideo(n), ws.Latest(), threading.Event()
    kw.setdefault("fps", 1000.0)
    player = ws.Player(video.read, video.rewind, lambda f: f"frame{f}".encode(), speed=1.0, frames=n,
                       latest=latest, **kw)
    thread = threading.Thread(target=player.run, args=(stop,), daemon=True)
    thread.start()
    return player, latest, stop


def _wait(cond, timeout_s: float = 3.0) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return cond()


def test_player_waits_for_play_then_plays_to_the_end() -> None:
    player, latest, stop = _player(5)
    try:
        time.sleep(0.1)
        assert player.status()["state"] == "ready" and player.status()["frame"] == 1
        assert latest.wait_newer(0, 0.0) == (1, b"frame0")  # only the poster frame before PLAY
        assert latest.wait_newer(1, 0.1) == (1, None)
        player.play()
        assert _wait(lambda: player.status()["state"] == "ended")
        assert player.status()["frame"] == 5
        assert latest.wait_newer(0, 0.0)[1] == b"frame4"
    finally:
        stop.set()


def test_while_not_playing_the_frame_on_screen_is_resent_to_keep_the_link() -> None:
    player, latest, stop = _player(5, hold_s=0.05)
    try:
        assert _wait(lambda: latest.wait_newer(0, 0.0)[0] >= 4)  # poster, then held copies
        seq, jpeg = latest.wait_newer(0, 0.0)
        assert jpeg == b"frame0" and player.status()["frame"] == 1  # the same frame, the video did not advance
    finally:
        stop.set()


def test_pause_holds_the_stream_and_play_resumes() -> None:
    player, latest, stop = _player(1000, fps=50.0, autoplay=True)
    try:
        assert _wait(lambda: player.status()["frame"] >= 3)
        player.pause()
        time.sleep(0.1)
        held = player.status()["frame"]
        time.sleep(0.2)
        assert player.status()["state"] == "paused" and player.status()["frame"] == held
        player.play()
        assert _wait(lambda: player.status()["frame"] > held)
    finally:
        stop.set()


def test_replay_and_play_after_the_end_start_over() -> None:
    player, latest, stop = _player(4, autoplay=True)
    try:
        assert _wait(lambda: player.status()["state"] == "ended")
        player.play()  # PLAY on an ended video replays it
        assert _wait(lambda: player.status()["state"] == "ended" and latest.wait_newer(0, 0.0)[0] >= 8)
        player.replay()
        assert _wait(lambda: latest.wait_newer(0, 0.0)[0] >= 12)
    finally:
        stop.set()


def test_sync_waits_for_the_mask_of_the_frame_it_sent() -> None:
    link = ready_link()
    player, latest, stop = _player(10, link=link, autoplay=True, gate_timeout_s=30.0)
    try:
        assert _wait(lambda: player.status()["state"] == "buffering")
        time.sleep(0.1)
        assert player.status()["frame"] == 1  # frame 0 sent, frame 1 held until frame 0's mask is here
        link.on_message(mask_for_last_sent(link))
        assert _wait(lambda: player.status()["frame"] == 2 and player.status()["state"] == "buffering")
        player.set_sync(False)
        link.on_message(mask_for_last_sent(link))  # releases the frame already waiting; from then on real time
        assert _wait(lambda: player.status()["state"] == "ended")
    finally:
        stop.set()


def test_sync_without_a_ready_link_plays_in_real_time() -> None:
    link = ws.PerceptionLink("ws://unused")  # never started: not connected
    player, latest, stop = _player(6, link=link, autoplay=True)
    try:
        assert _wait(lambda: player.status()["state"] == "ended")
        assert player.status()["perception"] is False
    finally:
        stop.set()


def test_sent_log_matches_a_stamp_to_the_frame_sent_just_before() -> None:
    log = ws.SentLog(max_lag_s=0.1, lead_s=0.02)
    log.add(10.00, 4)
    log.add(10.25, 5)  # sync play: frames far apart
    assert log.match(10.01) == 4 and log.match(10.26) == 5
    assert log.match(10.50) is None  # nothing sent shortly before
    log.add(12.00, 5)
    log.add(12.03, 5)  # a held frame sent again: still one frame
    assert log.match(12.04) == 5
    log.add(13.000, 6)
    log.add(13.033, 7)  # real-time play: two frames in the window, ambiguous
    assert log.match(13.04) is None


def test_link_saves_the_mask_depth_and_camera_of_the_matched_frame(tmp_path: Path) -> None:
    cache = ws.OverlayCache(tmp_path / "c", {"file": "clip.mp4"})
    link = ready_link(cache, offset=1000.0)  # the container clock is 1000 s ahead
    link.sent(3, b"jpeg3")
    mask = mask_for_last_sent(link, offset=1000.0, value=2)
    link.on_message(mask)
    depth = np.array([[1.5, np.nan, 0.0, 70.0], [2.0, 2.5, 3.0, 4.0]], "<f4")
    stamp = mask["msg"]["header"]["stamp"]
    link.on_message(image_msg(link.DEPTH, stamp["sec"] + stamp["nanosec"] / 1e9, "32FC1", depth.tobytes(), 4, 2))
    link.on_message({"op": "publish", "topic": link.INFO, "msg": {
        "header": {"frame_id": "cam"}, "width": 4, "height": 2, "k": [1.0] * 9, "d": [], "distortion_model": ""}})
    assert cache.read_frame(3) == b"jpeg3"
    assert cache.read_mask(3) == (4, 2, bytes([2]) * 8)
    w, h, raw = cache.read_depth(3)
    got = np.frombuffer(raw, "<f4").reshape(h, w)
    assert got[0, 0] == pytest.approx(1.5) and np.isnan(got[0, 1]) and np.isnan(got[0, 2])
    assert got[0, 3] == pytest.approx(65.535)  # beyond 16-bit millimetres: clipped
    summary = cache.summary(10)
    assert (summary["frames"], summary["masks"], summary["depths"]) == (1, 1, 1)
    assert summary["camera"]["frame_id"] == "cam" and summary["complete"] is False


def test_link_ignores_a_mask_it_cannot_place(tmp_path: Path) -> None:
    cache = ws.OverlayCache(tmp_path / "c", {"file": "clip.mp4"})
    link = ready_link(cache)
    link.sent(1, b"a")
    link.on_message(image_msg(link.MASK, time.time() + 5.0, "mono8", bytes(8), 4, 2))  # 5 s later: no frame
    assert cache.read_mask(1) is None


def test_cache_keeps_the_first_overlay_and_is_wiped_for_another_video(tmp_path: Path) -> None:
    root = tmp_path / "c"
    cache = ws.OverlayCache(root, {"file": "a.mp4"})
    assert cache.put_mask(0, 2, 1, bytes([1, 2]))
    assert not cache.put_mask(0, 2, 1, bytes([0, 0]))  # a frame is never rewritten
    assert not cache.put_mask(1, 2, 1, bytes([7, 0]))  # not a {0,1,2} mask
    assert ws.OverlayCache(root, {"file": "a.mp4"}).read_mask(0) == (2, 1, bytes([1, 2]))  # kept across runs
    assert ws.OverlayCache(root, {"file": "b.mp4"}).read_mask(0) is None


def test_image_payload_drops_row_padding() -> None:
    msg = image_msg("/x", 1.0, "mono8", bytes([1, 2, 9, 3, 4, 9]), 2, 2, step=3)["msg"]
    assert ws.image_payload(msg, "mono8", 1) == (2, 2, bytes([1, 2, 3, 4]))
    assert ws.image_payload(msg, "32FC1", 4) is None


def test_nearest_sample_matches_the_ui_grid_sampling() -> None:
    w, h = 8, 4
    img = np.arange(w * h, dtype=np.uint8).reshape(h, w)
    tw, th, data = ws.nearest_sample((w, h, img.tobytes()), "w=4&h=2", 1)
    got = np.frombuffer(data, np.uint8).reshape(th, tw)
    want = [[img[(v * h) // th, (u * w) // tw] for u in range(tw)] for v in range(th)]
    assert (tw, th) == (4, 2) and got.tolist() == want
    depth = np.arange(w * h, dtype="<f4").reshape(h, w)
    assert ws.nearest_sample((w, h, depth.tobytes()), "w=4&h=2", 4)[2] == depth[[0, 2]][:, [0, 2, 4, 6]].tobytes()
    assert ws.nearest_sample((w, h, img.tobytes()), "", 1) == (w, h, img.tobytes())  # no size: as it is
    assert ws.nearest_sample((w, h, img.tobytes()), "w=16&h=8", 1)[:2] == (w, h)  # never upsampled


def test_bundle_carries_a_frame_with_its_overlay_and_is_kept_in_memory(tmp_path: Path) -> None:
    cache = ws.OverlayCache(tmp_path / "c", {"file": "clip.mp4"})
    cache.put_frame(5, b"JPEG")
    assert cache.put_mask(5, 4, 2, bytes([0, 1, 2, 1, 1, 1, 2, 0]))
    cache.put_depth(5, 4, 2, np.full((2, 4), 2.0, "<f4").tobytes())
    out = cache.bundle(5, "w=2&h=1")
    jl, mw, mh, dw, dh = struct.unpack_from("<5I", out)
    assert (jl, mw, mh, dw, dh) == (4, 2, 1, 2, 1)
    body = out[20:]
    assert body[:4] == b"JPEG" and body[4:6] == bytes([0, 2])
    assert np.frombuffer(body[6:], "<f4").tolist() == [2.0, 2.0]
    (tmp_path / "c" / "mask" / "000005.png").unlink()
    assert cache.bundle(5, "w=2&h=1") == out  # served from memory
    cache.put_frame(6, b"J6")
    assert struct.unpack_from("<5I", cache.bundle(6, ""))[1:] == (0, 0, 0, 0)  # no overlay saved: frame only
    assert cache.bundle(7, "") is None
