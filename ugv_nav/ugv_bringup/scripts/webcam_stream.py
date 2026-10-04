"""Windows host -> Docker camera bridge for profile live_cam (Dev 5).

ROS runs in Docker / WSL, which cannot open a Windows USB webcam. This runs on the Windows host, owns the
webcam, and serves it as an MJPEG HTTP stream; the camera driver in the container opens it with
    device:=http://host.docker.internal:8090/cam.mjpg

Every client gets the newest frame only (no backlog), so a slow or late reader never sees stale frames.
The driver still stamps each frame on arrival (ugv_bringup README).

    python ugv_nav/ugv_bringup/scripts/webcam_stream.py [--index 0] [--width 640 --height 480] [--port 8090]

Needs Windows Python with opencv-python. --index is the DirectShow order
(ffmpeg -hide_banner -list_devices true -f dshow -i dummy). Width/height must match the calibration.

Phone browser camera instead of the webcam (the UI's phone.html through a tunnel; ugv_bringup README):

    python .../webcam_stream.py --phone [--width 640 --height 480] [--ingest-port 8091] [--stall-s 2]

The phone sends JPEG frames over a WebSocket (loopback :8091, reached through the UI dev server's /phone/ingest
proxy); frames that are not exactly --width x --height are dropped. /cam.mjpg answers 503 until a fresh phone frame
exists and ends when the phone leaves or stalls, so the driver reopens; /phone/status reports the counts.

Recorded video instead of the webcam (eval only, e.g. an RC car's camera; ugv_bringup README):

    python .../webcam_stream.py --video clip.mp4 --calibration-out cal.yaml [--width 640] [--hfov-deg 90]
    python .../webcam_stream.py --video clip.mp4 [--width 640] [--speed 1.0] [--loop]

The video keeps its aspect ratio and whole field of view: it is only scaled to --width (height from the aspect,
rounded to even; --width 0 or a narrower source keeps the native size; --height is ignored). It waits at frame 0
until PLAY (the web UI's video panel, or POST /video/play; --autoplay starts at once), so the stack can come up
first; pause and replay work the same way. Frames are put at the file's own rate times --speed. In sync mode (the
default) each frame also waits until Dev 1's perception has finished it (/segmentation/port_meta over --rosbridge),
so no frame goes without an overlay and slow perception stretches the playback ("buffering") instead of dropping
frames; with rosbridge unreachable it plays in real time. Ready, paused or ended, the frame on screen is sent again
every 2 s (a stopped car with a frozen view), because OpenCV drops a network stream that stays silent for about
30 s and the driver would never get it back; the arbiter still holds. --loop rewinds at the end instead, which
visual odometry sees as a teleport.

Overlays are recorded as they are made: during a sync play every frame's JPEG, Dev 1's mask and the depth image
(matched to the frame by its image stamp) are saved in <video stem>.overlays/ beside the video (--overlay-dir,
--no-record). The web UI plays that recording back from this bridge (/video/cache/...) at the video's own rate
without the ROS stack, as often as wanted; a later sync play only fills frames still missing. --calibration-out writes a `placeholder: true` calibration for exactly that output size
from an assumed horizontal FOV and exits: the driver then accepts the stream with allow_placeholder_calibration:=true.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import socket
import struct
import sys
import threading
import time
from collections import OrderedDict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

BOUNDARY = b"ugvframe"


class Latest:
    """The newest JPEG and a sequence number; readers block until a newer one exists.

    `cut()` (phone mode: the phone left or stalled) drops the frame and bumps `generation`, which wakes every reader
    so the stream responses end and the camera driver reopens instead of waiting on a dead connection."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq = 0
        self._gen = 0

    def put(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg, self._seq = jpeg, self._seq + 1
            self._cond.notify_all()

    def cut(self) -> None:
        with self._cond:
            self._jpeg, self._gen = None, self._gen + 1
            self._cond.notify_all()

    @property
    def generation(self) -> int:
        with self._cond:
            return self._gen

    def wait_newer(self, seq: int, timeout: float, gen: int | None = None) -> tuple[int, bytes | None]:
        with self._cond:
            self._cond.wait_for(lambda: self._seq > seq or (gen is not None and self._gen != gen), timeout=timeout)
            return self._seq, (self._jpeg if self._seq > seq else None)


def capture_loop(cap: cv2.VideoCapture, latest: Latest, quality: int, stop: threading.Event) -> None:
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    failures = 0
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok or frame is None:
            failures += 1
            if failures == 30:
                print("webcam_stream: camera is not delivering frames", file=sys.stderr)
            time.sleep(0.01)
            continue
        failures = 0
        ok, jpg = cv2.imencode(".jpg", frame, params)
        if ok:
            latest.put(jpg.tobytes())


def output_size(src_w: int, src_h: int, width: int) -> tuple[int, int]:
    """Size the video is served at: scaled to `width` keeping the aspect (height rounded to even, at least 2).
    `width` <= 0, or a source no wider than it, keeps the native size (never upscaled)."""
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"video size {src_w}x{src_h} is not valid")
    if width <= 0 or width >= src_w:
        return src_w, src_h
    return width, max(2, 2 * round(src_h * width / src_w / 2))


def source_fps(reported: float, fallback: float) -> float:
    """The file's frame rate, or `fallback` when the container does not report a usable one (0 or NaN)."""
    return reported if math.isfinite(reported) and reported > 0.0 else fallback


def frame_due_s(index: int, fps: float, speed: float) -> float:
    """Seconds after the start at which frame `index` is put: the file's timing scaled by `speed`."""
    return index / (fps * speed)


def placeholder_calibration(width: int, height: int, hfov_deg: float, source: str) -> str:
    """A `placeholder: true` calibration (camera_info_manager YAML) for a camera of unknown intrinsics: square
    pixels, principal point at the centre, fx = fy = (W/2) / tan(HFOV/2), no distortion. Not a calibration."""
    if not 0.0 < hfov_deg < 180.0:
        raise ValueError(f"hfov_deg must be between 0 and 180, got {hfov_deg}")
    f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    cx, cy = width / 2, height / 2
    return f"""\
# Camera: RC car camera, from a recorded video ({source}), served at {width}x{height} by webcam_stream.py --video.
#
# THIS IS NOT A CALIBRATION. The camera is not available, so K assumes a horizontal field of view of {hfov_deg:g} deg
# at this size: fx = fy = (W/2) / tan(HFOV/2) = {f:.5f}, principal point at the centre, no distortion. A wide-angle
# lens's barrel distortion is not modelled. Depth (DA3 converts with fx), the depth cloud and the RTAB-Map scale are
# off in proportion to the focal-length error. Eval only: never a mapping run that counts or an autonomous run.
# Regenerate with the real FOV: webcam_stream.py --video <clip> --calibration-out <this file> --hfov-deg <deg>
placeholder: true
image_width: {width}
image_height: {height}
camera_name: rc_car
camera_matrix:
  rows: 3
  cols: 3
  data: [{f:.5f}, 0.0, {cx:.5f},
         0.0, {f:.5f}, {cy:.5f},
         0.0, 0.0, 1.0]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [0.0, 0.0, 0.0, 0.0, 0.0]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0]
projection_matrix:
  rows: 3
  cols: 4
  data: [{f:.5f}, 0.0, {cx:.5f}, 0.0,
         0.0, {f:.5f}, {cy:.5f}, 0.0,
         0.0, 0.0, 1.0, 0.0]
"""


class SentLog:
    """When each video frame was handed to the stream (host wall clock, s), to find the frame a mask belongs to.

    The camera driver stamps a frame when it arrives, a few ms after it was sent, so a mask stamped at host time t
    belongs to the frame sent last at or before t (+`lead_s` for the clock-offset error), no more than `max_lag_s`
    earlier. If another frame was also sent in that window (real-time play, 33 ms apart) the match is ambiguous and
    nothing is returned: overlays are only recorded when frames are far apart (sync play, held frames)."""

    def __init__(self, keep: int = 512, max_lag_s: float = 0.1, lead_s: float = 0.02) -> None:
        self._log: deque[tuple[float, int]] = deque(maxlen=keep)
        self._max_lag_s, self._lead_s = max_lag_s, lead_s

    def add(self, t: float, index: int) -> None:
        self._log.append((t, index))

    def match(self, t: float) -> int | None:
        window = [i for (ts, i) in self._log if t - self._max_lag_s <= ts <= t + self._lead_s]
        return window[-1] if window and all(i == window[-1] for i in window) else None


def stamp_ns(msg: dict) -> int | None:
    s = msg.get("header", {}).get("stamp", {})
    sec, nsec = s.get("sec"), s.get("nanosec")
    return sec * 1_000_000_000 + nsec if isinstance(sec, int) and isinstance(nsec, int) else None


def image_payload(msg: dict, encoding: str, itemsize: int) -> tuple[int, int, bytes] | None:
    """(width, height, packed rows) of a rosbridge sensor_msgs/Image with this encoding, or None."""
    w, h, step, data = msg.get("width"), msg.get("height"), msg.get("step"), msg.get("data")
    if msg.get("encoding") != encoding or msg.get("is_bigendian") or not isinstance(data, str):
        return None
    if not (isinstance(w, int) and isinstance(h, int) and isinstance(step, int)) or w <= 0 or h <= 0:
        return None
    raw = base64.b64decode(data)
    row = w * itemsize
    if step < row or len(raw) < step * h:
        return None
    return w, h, raw if step == row else b"".join(raw[r * step:r * step + row] for r in range(h))


class OverlayCache:
    """Overlays saved during sync plays, so the video can be shown with them later without the stack.

    `<video stem>.overlays/` next to the video: frames/NNNNNN.jpg (as served), mask/NNNNNN.png (Dev 1's classes
    0/1/2), depth/NNNNNN.png (16-bit millimetres, 0 = no depth) by video frame index, and meta.json (the video it
    belongs to and the camera's CameraInfo). A cache made from another video file or size is wiped on open. Only
    the first overlay of a frame is kept: a later play fills the gaps but never rewrites a frame."""

    VERSION = 1

    def __init__(self, root: Path, video: dict) -> None:
        self.root = root
        self._lock = threading.Lock()
        meta = self._load_meta()
        if meta.get("version") != self.VERSION or meta.get("video") != video:
            self._wipe()
            meta = {"version": self.VERSION, "video": video, "camera": None}
        self._meta = meta
        for d in ("frames", "mask", "depth"):
            (root / d).mkdir(parents=True, exist_ok=True)
        self._write_meta()
        self._have = {d: {int(p.stem) for p in (root / d).iterdir() if p.stem.isdigit()}
                      for d in ("frames", "mask", "depth")}
        self._bundles: OrderedDict[tuple[int, str], bytes] = OrderedDict()

    def _load_meta(self) -> dict:
        try:
            return json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _write_meta(self) -> None:
        (self.root / "meta.json").write_text(json.dumps(self._meta, indent=1), encoding="utf-8")

    def _wipe(self) -> None:
        for d in ("frames", "mask", "depth"):
            for p in (self.root / d).glob("*") if (self.root / d).is_dir() else ():
                p.unlink()

    def clear(self) -> None:
        with self._lock:
            self._wipe()
            self._have = {d: set() for d in self._have}
            self._bundles.clear()

    def _path(self, kind: str, index: int) -> Path:
        return self.root / kind / f"{index:06d}.{'jpg' if kind == 'frames' else 'png'}"

    def put_frame(self, index: int, jpeg: bytes) -> None:
        with self._lock:
            if index not in self._have["frames"]:
                self._path("frames", index).write_bytes(jpeg)
                self._have["frames"].add(index)

    def put_mask(self, index: int, w: int, h: int, data: bytes) -> bool:
        mask = np.frombuffer(data, np.uint8).reshape(h, w)
        if mask.max(initial=0) > 2:
            return False  # not a Perception Port mask
        with self._lock:
            if index in self._have["mask"]:
                return False
            cv2.imwrite(str(self._path("mask", index)), mask)
            self._have["mask"].add(index)
        return True

    def put_depth(self, index: int, w: int, h: int, data: bytes) -> None:
        metres = np.frombuffer(data, "<f4").reshape(h, w)
        mm = np.where(np.isfinite(metres) & (metres > 0), np.clip(np.rint(metres * 1000), 1, 65535), 0)
        with self._lock:
            if index not in self._have["depth"]:
                cv2.imwrite(str(self._path("depth", index)), mm.astype(np.uint16))
                self._have["depth"].add(index)

    def put_camera(self, info: dict) -> None:
        with self._lock:
            if self._meta.get("camera") != info:
                self._meta["camera"] = info
                self._write_meta()

    def summary(self, total: int) -> dict:
        with self._lock:
            masks = len(self._have["mask"])
            return {"frames": len(self._have["frames"]), "masks": masks, "depths": len(self._have["depth"]),
                    "total": total, "complete": total > 0 and masks >= total, "camera": self._meta.get("camera")}

    _BUNDLES = 600  # finished bundles kept in memory (about 120 KB each at the UI's grid size)

    def bundle(self, index: int, query: str) -> bytes | None:
        """Frame `index` in one response, for the UI's recorded playback: a little-endian header of five uint32
        (JPEG length, mask width, mask height, depth width, depth height; 0 x 0 = none saved), then the JPEG, the mask
        (uint8 classes) and the depth (float32 metres, NaN = none), mask and depth sampled as `nearest_sample` does.
        Kept in memory, so a replay costs no decoding. None when the frame itself was not recorded."""
        key = (index, query)
        with self._lock:
            hit = self._bundles.get(key)
            if hit is not None:
                self._bundles.move_to_end(key)
                return hit
        jpeg = self.read_frame(index)
        if jpeg is None:
            return None
        mask = self.read_mask(index)
        depth = self.read_depth(index)
        mask = nearest_sample(mask, query, 1) if mask else (0, 0, b"")
        depth = nearest_sample(depth, query, 4) if depth else (0, 0, b"")
        out = struct.pack("<5I", len(jpeg), mask[0], mask[1], depth[0], depth[1]) + jpeg + mask[2] + depth[2]
        with self._lock:
            if index in self._have["mask"] and (index in self._have["depth"] or depth[0] == 0):
                self._bundles[key] = out  # only complete frames: one still being recorded is read again
                while len(self._bundles) > self._BUNDLES:
                    self._bundles.popitem(last=False)
        return out

    def read_frame(self, index: int) -> bytes | None:
        p = self._path("frames", index)
        return p.read_bytes() if p.is_file() else None

    def read_mask(self, index: int) -> tuple[int, int, bytes] | None:
        m = cv2.imread(str(self._path("mask", index)), cv2.IMREAD_GRAYSCALE)
        return None if m is None else (m.shape[1], m.shape[0], m.tobytes())

    def read_depth(self, index: int) -> tuple[int, int, bytes] | None:
        mm = cv2.imread(str(self._path("depth", index)), cv2.IMREAD_UNCHANGED)
        if mm is None:
            return None
        metres = np.where(mm > 0, mm.astype("<f4") / 1000.0, np.float32("nan")).astype("<f4")
        return metres.shape[1], metres.shape[0], metres.tobytes()


class PerceptionLink:
    """Dev 1's perception output over rosbridge, matched to the video frames that produced it.

    Masks are matched to frames by their image stamp (`SentLog`), with the container's clock offset measured through
    rosapi's get_time (the camera driver stamps with the container clock). The player waits on `wait_frame` in sync
    mode; with an `OverlayCache` every matched mask, its depth image (same stamp) and the CameraInfo are saved.
    Reconnects every 2 s while rosbridge is down; `ready` is False until connected with a measured offset."""

    MASK, DEPTH, INFO = "/segmentation/mask", "/perception/depth/image", "/segmentation/camera_info"
    MATCH_GAP_S = 0.15  # frames sent closer than SentLog's window (0.1 s + 0.02 s) would be ambiguous
    _SYNC_PERIOD_S = 10.0

    def __init__(self, url: str, cache: OverlayCache | None = None, clock=time.time) -> None:
        self.url, self.cache, self._clock = url, cache, clock
        self._cond = threading.Condition()
        self._sent = SentLog()
        self._seen: dict[int, float] = {}  # frame index -> host time of its latest matched mask
        self._stamp_index: OrderedDict[int, int] = OrderedDict()  # mask stamp (ns) -> frame index, for the depth
        self._offset: float | None = None  # container clock - host clock, s
        self._samples: deque[tuple[float, float]] = deque(maxlen=8)  # (round trip, offset)
        self._calls: dict[str, float] = {}
        self._ws = None
        self.connected = False
        self.error = ""

    @property
    def ready(self) -> bool:
        return self.connected and self._offset is not None

    def sent(self, index: int, jpeg: bytes | None = None) -> float:
        """Note that frame `index` went out now (and save its JPEG); returns that time for `wait_frame`."""
        t = self._clock()
        with self._cond:
            self._sent.add(t, index)
        if self.cache is not None and jpeg is not None:
            self.cache.put_frame(index, jpeg)
        return t

    def wait_frame(self, index: int, since: float, timeout_s: float) -> bool:
        """True once a mask of frame `index` sent at or after `since` has arrived; False after `timeout_s`."""
        with self._cond:
            return self._cond.wait_for(lambda: self._seen.get(index, -math.inf) >= since - 0.05, timeout=timeout_s)

    def note_offset(self, t_send: float, t_recv: float, container_s: float) -> None:
        """One clock sample: the container's time, read between t_send and t_recv on the host."""
        with self._cond:
            self._samples.append((t_recv - t_send, container_s - (t_send + t_recv) / 2))
            self._offset = min(self._samples)[1]  # the fastest round trip is the most precise

    def on_message(self, msg: dict) -> None:
        if msg.get("op") == "service_response":
            t_send = self._calls.pop(str(msg.get("id")), None)
            values = msg.get("values")
            t = values.get("time") if isinstance(values, dict) else None
            if t_send is not None and isinstance(t, dict) and isinstance(t.get("sec"), int) \
                    and isinstance(t.get("nanosec"), int):
                self.note_offset(t_send, self._clock(), t["sec"] + t["nanosec"] / 1e9)
            elif not self.error.startswith("clock"):
                self.error = f"clock sync failed: {values!r}"[:200]  # a failed call answers with a message string
                print(f"webcam_stream: perception link: {self.error}")
            return
        if msg.get("op") != "publish":
            return
        topic, body = msg.get("topic"), msg.get("msg") or {}
        if topic == self.MASK:
            self._on_mask(body)
        elif topic == self.DEPTH:
            self._on_depth(body)
        elif topic == self.INFO and self.cache is not None:
            self.cache.put_camera({k: body.get(k) for k in ("width", "height", "k", "d", "distortion_model")}
                                  | {"frame_id": body.get("header", {}).get("frame_id", "")})

    def _on_mask(self, msg: dict) -> None:
        ns = stamp_ns(msg)
        with self._cond:
            if ns is None or self._offset is None:
                return
            t_host = ns / 1e9 - self._offset
            index = self._sent.match(t_host)
            if index is None:
                return
            self._seen[index] = max(self._seen.get(index, -math.inf), t_host)
            self._stamp_index[ns] = index
            while len(self._stamp_index) > 64:
                self._stamp_index.popitem(last=False)
            self._cond.notify_all()
        if self.cache is not None:
            img = image_payload(msg, "mono8", 1)
            if img is not None:
                self.cache.put_mask(index, *img)

    def _on_depth(self, msg: dict) -> None:
        ns = stamp_ns(msg)
        with self._cond:
            index = self._stamp_index.get(ns) if ns is not None else None
        if index is not None and self.cache is not None:
            img = image_payload(msg, "32FC1", 4)
            if img is not None:
                self.cache.put_depth(index, *img)

    def start(self) -> None:
        threading.Thread(target=self._run, name="perception-link", daemon=True).start()
        threading.Thread(target=self._clock_sync, name="perception-clock", daemon=True).start()

    def _clock_sync(self) -> None:
        n = 0
        while True:
            ws = self._ws
            if ws is not None and self.connected:
                for _ in range(1 if self._offset is not None else 5):
                    n += 1
                    self._calls[f"t{n}"] = self._clock()
                    try:
                        ws.send(json.dumps({"op": "call_service", "service": "/rosapi/get_time", "id": f"t{n}"}))
                    except Exception:
                        break
                    time.sleep(0.05)
            time.sleep(self._SYNC_PERIOD_S if self._offset is not None else 1.0)

    def _run(self) -> None:
        from websockets.sync.client import connect  # video mode only: the webcam path does not need it

        subs = [(self.MASK, "sensor_msgs/msg/Image"), (self.DEPTH, "sensor_msgs/msg/Image"),
                (self.INFO, "sensor_msgs/msg/CameraInfo")]
        while True:
            try:
                with connect(self.url, open_timeout=3, max_size=None) as ws:
                    for topic, typ in subs if self.cache is not None else subs[:1]:
                        ws.send(json.dumps({"op": "subscribe", "topic": topic, "type": typ}))
                    self._ws, self.connected, self.error = ws, True, ""
                    print(f"webcam_stream: perception link on {self.url}")
                    for raw in ws:
                        self.on_message(json.loads(raw))
            except Exception as exc:  # rosbridge not up yet, or restarted: keep trying
                if self.connected or not self.error:
                    print(f"webcam_stream: perception link: {type(exc).__name__}: {exc}; retrying")
                self.error = f"{type(exc).__name__}: {exc}"
            self._ws, self.connected = None, False
            time.sleep(2.0)


class Player:
    """Plays a video into `latest`, driven by play / pause / replay / sync commands (the UI, over HTTP).

    Starts in `ready` unless autoplay, so the stack can come up before the clip runs: frame 0 is sent once (a client
    cannot open a stream that has no frame) and nothing more until PLAY.
    Frames are put at the file's rate times `speed`. With `sync` on and the perception gate connected, each frame
    waits until perception has finished it (state `buffering`) before the next is sent, so every frame gets its
    overlay and the timeline stretches instead of frames being skipped; without a gate it plays in real time like a
    live camera. Paused, ready or ended, the frame on screen is sent again every `hold_s` (see `_put`)."""

    def __init__(self, read, rewind, encode, *, fps: float, speed: float, frames: int, latest: Latest,
                 link: PerceptionLink | None = None, sync: bool = True, loop: bool = False, autoplay: bool = False,
                 gate_timeout_s: float = 5.0, hold_s: float = 2.0, clock=time.monotonic) -> None:
        self._read, self._rewind, self._encode = read, rewind, encode
        self._fps, self._speed, self._frames = fps, speed, frames
        self._latest, self._link, self._gate_timeout_s, self._clock = latest, link, gate_timeout_s, clock
        self._cond = threading.Condition()
        self._state = "playing" if autoplay else "ready"
        self._sync, self._loop = sync, loop
        self._index = 0  # next frame to send
        self._replay = False
        self._rebase = True
        self._buffering = False
        self._note = ""
        self._hold_s = hold_s
        self._shown: bytes | None = None  # the frame on screen
        self._shown_index = 0
        self._sent_at = -math.inf  # when the last new frame went out (player clock)
        self._shown_at = 0.0

    def play(self) -> None:
        with self._cond:
            if self._state == "ended":
                self._replay = True
            self._state, self._rebase = "playing", True
            self._cond.notify_all()

    def pause(self) -> None:
        with self._cond:
            if self._state in ("playing", "ready"):
                self._state = "paused"
            self._cond.notify_all()

    def replay(self) -> None:
        with self._cond:
            self._replay, self._state = True, "playing"
            self._cond.notify_all()

    def set_sync(self, on: bool) -> None:
        with self._cond:
            self._sync, self._rebase = on, True
            self._cond.notify_all()

    def status(self) -> dict:
        link = self._link
        with self._cond:
            state = "buffering" if self._state == "playing" and self._buffering else self._state
            out = {
                "state": state, "frame": self._index, "frames": self._frames, "fps": self._fps,
                "speed": self._speed, "sync": self._sync, "loop": self._loop,
                "perception": bool(link and link.ready), "note": self._note,
            }
        if link is not None and link.cache is not None:
            out["cache"] = link.cache.summary(self._frames)
        return out

    def _gated(self) -> bool:
        return self._sync and self._link is not None and self._link.ready

    def _put(self, jpeg: bytes, index: int) -> float:
        """Send a frame (call with the lock held). While not playing, the frame on screen is sent again every
        `hold_s`: OpenCV's network read gives up after about 30 s without a frame and the capture never recovers,
        so a silent pause would cut the camera driver off for good. To the stack a held frame is a stopped car
        with a frozen view; the arbiter still holds (the camera age exceeds its limit between held frames)."""
        self._latest.put(jpeg)
        self._shown, self._shown_index, self._shown_at = jpeg, index, self._clock()
        return self._link.sent(index, jpeg) if self._link is not None else 0.0

    def _poster(self) -> None:
        """Send frame 0 once while `ready`: a client (the camera driver) opening the stream needs one frame to open
        it, or OpenCV gives up and the driver exits. Sent once, never repeated; PLAY goes on from frame 1."""
        ok, frame = self._read()
        jpeg = self._encode(frame) if ok and frame is not None else None
        if jpeg is not None:
            with self._cond:
                self._put(jpeg, 0)
                self._index = 1

    def run(self, stop: threading.Event) -> None:
        pending, t0 = None, 0.0
        if self._state == "ready":
            self._poster()
        while not stop.is_set():
            with self._cond:
                self._cond.wait_for(lambda: stop.is_set() or self._replay or self._state == "playing",
                                    timeout=min(0.5, self._hold_s))
                held = self._state != "playing" and not self._replay and self._shown is not None
                if held and self._clock() - self._shown_at >= self._hold_s:
                    self._put(self._shown, self._shown_index)  # held: keep the client's link alive (see _put)
                if stop.is_set():
                    return
                if self._replay:
                    self._rewind()
                    self._index, pending, self._replay, self._rebase, self._note = 0, None, False, True, ""
                if self._state != "playing":
                    continue
                if self._rebase:
                    t0, self._rebase = self._clock() - frame_due_s(self._index, self._fps, self._speed), False
                index = self._index
            if pending is None:
                ok, frame = self._read()
                if not ok or frame is None:
                    with self._cond:
                        if self._loop:
                            self._replay = True
                            print("webcam_stream: end of video, looping (visual odometry sees a jump)")
                        else:
                            self._state = "ended"
                            print("webcam_stream: end of video; holding the last frame (PLAY or REPLAY starts over)")
                    continue
                pending = self._encode(frame)
                if pending is None:
                    with self._cond:
                        self._index += 1  # a frame that cannot be encoded is skipped
                    continue
            with self._cond:
                due = t0 + frame_due_s(index, self._fps, self._speed)
                if self._gated():  # frames closer than the link's match window cannot be told apart: no overlay
                    due = max(due, self._sent_at + self._link.MATCH_GAP_S)
                self._cond.wait_for(lambda: stop.is_set() or self._replay or self._state != "playing",
                                    timeout=max(0.0, due - self._clock()))
                if stop.is_set() or self._replay or self._state != "playing":
                    self._rebase = True  # paused mid-wait: keep the frame, restart the clock on play
                    continue
                gated = self._gated()
                since = self._put(pending, index)
                self._sent_at = self._clock()
                pending, self._index = None, index + 1
                self._buffering = gated
            if gated:
                done = self._link.wait_frame(index, since, self._gate_timeout_s)
                with self._cond:
                    self._buffering = False
                    self._note = "" if done else f"no perception output for {self._gate_timeout_s:g} s; went on"
                    if self._clock() > t0 + frame_due_s(self._index, self._fps, self._speed):
                        self._rebase = True  # perception was slower than the video: stretch, never catch up


_SOF_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}  # frame headers (not DHT, JPG, DAC)


def jpeg_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from a JPEG's frame header, without decoding it; None if it is not a readable JPEG."""
    if data[:2] != b"\xff\xd8":
        return None
    i, n = 2, len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:  # standalone markers, no length
            i += 2
            continue
        if marker in (0xD9, 0xDA):  # end of image or scan data before any frame header
            return None
        seg = struct.unpack(">H", data[i + 2:i + 4])[0]
        if seg < 2:
            return None
        if marker in _SOF_MARKERS:
            if i + 9 > n:
                return None
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return (w, h) if w > 0 and h > 0 else None
        i += 2 + seg
    return None


class PhoneIngest:
    """Frames from a phone's browser (the UI's phone.html, over a WebSocket) into `latest` for /cam.mjpg.

    One sender at a time: a second phone is refused while the first one sends, so two cameras never interleave on one
    calibration; it takes over from a sender silent for `stall_s` (a half-open connection after the phone changed
    network would otherwise lock every phone out), and the old connection is then closed (`is_sender`). Each
    message is one JPEG; one that is not exactly `width` x `height` is dropped and counted, never scaled or cropped
    (K is only valid for the calibrated size). When the sender leaves, or sends nothing for `stall_s`,
    `latest.cut()` ends every stream response, so the camera driver's read fails at once and it reopens (instead
    of blocking ~30 s in OpenCV); until a fresh frame arrives `ready()` is False and /cam.mjpg answers 503. No frame
    is ever repeated: silence is a dead camera, and the safety arbiter holds."""

    def __init__(self, latest: Latest, width: int, height: int, stall_s: float = 2.0, clock=time.monotonic) -> None:
        self._latest, self._size, self.stall_s, self._clock = latest, (width, height), stall_s, clock
        self._lock = threading.Lock()
        self._sender: object | None = None
        self._last_at: float | None = None  # last accepted frame
        self._heard_at = 0.0  # last message of any kind, or the claim
        self._cut = True
        self.accepted = self.rejected = self.refused_senders = self.taken_over = 0
        self.last_rejected = ""

    def claim(self, sender: object) -> bool:
        now = self._clock()
        with self._lock:
            if self._sender is not None and now - self._heard_at <= self.stall_s:
                self.refused_senders += 1
                return False
            if self._sender is not None:
                self.taken_over += 1
            self._sender, self._last_at, self._heard_at = sender, None, now
            cut, self._cut = not self._cut, True
        if cut:
            self._latest.cut()
        return True

    def is_sender(self, sender: object) -> bool:
        with self._lock:
            return self._sender is sender

    def release(self, sender: object) -> None:
        with self._lock:
            if self._sender is not sender:
                return
            self._sender, self._last_at, self._cut = None, None, True
        self._latest.cut()

    def stalled(self, sender: object) -> None:
        """The sender's connection is open but nothing arrived for `stall_s` (the page was hidden, the network
        stalled): end the stream responses once, so the driver reopens and gets 503 until frames return."""
        with self._lock:
            if self._sender is not sender or self._cut:
                return
            self._cut = True
        self._latest.cut()

    def frame(self, sender: object, data: bytes) -> dict:
        """Take one message; the reply the phone gets for it (it measures the round trip with these)."""
        size = jpeg_size(data)
        with self._lock:
            if self._sender is not sender:
                return {"ok": False, "error": "not the active sender"}
            self._heard_at = self._clock()
            if size != self._size:
                self.rejected += 1
                self.last_rejected = "not a JPEG" if size is None else f"{size[0]}x{size[1]}"
                want = f"{self._size[0]}x{self._size[1]}"
                return {"ok": False, "error": f"frame is {self.last_rejected}, need {want}"}
            self.accepted += 1
            self._last_at, self._cut = self._clock(), False
            self._latest.put(data)
        return {"ok": True, "w": size[0], "h": size[1]}

    def ready(self) -> bool:
        with self._lock:
            return (self._sender is not None and self._last_at is not None and not self._cut
                    and self._clock() - self._last_at <= self.stall_s)

    def status(self) -> dict:
        ready = self.ready()
        with self._lock:
            age = None if self._last_at is None else round(self._clock() - self._last_at, 3)
            return {"connected": self._sender is not None, "ready": ready, "accepted": self.accepted,
                    "rejected": self.rejected, "last_rejected": self.last_rejected, "last_frame_age_s": age,
                    "refused_senders": self.refused_senders, "taken_over": self.taken_over,
                    "size": list(self._size)}


def run_ingest(ingest: PhoneIngest, bind: str, port: int) -> None:
    """The phone's WebSocket (binary messages = JPEG frames, each answered with PhoneIngest.frame's reply plus the
    message number). Loopback by default: the UI's dev server proxies /phone/ingest here behind its token gate."""
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.server import serve as ws_serve

    def handle(conn) -> None:
        if not ingest.claim(conn):
            conn.close(1013, "another phone is already streaming")
            return
        print("webcam_stream: phone connected")
        n = 0
        try:
            while True:
                if not ingest.is_sender(conn):  # a newer connection took over from this silent one
                    conn.close(1000, "replaced by a newer phone connection")
                    return
                try:
                    msg = conn.recv(timeout=ingest.stall_s)
                except TimeoutError:
                    ingest.stalled(conn)
                    continue
                if isinstance(msg, str):
                    continue  # text is not a frame
                n += 1
                conn.send(json.dumps({"seq": n} | ingest.frame(conn, msg)))
        except ConnectionClosed:
            pass
        finally:
            ingest.release(conn)
            print(f"webcam_stream: phone disconnected ({ingest.accepted} frames accepted, {ingest.rejected} rejected)")

    with ws_serve(handle, bind, port, max_size=1 << 21, compression=None) as server:
        server.serve_forever()


def open_video(path: str, width: int, fallback_fps: float
               ) -> tuple[cv2.VideoCapture, tuple[int, int], tuple[int, int], float]:
    """The opened file, its decoded size, the size it is served at and its frame rate."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path!r}")
    ok, frame = cap.read()  # the decoded size, not only the container's claim
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"video {path!r} has no readable frame")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    src_w, src_h = frame.shape[1], frame.shape[0]
    fps = source_fps(cap.get(cv2.CAP_PROP_FPS), fallback_fps)
    return cap, (src_w, src_h), output_size(src_w, src_h, width), fps


_COMMANDS = {"play": Player.play, "pause": Player.pause, "replay": Player.replay}


def nearest_sample(img: tuple[int, int, bytes], query: str, itemsize: int) -> tuple[int, int, bytes]:
    """`?w=..&h=..`: the image sampled to w x h by nearest pixel, as the UI samples it onto its analysis grid
    (row floor(v*H/h), column floor(u*W/w)), so a 640x480 depth goes out as 77 KB instead of 1.2 MB with the same
    result on screen. Without a smaller size the image is returned as it is."""
    params = dict(kv.partition("=")[::2] for kv in query.split("&") if kv)
    w, h, data = img
    try:
        tw, th = int(params.get("w", 0)), int(params.get("h", 0))
    except ValueError:
        return img
    if not (0 < tw < w and 0 < th < h):
        return img
    arr = np.frombuffer(data, np.uint8 if itemsize == 1 else "<f4").reshape(h, w)
    rows = (np.arange(th) * h) // th
    cols = (np.arange(tw) * w) // tw
    return tw, th, np.ascontiguousarray(arr[rows][:, cols]).tobytes()


def make_handler(latest: Latest, player: Player | None = None, cache: OverlayCache | None = None,
                 phone: PhoneIngest | None = None):
    """GET /cam.mjpg (the stream). With a phone ingest: GET /phone/status, and /cam.mjpg answers 503 while no phone
    frame is fresh and ends when the phone leaves or stalls (`Latest.cut`). With a video player also: GET /video/status, POST /video/{play,pause,replay} and
    POST /video/sync?on=0|1. With an overlay cache, the recorded frames for the UI's recorded playback:
    GET /video/cache/bundle/<i>?w=&h= (all of one frame in one response, OverlayCache.bundle),
    /video/cache/frame/<i> (JPEG), /video/cache/mask/<i> (raw uint8 classes) and /video/cache/depth/<i> (raw
    little-endian float32 metres, NaN = none), both sized by X-Width / X-Height; POST /video/cache/clear. All with CORS
    so the web UI (another port) can call them."""

    class Handler(BaseHTTPRequestHandler):
        # TCP_NODELAY: headers and body go out as separate writes, and Nagle's algorithm holding the body back for
        # the client's delayed ACK cost the browser about 200 ms per request (curl did not show it).
        disable_nagle_algorithm = True

        def _json(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self) -> None:  # noqa: N802 (CORS preflight)
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def _bytes(self, data: bytes, ctype: str, size: tuple[int, int] | None = None) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=60")
            self.send_header("Access-Control-Allow-Origin", "*")
            if size is not None:
                self.send_header("X-Width", str(size[0]))
                self.send_header("X-Height", str(size[1]))
                self.send_header("Access-Control-Expose-Headers", "X-Width, X-Height")
            self.end_headers()
            self.wfile.write(data)

        def _cache_get(self, path: str, query: str) -> None:
            parts = path.split("/")  # ['', 'video', 'cache', kind, index]
            if cache is None or len(parts) != 5 or not parts[4].isdigit():
                self._json(404, {"error": "no such recorded item"})
                return
            kind, index = parts[3], int(parts[4])
            if kind == "bundle":
                data = cache.bundle(index, query)
                if data is not None:
                    self._bytes(data, "application/octet-stream")
                    return
            elif kind == "frame":
                jpeg = cache.read_frame(index)
                if jpeg is not None:
                    self._bytes(jpeg, "image/jpeg")
                    return
            elif kind in ("mask", "depth"):
                got = cache.read_mask(index) if kind == "mask" else cache.read_depth(index)
                if got is not None:
                    got = nearest_sample(got, query, 1 if kind == "mask" else 4)
                    self._bytes(got[2], "application/octet-stream", (got[0], got[1]))
                    return
            self._json(404, {"error": f"frame {index} has no recorded {kind}"})

        def do_POST(self) -> None:  # noqa: N802
            path, _, query = self.path.partition("?")
            if player is None or not path.startswith("/video/"):
                self._json(404, {"error": "no video player"})
                return
            cmd = path.removeprefix("/video/")
            if cmd == "cache/clear" and cache is not None:
                cache.clear()
            elif cmd == "sync":
                player.set_sync("on=0" not in query.split("&"))
            elif cmd in _COMMANDS:
                _COMMANDS[cmd](player)
            else:
                self._json(404, {"error": f"unknown command {cmd!r}"})
                return
            self._json(200, player.status())

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            path = self.path.split("?")[0]
            if path == "/phone/status" and phone is not None:
                self._json(200, phone.status())
                return
            if path == "/video/status":
                if player is None:
                    self._json(404, {"error": "no video player"})
                else:
                    self._json(200, player.status())
                return
            if path.startswith("/video/cache/"):
                self._cache_get(path, self.path.partition("?")[2])
                return
            if path != "/cam.mjpg":
                self.send_error(404)
                return
            gen = latest.generation
            if phone is not None and not phone.ready():
                self.send_error(503, "no phone is streaming")  # fails the driver's open fast; it retries
                return
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = 0
            try:
                while True:
                    seq, jpeg = latest.wait_newer(seq, timeout=2.0, gen=gen)
                    if latest.generation != gen:
                        break  # the phone left or stalled: end the response so the driver reopens
                    if jpeg is None:
                        continue  # camera stalled; keep the connection, send nothing (silence = dead camera)
                    self.wfile.write(b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def log_message(self, fmt: str, *args) -> None:
            if self.path.startswith(("/video/status", "/video/cache/", "/phone/status")):
                return  # the UI polls the status twice a second and fetches recorded frames by the hundred
            print(f"webcam_stream: {self.client_address[0]} {fmt % args}")

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--video", help="serve this video file instead of the webcam (eval only)")
    ap.add_argument("--speed", type=float, default=1.0, help="--video: playback speed (0.5 = slow motion)")
    ap.add_argument("--loop", action="store_true", help="--video: rewind at the end instead of going silent")
    ap.add_argument("--calibration-out", help="--video: write a placeholder calibration for the output size, exit")
    ap.add_argument("--hfov-deg", type=float, default=90.0, help="--calibration-out: assumed horizontal FOV")
    ap.add_argument("--autoplay", action="store_true", help="--video: start playing at once (default: wait for PLAY)")
    ap.add_argument("--rosbridge", default="ws://localhost:9090",
                    help="--video: rosbridge for perception ('' disables it: always real time, nothing recorded)")
    ap.add_argument("--no-sync", action="store_true", help="--video: start in real time, not waiting for perception")
    ap.add_argument("--overlay-dir", help="--video: where overlays are saved (default: <video stem>.overlays/ beside it)")
    ap.add_argument("--no-record", action="store_true", help="--video: do not save overlays")
    ap.add_argument("--phone", action="store_true",
                    help="serve frames a phone browser sends over a WebSocket (the UI's phone.html) instead of the webcam")
    ap.add_argument("--ingest-bind", default="127.0.0.1", help="--phone: WebSocket address (loopback: only via the UI)")
    ap.add_argument("--ingest-port", type=int, default=8091, help="--phone: WebSocket port")
    ap.add_argument("--stall-s", type=float, default=2.0,
                    help="--phone: no frame for this long ends the stream (the driver reopens)")
    a = ap.parse_args()

    if a.video:
        return serve_video(a)
    if a.phone:
        return serve_phone(a)

    cap = cv2.VideoCapture(a.index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"webcam_stream: cannot open camera index {a.index}", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    cap.set(cv2.CAP_PROP_FPS, a.fps)
    got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if got != (a.width, a.height):
        print(f"webcam_stream: camera gives {got[0]}x{got[1]}, asked {a.width}x{a.height}", file=sys.stderr)
        cap.release()
        return 1

    latest, stop = Latest(), threading.Event()
    threading.Thread(target=capture_loop, args=(cap, latest, a.quality, stop), daemon=True).start()
    return serve(a, latest, stop, cap, f"camera {a.index} {a.width}x{a.height}")


class DualStackServer(ThreadingHTTPServer):
    """Listens on IPv6 and IPv4 at once. On Windows `localhost` resolves to ::1 first, and an IPv4-only server costs
    every request from the browser about 200 ms of failed IPv6 connect before the IPv4 retry."""

    address_family = socket.AF_INET6

    def server_bind(self) -> None:
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


def make_server(bind: str, port: int, handler) -> ThreadingHTTPServer:
    if bind in ("0.0.0.0", "::") and socket.has_ipv6:
        try:
            return DualStackServer(("::", port), handler)
        except OSError:
            pass  # no IPv6 on this host: IPv4 only
    return ThreadingHTTPServer((bind, port), handler)


def serve(a: argparse.Namespace, latest: Latest, stop: threading.Event, cap: cv2.VideoCapture | None, what: str,
          player: Player | None = None, cache: OverlayCache | None = None, phone: PhoneIngest | None = None) -> int:
    server = make_server(a.bind, a.port, make_handler(latest, player, cache, phone))
    server.daemon_threads = True
    print(f"webcam_stream: {what} -> http://{a.bind}:{a.port}/cam.mjpg")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
        if cap is not None:
            cap.release()
    return 0


def serve_phone(a: argparse.Namespace) -> int:
    if a.width <= 0 or a.height <= 0 or not a.stall_s > 0.0:
        print("webcam_stream: --width, --height and --stall-s must be > 0", file=sys.stderr)
        return 1
    try:
        import websockets.sync.server  # noqa: F401  (fail here, not silently in the ingest thread)
    except ImportError:
        print("webcam_stream: --phone needs the websockets package (pip install websockets)", file=sys.stderr)
        return 1
    latest = Latest()
    ingest = PhoneIngest(latest, a.width, a.height, stall_s=a.stall_s)
    threading.Thread(target=run_ingest, args=(ingest, a.ingest_bind, a.ingest_port), name="phone-ingest",
                     daemon=True).start()
    print(f"webcam_stream: phone ingest on ws://{a.ingest_bind}:{a.ingest_port} (the UI proxies /phone/ingest here); "
          f"{a.width}x{a.height} JPEG frames only; status at /phone/status")
    return serve(a, latest, threading.Event(), None, f"phone {a.width}x{a.height}", phone=ingest)


def serve_video(a: argparse.Namespace) -> int:
    if not a.speed > 0.0:
        print("webcam_stream: --speed must be > 0", file=sys.stderr)
        return 1
    try:
        cap, src, size, fps = open_video(a.video, a.width, a.fps)
    except (RuntimeError, ValueError) as exc:
        print(f"webcam_stream: {exc}", file=sys.stderr)
        return 1
    if a.calibration_out:
        cap.release()
        try:
            text = placeholder_calibration(size[0], size[1], a.hfov_deg, f"{Path(a.video).name}, {src[0]}x{src[1]}")
        except ValueError as exc:
            print(f"webcam_stream: {exc}", file=sys.stderr)
            return 1
        Path(a.calibration_out).write_text(text, encoding="utf-8", newline="\n")
        print(f"{size[0]}x{size[1]} {a.calibration_out}")
        return 0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    length = f"{frames / fps:.1f} s" if math.isfinite(frames) and frames > 0 else "unknown length"
    print(f"webcam_stream: video {a.video!r} {src[0]}x{src[1]} @ {fps:g} fps, {length}; served at "
          f"{size[0]}x{size[1]} (scaled, not cropped), speed {a.speed:g}" + (", looping" if a.loop else ""))
    params = [int(cv2.IMWRITE_JPEG_QUALITY), a.quality]

    def encode(frame):
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", frame, params)
        return jpg.tobytes() if ok else None

    total = int(frames) if math.isfinite(frames) and frames > 0 else 0
    cache = None
    if not a.no_record:
        video = Path(a.video).resolve()
        root = Path(a.overlay_dir) if a.overlay_dir else video.with_name(video.stem + ".overlays")
        st = video.stat()
        cache = OverlayCache(root, {"file": video.name, "bytes": st.st_size, "mtime": int(st.st_mtime),
                                    "size": list(size), "frames": total})
        got = cache.summary(total)
        print(f"webcam_stream: overlays saved in {root} ({got['masks']} of {total} frames so far)")
    link = None
    if a.rosbridge:
        link = PerceptionLink(a.rosbridge, cache)
        link.start()
    latest, stop = Latest(), threading.Event()
    player = Player(cap.read, lambda: cap.set(cv2.CAP_PROP_POS_FRAMES, 0), encode, fps=fps, speed=a.speed,
                    frames=total, latest=latest, link=link,
                    sync=not a.no_sync, loop=a.loop, autoplay=a.autoplay)
    threading.Thread(target=player.run, args=(stop,), name="video-player", daemon=True).start()
    if not a.autoplay:
        print("webcam_stream: ready at frame 0; press PLAY in the UI (or POST /video/play)")
    return serve(a, latest, stop, cap, f"video {size[0]}x{size[1]}", player, cache)


if __name__ == "__main__":
    sys.exit(main())
