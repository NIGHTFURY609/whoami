"""Camera driver helpers (no ROS, no OpenCV): calibration -> CameraInfo fields, capture-size
check, the placeholder-calibration refusal, device parsing, the rate pacing of the UI stream, and the newest-frame reader for network cameras.

The calibration itself is loaded and validated by Dev 2's ugv_localization.camera (the single place
that refuses zero / fake K), so a driver can never publish a lying CameraInfo.

A network stream (a phone behind a tunnel) stalls and then delivers a burst of old frames. `LatestFrameReader`
drains the capture on its own thread and keeps only the newest frame, so a burst is published as ONE frame
and a stall is silence (never a repeated old frame), which is what lets the safety arbiter see a dead camera.
Such a stream carries no capture timestamps: the stamp is the arrival time minus a measured, configured
`transport_latency_s` (`frame_stamp_ns`).
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


class CaptureError(RuntimeError):
    """The camera cannot produce frames that match its calibration."""


@dataclass(frozen=True, slots=True)
class CameraInfoFields:
    width: int
    height: int
    distortion_model: str
    d: tuple[float, ...]
    k: tuple[float, ...]
    r: tuple[float, ...]
    p: tuple[float, ...]


def camera_info_fields(cal: object) -> CameraInfoFields:
    """Fields of a sensor_msgs/CameraInfo from a validated ugv_localization CameraCalibration."""
    return CameraInfoFields(
        width=cal.width,  # type: ignore[attr-defined]
        height=cal.height,  # type: ignore[attr-defined]
        distortion_model=cal.distortion_model,  # type: ignore[attr-defined]
        d=tuple(cal.d),  # type: ignore[attr-defined]
        k=tuple(cal.k),  # type: ignore[attr-defined]
        r=tuple(cal.r),  # type: ignore[attr-defined]
        p=tuple(cal.p),  # type: ignore[attr-defined]
    )


def load_calibration_or_refuse(loader: Callable[[str], T], path: str) -> T:
    """Run Dev 2's calibration loader; whatever goes wrong (missing, empty or malformed file, bad K)
    becomes one clear refusal to start instead of a raw traceback. Nothing is ever published without it."""
    try:
        return loader(path)
    except Exception as exc:  # any failure to get a real calibration means: do not start
        raise CaptureError(f"refusing to start: calibration file {path!r}: {type(exc).__name__}: {exc}") from exc


def check_placeholder_calibration(path: str, placeholder: bool, allowed: bool) -> None:
    """A calibration flagged `placeholder: true` holds stand-in numbers (another camera's K), not a calibration of this
    camera: every DA3 distance, the depth cloud Nav2 marks and the RTAB-Map geometry would be scaled by the focal-length
    error with full confidence. Refuse it (fail closed, dev.md Dev 1 task 1 "reject fake K") unless the operator opts in
    with `allow_placeholder_calibration:=true` for bring-up (a cart or handheld run, never an autonomous one that counts)."""
    if placeholder and not allowed:
        raise CaptureError(
            f"refusing to start: calibration file {path!r} is a PLACEHOLDER (placeholder: true), not a calibration of "
            "this camera, so depth scale and map geometry would be wrong; calibrate the camera (config/cameras/README.md) "
            "or, for bring-up only, launch with allow_placeholder_calibration:=true"
        )


def check_capture_size(cal_width: int, cal_height: int, got_width: int, got_height: int) -> None:
    """K is only valid at the resolution it was calibrated at: refuse a camera that delivers another."""
    if (got_width, got_height) != (cal_width, cal_height):
        raise CaptureError(
            f"camera delivers {got_width}x{got_height} but the calibration is for {cal_width}x{cal_height}; "
            "recalibrate at this resolution or set the camera to the calibrated one"
        )


def parse_device(value: str) -> int | str:
    """`0` becomes index 0; `/dev/video0` (or a URL / file) stays as given."""
    v = str(value).strip()
    if not v:
        raise CaptureError("device parameter is empty")
    return int(v) if v.isdigit() else v


MAX_TRANSPORT_LATENCY_S = 5.0  # a tunnel slower than this is not a camera; a bigger number is a unit mistake


def check_transport_latency(value: object) -> float:
    """`transport_latency_s` in seconds: finite, from 0 to MAX_TRANSPORT_LATENCY_S, else refuse to start. A NaN or
    negative latency would put every stamp in the future or nowhere, and `350` typed for milliseconds would
    silently stamp every frame at 0 (the clamp in `frame_stamp_ns`)."""
    try:
        latency = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        latency = math.nan
    if not math.isfinite(latency) or not 0.0 <= latency <= MAX_TRANSPORT_LATENCY_S:
        raise CaptureError(
            f"transport_latency_s is in seconds and must be a finite number from 0 to {MAX_TRANSPORT_LATENCY_S:g} "
            f"(did you give milliseconds?), got {value!r}"
        )
    return latency


def frame_stamp_ns(arrival_s: float, latency_s: float) -> int:
    """Header stamp of a frame: when it arrived here minus the (measured) transport latency, in nanoseconds.
    Clamped at 0 as a last resort (a sim clock near zero with a large latency would otherwise give an invalid
    stamp); the driver logs once if that ever happens."""
    return max(0, round((arrival_s - latency_s) * 1e9))


class LatestFrameReader:
    """Drains a capture on its own thread and keeps only the newest frame (no ROS, no OpenCV).

    `read()` is the capture's blocking read, `(ok, frame)`; `now_s()` is the clock the arrival time is taken
    from (the node's ROS clock, so stamps stay in ROS time). On a successful read the frame is stored with its
    arrival time, replacing a frame nobody took (counted in `dropped`). `take()` hands each frame out once and
    returns None until a newer one arrives. A failed read (not ok, no frame, or `read()` raising) keeps the
    stored frame, is counted in `failed_reads` (consecutive; a good read resets it) and backs off `retry_s`
    so a dead source does not spin a core.

    With a `reopen` hook (network URLs: OpenCV never recovers a stream that dropped, e.g. a phone that locked its
    screen) every `reopen_after` consecutive failed reads call it on this thread; True means the capture is open
    again (`reopens` counts those). Only a good read resets `failed_reads`, so a camera that reopens but sends
    nothing still reads as dead. A failed or raising reopen is reported in `last_error` and backs off
    `reopen_backoff_s`.
    """

    def __init__(
        self, read: Callable[[], tuple[bool, object]], now_s: Callable[[], float], *, retry_s: float = 0.05,
        reopen: Callable[[], bool] | None = None, reopen_after: int = 20, reopen_backoff_s: float = 1.0,
    ) -> None:
        if not retry_s >= 0.0 or not reopen_backoff_s >= 0.0:
            raise ValueError("retry_s and reopen_backoff_s must be >= 0")
        if reopen_after < 1:
            raise ValueError("reopen_after must be >= 1")
        self._read = read
        self._now_s = now_s
        self._retry_s = retry_s
        self._reopen = reopen
        self._reopen_after = reopen_after
        self._reopen_backoff_s = reopen_backoff_s
        self._reopens = 0
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: tuple[object, float] | None = None
        self._dropped = 0
        self._failed = 0
        self._last_error = ""

    def start(self) -> None:
        if self._thread is not None:
            return
        # daemon: a read() blocked inside a network library must never keep the process alive
        self._thread = threading.Thread(target=self._run, name="latest-frame-reader", daemon=True)
        self._thread.start()

    def take(self) -> tuple[object, float] | None:
        """(frame, arrival_s) once per stored frame, else None."""
        with self._lock:
            item, self._latest = self._latest, None
        return item

    @property
    def dropped(self) -> int:
        """Frames superseded by a newer one before anybody took them."""
        with self._lock:
            return self._dropped

    @property
    def failed_reads(self) -> int:
        """Consecutive failed reads (0 after any good read)."""
        with self._lock:
            return self._failed

    @property
    def last_error(self) -> str:
        """Why the latest failed read failed ('' if none has)."""
        with self._lock:
            return self._last_error

    @property
    def reopens(self) -> int:
        """Successful reopens of the capture so far."""
        with self._lock:
            return self._reopens

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self, timeout_s: float = 2.0) -> None:
        """Ask the thread to end and wait up to `timeout_s`. A read() stuck inside a network library cannot be
        interrupted: then the thread outlives this call (check `is_alive`). Safe to call twice."""
        self._stopping.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout_s)

    def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                ok, frame = self._read()
                if ok and frame is not None:
                    self._store(frame, self._now_s())
                    continue
                error = "read returned no frame"
            except Exception as exc:  # a raising read must show up as a failure, not as a silently dead thread
                error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._failed += 1
                self._last_error = error
                failed = self._failed
            if self._reopen is not None and failed % self._reopen_after == 0:
                if self._try_reopen():
                    continue
                self._stopping.wait(self._reopen_backoff_s)
                continue
            self._stopping.wait(self._retry_s)

    def _try_reopen(self) -> bool:
        try:
            ok, why = bool(self._reopen()), "reopen failed"
        except Exception as exc:
            ok, why = False, f"reopen failed: {type(exc).__name__}: {exc}"
        with self._lock:
            if ok:
                self._reopens += 1
            else:
                self._last_error = f"{self._last_error}; {why}"
        return ok

    def _store(self, frame: object, arrival_s: float) -> None:
        with self._lock:
            if self._latest is not None:
                self._dropped += 1
            self._latest = (frame, arrival_s)
            self._failed = 0


class RatePacer:
    """True at most `rate_hz` times per second, given calls at a higher, steady frame rate.

    `slack_s` (about half a frame period) stops a 30 fps camera landing just short of every deadline.
    """

    def __init__(self, rate_hz: float, slack_s: float = 0.0) -> None:
        if not rate_hz > 0.0:
            raise ValueError("rate_hz must be > 0")
        self._period = 1.0 / rate_hz
        self._slack = max(0.0, slack_s)
        self._last: float | None = None

    def due(self, now_s: float) -> bool:
        if self._last is None or now_s - self._last >= self._period - self._slack:
            self._last = now_s
            return True
        return False
