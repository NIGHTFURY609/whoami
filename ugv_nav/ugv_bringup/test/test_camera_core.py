"""Camera driver helpers (pure Python, no ROS / OpenCV). The calibration here is a test fixture."""

from __future__ import annotations

import math
import threading

import pytest
import yaml

from ugv_bringup.camera_core import (
    CaptureError, LatestFrameReader, RatePacer, camera_info_fields, check_capture_size, check_placeholder_calibration,
    check_transport_latency, frame_stamp_ns, load_calibration_or_refuse, parse_device,
)
from ugv_localization.camera import CalibrationError, calibration_to_yaml_dict, load_calibration
from ugv_localization.camera.calib import CameraCalibration


def fixture_calibration(**kw) -> CameraCalibration:
    base = dict(
        camera_name="test_cam", width=640, height=480,
        k=(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0),
        distortion_model="plumb_bob", d=(0.0,) * 5,
        r=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        p=(500.0, 0.0, 320.0, 0.0, 0.0, 500.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    )
    base.update(kw)
    return CameraCalibration(**base)


def test_camera_info_fields_carry_the_calibration_through(tmp_path):
    path = tmp_path / "cam.yaml"
    path.write_text(yaml.safe_dump(calibration_to_yaml_dict(fixture_calibration())), encoding="utf-8")
    f = camera_info_fields(load_calibration(path))
    assert (f.width, f.height, f.distortion_model) == (640, 480, "plumb_bob")
    assert f.k[0] == 500.0 and f.k[2] == 320.0 and len(f.k) == 9 and len(f.p) == 12 and len(f.d) == 5


def test_a_zero_k_calibration_cannot_be_loaded_so_it_can_never_be_published(tmp_path):
    bad = calibration_to_yaml_dict(fixture_calibration())
    bad["camera_matrix"]["data"] = [0.0] * 9
    path = tmp_path / "zero.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(CalibrationError):
        load_calibration(path)


def test_missing_calibration_file_is_refused(tmp_path):
    with pytest.raises(Exception):
        load_calibration(tmp_path / "nope.yaml")


def test_capture_size_must_match_calibration():
    check_capture_size(640, 480, 640, 480)
    with pytest.raises(CaptureError, match="640x480"):
        check_capture_size(640, 480, 1280, 720)


@pytest.mark.parametrize("value,expected", [("0", 0), ("2", 2), ("/dev/video0", "/dev/video0"), (" /dev/video1 ", "/dev/video1")])
def test_parse_device(value, expected):
    assert parse_device(value) == expected


def test_parse_device_rejects_empty():
    with pytest.raises(CaptureError):
        parse_device("  ")


def test_pacer_limits_a_30fps_stream_to_about_5hz():
    pacer = RatePacer(5.0, slack_s=0.5 / 30.0)
    hits = [i for i in range(300) if pacer.due(i / 30.0)]  # 10 s of 30 fps
    assert 48 <= len(hits) <= 52
    assert hits[0] == 0


def test_pacer_rejects_nonpositive_rate():
    with pytest.raises(ValueError):
        RatePacer(0.0)


def test_a_good_calibration_loads_through_the_refusal_wrapper(tmp_path):
    path = tmp_path / "ok.yaml"
    path.write_text(yaml.safe_dump(calibration_to_yaml_dict(fixture_calibration())), encoding="utf-8")
    assert load_calibration_or_refuse(load_calibration, str(path)).width == 640


@pytest.mark.parametrize(
    "content",
    [None, "", "just a string", "key: [unclosed", "image_width: 640"],
    ids=["missing", "empty", "not-a-mapping", "malformed-yaml", "missing-keys"],
)
def test_every_bad_calibration_file_is_one_clean_refusal(tmp_path, content):
    path = tmp_path / "bad.yaml"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(CaptureError, match="refusing to start"):
        load_calibration_or_refuse(load_calibration, str(path))


def test_a_zero_k_file_is_refused_cleanly(tmp_path):
    bad = calibration_to_yaml_dict(fixture_calibration())
    bad["camera_matrix"]["data"] = [0.0] * 9
    path = tmp_path / "zero.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(CaptureError, match="refusing to start"):
        load_calibration_or_refuse(load_calibration, str(path))


def test_a_placeholder_calibration_is_refused_by_default_with_the_override_named():
    with pytest.raises(CaptureError, match="refusing to start") as exc:
        check_placeholder_calibration("phone.yaml", placeholder=True, allowed=False)
    assert "PLACEHOLDER" in str(exc.value) and "allow_placeholder_calibration:=true" in str(exc.value)


def test_a_placeholder_calibration_loads_only_with_the_explicit_override():
    check_placeholder_calibration("phone.yaml", placeholder=True, allowed=True)  # no raise


@pytest.mark.parametrize("allowed", [False, True], ids=["default", "override"])
def test_a_real_calibration_is_never_refused_as_a_placeholder(allowed):
    check_placeholder_calibration("cam.yaml", placeholder=False, allowed=allowed)  # no raise


# --- newest-frame reader --------------------------------------------------------------------------------------
# The reader runs a thread, so every test waits on the reader itself: QueueSource counts read() entries, and the
# Nth entry proves the first N-1 results were consumed (the thread only calls read() again after storing/counting
# the previous one). No test relies on a sleep for ordering.


class QueueSource:
    """A capture whose results the test pushes; read() blocks until one is queued, like a stalled network stream."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._items: list[tuple[object, ...]] = []
        self._closed = False
        self._clock = 0.0
        self.calls = 0

    def push(self, ok: bool, frame: object, t: float = 0.0) -> None:
        with self._cond:
            self._items.append((ok, frame, t))
            self._cond.notify_all()

    def push_error(self, exc: Exception) -> None:
        with self._cond:
            self._items.append(exc)
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def read(self) -> tuple[bool, object]:
        with self._cond:
            self.calls += 1
            self._cond.notify_all()
            self._cond.wait_for(lambda: self._items or self._closed)
            if not self._items:
                return False, None
            item = self._items.pop(0)
        if isinstance(item, Exception):
            raise item
        ok, frame, t = item
        self._clock = t  # the "arrival time" the reader will see when it stamps this frame
        return ok, frame

    def now(self) -> float:
        return self._clock

    def wait_calls(self, n: int, timeout_s: float = 5.0) -> bool:
        with self._cond:
            return self._cond.wait_for(lambda: self.calls >= n, timeout_s)


@pytest.fixture
def source():
    src = QueueSource()
    yield src
    src.close()


@pytest.fixture
def make_reader(source):
    readers: list[LatestFrameReader] = []

    def make(**kw) -> LatestFrameReader:
        reader = LatestFrameReader(source.read, source.now, **kw)
        readers.append(reader)
        return reader

    yield make
    source.close()  # unblock a read() still waiting, then stop the threads
    for reader in readers:
        reader.stop()


def wait_until(predicate, timeout_s: float = 5.0) -> bool:
    done = threading.Event()
    deadline = threading.Timer(timeout_s, done.set)
    deadline.start()
    try:
        while not predicate():
            if done.wait(0.002):
                return predicate()
        return True
    finally:
        deadline.cancel()


def test_a_burst_leaves_only_the_newest_frame_and_counts_the_rest_as_dropped(source, make_reader):
    reader = make_reader()
    for i in range(10):
        source.push(True, f"frame{i}", t=float(i))
    reader.start()
    assert source.wait_calls(11)  # the reader is blocked in read() again: all ten were consumed
    assert reader.take() == ("frame9", 9.0)  # (frame, arrival time as the reader saw it)
    assert reader.dropped == 9
    assert reader.take() is None


def test_take_returns_each_frame_once_and_none_when_nothing_new_arrived(source, make_reader):
    reader = make_reader()
    assert reader.take() is None  # not started
    reader.start()
    assert source.wait_calls(1)
    assert reader.take() is None  # started, nothing arrived
    source.push(True, "a", t=1.0)
    assert source.wait_calls(2)
    assert reader.take() == ("a", 1.0)
    assert reader.take() is None  # the same frame is never handed out twice
    source.push(True, "b", t=2.0)
    assert source.wait_calls(3)
    assert reader.take() == ("b", 2.0)
    assert reader.dropped == 0  # a frame that was taken before the next one arrived was not dropped


def test_a_failed_read_keeps_the_stored_frame_and_is_counted_until_a_read_succeeds(source, make_reader):
    reader = make_reader(retry_s=0.0)
    source.push(True, "a", t=1.0)
    source.push(False, None)
    source.push(False, None)
    reader.start()
    assert source.wait_calls(4)  # a, fail, fail consumed
    assert reader.failed_reads == 2
    assert reader.take() == ("a", 1.0)  # failures neither replaced nor erased the frame
    source.push(True, "b", t=2.0)
    assert source.wait_calls(5)
    assert reader.failed_reads == 0  # consecutive failures: a good read resets the count
    assert reader.take() == ("b", 2.0)
    assert reader.dropped == 0


def test_a_read_that_raises_is_a_failed_read_not_a_dead_thread(source, make_reader):
    reader = make_reader(retry_s=0.0)
    source.push_error(RuntimeError("boom"))
    source.push_error(RuntimeError("boom again"))
    reader.start()
    assert source.wait_calls(3)
    assert reader.failed_reads == 2
    assert "boom again" in reader.last_error
    source.push(True, "a", t=1.0)  # the thread survived and still delivers
    assert source.wait_calls(4)
    assert reader.take() == ("a", 1.0)
    assert reader.failed_reads == 0


def test_failed_reads_back_off_instead_of_spinning_and_stop_interrupts_the_wait(source, make_reader):
    reader = make_reader(retry_s=3600.0)
    source.push(False, None)
    reader.start()
    assert wait_until(lambda: reader.failed_reads == 1)
    reader.stop()  # returns long before the hour is up: the backoff is interruptible
    assert not reader.is_alive
    assert source.calls == 1  # it waited after the failure; it did not call read() again in a loop


def test_stop_is_safe_twice_and_does_not_hang_on_a_read_that_never_returns(source, make_reader):
    reader = make_reader()
    reader.start()
    assert source.wait_calls(1)  # the thread is inside read()
    reader.stop(timeout_s=0.05)
    assert reader.is_alive  # stop() gave up waiting and says so, instead of hanging or lying
    source.close()
    reader.stop()
    reader.stop()
    assert not reader.is_alive


@pytest.mark.parametrize("value", [0.0, 0.25, 3, 0.001, 5.0])
def test_transport_latency_accepts_finite_seconds_from_zero_to_five(value):
    assert check_transport_latency(value) == float(value)


@pytest.mark.parametrize("value", [-0.001, -1.0, math.nan, math.inf, -math.inf, "fast", None, 5.001, 350, 1e9])
def test_transport_latency_rejects_anything_else(value):
    with pytest.raises(CaptureError, match="transport_latency_s"):
        check_transport_latency(value)


def test_a_transport_latency_typed_in_milliseconds_is_refused_with_the_unit_named():
    # 350 meant milliseconds: accepted, it would clamp every stamp to 0 and nobody would notice
    with pytest.raises(CaptureError, match=r"in seconds.*0 to 5.*350"):
        check_transport_latency(350)


def test_frame_stamp_is_arrival_minus_latency_in_nanoseconds():
    assert frame_stamp_ns(100.0, 0.25) == 99_750_000_000
    assert frame_stamp_ns(100.0, 0.0) == 100_000_000_000


def test_frame_stamp_is_never_negative():
    assert frame_stamp_ns(0.1, 0.5) == 0  # sim clock near zero with a large configured latency


class Reopener:
    """The reopen hook: counts calls and answers from a queue (True = reopened, False or an exception = failed)."""

    def __init__(self, *answers: object) -> None:
        self._answers = list(answers)
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        answer = self._answers.pop(0) if self._answers else True
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_without_a_reopen_hook_failed_reads_never_reopen(source, make_reader):
    reader = make_reader(retry_s=0.0, reopen_after=2)
    for _ in range(5):
        source.push(False, None)
    reader.start()
    assert source.wait_calls(6)
    assert reader.failed_reads == 5
    assert reader.reopens == 0  # V4L2 and files keep the old behaviour: no hook, no reopen


def test_the_stream_is_reopened_after_reopen_after_consecutive_failed_reads(source, make_reader):
    hook = Reopener(True)
    reader = make_reader(retry_s=0.0, reopen=hook, reopen_after=3, reopen_backoff_s=0.0)
    for _ in range(3):
        source.push(False, None)
    reader.start()
    assert wait_until(lambda: reader.reopens == 1)
    source.push(True, "a", t=1.0)
    assert wait_until(lambda: reader.failed_reads == 0)  # a good read after the reopen resets the count
    assert reader.take() == ("a", 1.0)
    assert hook.calls == 1


def test_a_successful_reopen_does_not_hide_a_dead_camera(source, make_reader):
    """The count only resets on a good read, so the driver's 'not delivering frames' error still fires."""
    hook = Reopener(True, True, True)
    reader = make_reader(retry_s=0.0, reopen=hook, reopen_after=2, reopen_backoff_s=0.0)
    for _ in range(4):
        source.push(False, None)
    reader.start()
    assert wait_until(lambda: hook.calls == 2)  # at 2 and at 4 failures
    assert reader.failed_reads == 4


def test_a_failed_or_raising_reopen_backs_off_and_is_reported(source, make_reader):
    hook = Reopener(RuntimeError("503 Service Unavailable"))
    reader = make_reader(retry_s=0.0, reopen=hook, reopen_after=1, reopen_backoff_s=3600.0)
    source.push(False, None)
    reader.start()
    assert wait_until(lambda: hook.calls == 1)
    assert wait_until(lambda: "503" in reader.last_error)
    assert reader.reopens == 0
    reader.stop()  # the backoff is interruptible
    assert not reader.is_alive
    assert source.calls == 1  # it backed off after the failed reopen instead of reading again at once


def test_reopen_after_must_be_positive():
    with pytest.raises(ValueError):
        LatestFrameReader(lambda: (False, None), lambda: 0.0, reopen=lambda: True, reopen_after=0)
