"""The camera driver over real ROS 2 topics. Skipped without ROS 2 / OpenCV (runs under colcon test).

The "camera" is a generated video file (OpenCV opens it exactly like a device) or, for the network-stream
cases, a fake capture that stalls and then delivers a burst; the calibration is a test fixture - never a
product calibration.

ROS domain: set UGV_TEST_ROS_DOMAIN_ID to run next to other ROS work (default: 90 + pid % 9).
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from pathlib import Path

import pytest
import yaml

rclpy = pytest.importorskip("rclpy")
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from sensor_msgs.msg import CameraInfo, CompressedImage, Image  # noqa: E402

from ugv_localization.camera import calibration_to_yaml_dict  # noqa: E402
from test_camera_core import fixture_calibration  # noqa: E402


@pytest.fixture
def video(tmp_path) -> Path:
    path = tmp_path / "cam.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (640, 480))
    for i in range(300):
        frame = np.full((480, 640, 3), (i % 255, 80, 200 - i % 200), np.uint8)
        w.write(frame)
    w.release()
    return path


def write_cal(tmp_path, **kw) -> Path:
    path = tmp_path / "cal.yaml"
    path.write_text(yaml.safe_dump(calibration_to_yaml_dict(fixture_calibration(**kw))), encoding="utf-8")
    return path


def start(args: list[str]):
    os.environ["ROS_DOMAIN_ID"] = os.environ.get("UGV_TEST_ROS_DOMAIN_ID", str(90 + os.getpid() % 9))
    rclpy.init(args=["--ros-args", *args])
    from ugv_bringup.nodes.camera_driver import CameraDriver

    return CameraDriver()


_LIVE = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5, reliability=ReliabilityPolicy.RELIABLE)
_URL = "http://phone.test/cam.mjpg"  # a non-V4L2 source: the driver reads it through LatestFrameReader


class _PacedCapture:
    """The real capture of the test video; read() takes one frame period, like a camera delivering at `fps`."""

    def __init__(self, cap, fps: float | None) -> None:
        self._cap, self._period, self._next = cap, (1.0 / fps if fps else None), time.monotonic()

    def read(self):
        if self._period is not None:
            self._next += self._period
            time.sleep(max(0.0, self._next - time.monotonic()))
        return self._cap.read()

    def __getattr__(self, name):  # isOpened / set / get / release
        return getattr(self._cap, name)


@pytest.fixture
def open_video(monkeypatch, video):
    """Make every cv2.VideoCapture(...) open the test video; call it with `paced`, it returns the call log."""
    calls: list[tuple] = []
    real = cv2.VideoCapture

    def install(paced: bool) -> list[tuple]:
        def factory(src, *api):
            calls.append((src, api))
            return _PacedCapture(real(str(video)), 30.0 if paced else None)

        monkeypatch.setattr(cv2, "VideoCapture", factory)
        return calls

    return install


class FakeStream:
    """A network camera stand-in: the test queues frames, read() blocks until one is there (a stalled stream).
    Frames queued while the node is not spinning therefore arrive as a burst, like a tunnel catching up."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frames: list = []
        self._unblocked = False
        self.calls = 0
        self.released = False
        self.opens = 0  # reopens by the driver after failed reads
        self.size = (640, 480)

    def isOpened(self) -> bool:  # noqa: N802 (cv2 API)
        return True

    def open(self, src) -> bool:
        with self._cond:
            self.opens += 1
            self._unblocked = self.released = False  # a live stream again: read() blocks until a frame is pushed
            self._cond.notify_all()
        return True

    def set(self, prop, value) -> bool:
        return True

    def get(self, prop) -> float:
        return {cv2.CAP_PROP_FRAME_WIDTH: float(self.size[0]), cv2.CAP_PROP_FRAME_HEIGHT: float(self.size[1])}.get(prop, 0.0)

    def push(self, shade: int) -> None:
        with self._cond:
            self._frames.append(np.full((480, 640, 3), shade, np.uint8))
            self._cond.notify_all()

    def read(self):
        with self._cond:
            self.calls += 1
            self._cond.notify_all()
            self._cond.wait_for(lambda: self._frames or self._unblocked)
            return (True, self._frames.pop(0)) if self._frames else (False, None)

    def wait_calls(self, n: int, timeout_s: float = 5.0) -> bool:
        """True once read() has been entered n times: the first n-1 frames are in the reader's hands."""
        with self._cond:
            return self._cond.wait_for(lambda: self.calls >= n, timeout_s)

    def unblock(self) -> None:
        with self._cond:
            self._unblocked = True
            self._cond.notify_all()

    def release(self) -> None:
        self.released = True
        self.unblock()


@pytest.fixture
def stream(monkeypatch) -> FakeStream:
    fake = FakeStream()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: fake)
    return fake


class LogRecorder:
    """Stands in for the node's logger: remembers (level, message)."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def _log(self, level: str, msg, **_kw) -> None:
        self.records.append((level, str(msg)))

    def debug(self, msg, **kw) -> None:
        self._log("debug", msg, **kw)

    def info(self, msg, **kw) -> None:
        self._log("info", msg, **kw)

    def warning(self, msg, **kw) -> None:
        self._log("warning", msg, **kw)

    def error(self, msg, **kw) -> None:
        self._log("error", msg, **kw)

    def count(self, level: str, needle: str) -> int:
        return sum(1 for lv, m in self.records if lv == level and needle in m.lower())


@pytest.fixture
def log(monkeypatch) -> LogRecorder:
    rec = LogRecorder()
    monkeypatch.setattr("ugv_bringup.nodes.camera_driver.CameraDriver.get_logger", lambda self: rec)
    return rec


@contextlib.contextmanager
def running(args: list[str], stream: FakeStream | None = None):
    """The driver, a probe node and an executor over both; everything is torn down afterwards."""
    node = start(args)
    probe = rclpy.create_node("camera_probe")
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(probe)
    try:
        yield node, probe, ex
    finally:
        if stream is not None:
            stream.unblock()  # a real read() stuck in the network cannot be unblocked; the fake can
        ex.shutdown()
        node.close()
        node.destroy_node()
        probe.destroy_node()
        rclpy.shutdown()


def spin(ex, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        ex.spin_once(timeout_sec=0.02)


def spin_until(ex, predicate, timeout_s: float = 5.0) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if predicate():
            return True
        ex.spin_once(timeout_sec=0.02)
    return predicate()


def stamp_s(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def clock_s(node) -> float:
    return node.get_clock().now().nanoseconds / 1e9


@pytest.mark.parametrize(
    "device,paced,backend,reads_through_reader",
    [("/dev/video0", False, (cv2.CAP_V4L2,), False), (_URL, True, (), True)],
    ids=["v4l2-direct-read", "network-newest-frame-reader"],
)
def test_publishes_system_and_ui_topics_from_one_capture(
    tmp_path, open_video, device, paced, backend, reads_through_reader
):
    calls = open_video(paced)
    cal = write_cal(tmp_path)
    node = start(["-p", f"calibration_file:={cal}", "-p", f"device:={device}", "-p", "fps:=30.0", "-p", "compressed_rate_hz:=10.0"])
    probe = rclpy.create_node("camera_probe")
    got: dict[str, list] = {"img": [], "info": [], "ui_img": [], "ui_info": []}
    live = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5, reliability=ReliabilityPolicy.RELIABLE)
    latched = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
    probe.create_subscription(Image, "/camera/image_raw", got["img"].append, live)
    probe.create_subscription(CameraInfo, "/camera/camera_info", got["info"].append, latched)
    probe.create_subscription(CompressedImage, "/image_raw/compressed", got["ui_img"].append, live)
    probe.create_subscription(CameraInfo, "/camera_info", got["ui_info"].append, latched)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(probe)
    try:
        assert calls == [(device, backend)]  # V4L2 devices still open with the V4L2 backend, a URL without one
        assert (node._reader is not None) == reads_through_reader  # only network sources get the newest-frame reader
        end = time.monotonic() + 3.0
        while time.monotonic() < end:
            ex.spin_once(timeout_sec=0.02)

        assert got["img"] and got["info"] and got["ui_img"] and got["ui_info"]
        img, info = got["img"][-1], got["info"][-1]
        assert img.encoding == "rgb8" and (img.width, img.height) == (640, 480)
        assert img.header.frame_id == info.header.frame_id == "camera_optical_frame"
        assert info.k[0] == 500.0 and info.k[2] == 320.0 and (info.width, info.height) == (640, 480)
        # one CameraInfo per image, identical stamp (rgbd_sync pairs by exact stamp)
        stamps = {(m.header.stamp.sec, m.header.stamp.nanosec) for m in got["img"]}
        assert (info.header.stamp.sec, info.header.stamp.nanosec) in stamps
        # UI stream: jpeg that decodes, with its own CameraInfo of the same K and a matching stamp
        ui = got["ui_img"][-1]
        assert ui.format == "jpeg"
        decoded = cv2.imdecode(np.frombuffer(bytes(ui.data), np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None and decoded.shape[:2] == (480, 640)
        ui_info = got["ui_info"][-1]
        assert ui_info.k[0] == 500.0
        assert len(got["ui_img"]) < len(got["img"])  # rate limited
        assert probe.count_publishers("/camera/image_raw") == 1  # a single pipeline
    finally:
        ex.shutdown()
        node.close()
        node.destroy_node()
        probe.destroy_node()
        rclpy.shutdown()


def test_calibration_mode_publishes_raw_images_only(tmp_path, video):
    node = start(["-p", "calibration_mode:=true", "-p", f"device:={video}", "-p", "width:=640", "-p", "height:=480"])
    probe = rclpy.create_node("camera_probe")
    imgs: list = []
    probe.create_subscription(Image, "/camera/image_raw", imgs.append,
                              QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5, reliability=ReliabilityPolicy.RELIABLE))
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(probe)
    try:
        end = time.monotonic() + 2.0
        while time.monotonic() < end:
            ex.spin_once(timeout_sec=0.02)
        assert imgs and imgs[-1].encoding == "rgb8"
        # nothing that downstream could mistake for a calibrated camera
        assert probe.count_publishers("/camera/camera_info") == 0
        assert probe.count_publishers("/camera_info") == 0
        assert probe.count_publishers("/image_raw/compressed") == 0
    finally:
        ex.shutdown()
        node.close()
        node.destroy_node()
        probe.destroy_node()
        rclpy.shutdown()


def test_refuses_cleanly_when_the_calibration_file_is_missing(tmp_path, video):
    with pytest.raises(RuntimeError, match="refusing to start"):
        try:
            start(["-p", f"calibration_file:={tmp_path / 'nope.yaml'}", "-p", f"device:={video}"])
        finally:
            rclpy.shutdown()


def test_refuses_to_start_without_a_valid_calibration(tmp_path, video):
    bad = calibration_to_yaml_dict(fixture_calibration())
    bad["camera_matrix"]["data"] = [0.0] * 9
    path = tmp_path / "zero.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to start"):
        try:
            start(["-p", f"calibration_file:={path}", "-p", f"device:={video}"])
        finally:
            rclpy.shutdown()


def test_refuses_a_camera_whose_resolution_differs_from_the_calibration(tmp_path, video):
    cal = write_cal(tmp_path, width=1280, height=720,
                    k=(900.0, 0.0, 640.0, 0.0, 900.0, 360.0, 0.0, 0.0, 1.0),
                    p=(900.0, 0.0, 640.0, 0.0, 0.0, 900.0, 360.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    with pytest.raises(RuntimeError, match="calibration is for 1280x720"):
        try:
            start(["-p", f"calibration_file:={cal}", "-p", f"device:={video}"])
        finally:
            rclpy.shutdown()


# --- network streams: newest frame only, arrival stamp, transport latency, placeholder calibration ---------------


def test_a_burst_after_a_stall_publishes_only_the_newest_frame_and_then_goes_silent(tmp_path, stream):
    cal = write_cal(tmp_path)
    with running(["-p", f"calibration_file:={cal}", "-p", f"device:={_URL}", "-p", "fps:=30.0"], stream) as (node, probe, ex):
        imgs: list = []
        probe.create_subscription(Image, "/camera/image_raw", imgs.append, _LIVE)
        assert spin_until(ex, lambda: node.count_subscribers("/camera/image_raw") == 1)  # matched; nothing sent yet
        for shade in range(10):
            stream.push(shade * 20)  # the node is not spinning: ten frames pile up, as when a tunnel catches up
        assert stream.wait_calls(11)  # the reader took all ten and is waiting for an eleventh
        assert spin_until(ex, lambda: len(imgs) >= 1)
        spin(ex, 0.5)  # many timer ticks with no new frame
        assert [m.data[0] for m in imgs] == [9 * 20]  # the newest frame, once; the old frame is never repeated
        assert node._reader.dropped == 9
    assert not node._reader.is_alive  # close() stopped the thread
    assert stream.released  # ...and then released the capture


def test_the_stamp_of_a_network_frame_is_its_arrival_time_minus_the_transport_latency(tmp_path, stream):
    cal = write_cal(tmp_path)
    args = ["-p", f"calibration_file:={cal}", "-p", f"device:={_URL}", "-p", "transport_latency_s:=0.5"]
    with running(args, stream) as (node, probe, ex):
        imgs: list = []
        infos: list = []
        probe.create_subscription(Image, "/camera/image_raw", imgs.append, _LIVE)
        probe.create_subscription(CameraInfo, "/camera/camera_info", infos.append,
                                  QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE,
                                             durability=DurabilityPolicy.TRANSIENT_LOCAL))
        assert spin_until(ex, lambda: node.count_subscribers("/camera/image_raw") == 1)
        before = clock_s(node)
        stream.push(7)
        assert stream.wait_calls(2)  # the frame has arrived (the reader holds it); the node has not published it
        arrived_by = clock_s(node)
        time.sleep(0.1)  # a gap between arrival and publication (not a synchronisation): the stamp must not be "now"
        assert spin_until(ex, lambda: imgs and infos)
        stamp = stamp_s(imgs[0])
        assert before - 0.5 - 1e-3 <= stamp <= arrived_by - 0.5 + 1e-3
        assert (infos[0].header.stamp.sec, infos[0].header.stamp.nanosec) == (imgs[0].header.stamp.sec, imgs[0].header.stamp.nanosec)


def test_the_transport_latency_also_shifts_the_stamp_of_a_v4l2_frame(tmp_path, open_video):
    open_video(False)
    cal = write_cal(tmp_path)
    args = ["-p", f"calibration_file:={cal}", "-p", "device:=/dev/video0", "-p", "transport_latency_s:=0.5"]
    with running(args) as (node, probe, ex):
        imgs: list = []
        probe.create_subscription(Image, "/camera/image_raw", imgs.append, _LIVE)
        assert spin_until(ex, lambda: len(imgs) >= 1)
        assert clock_s(node) - stamp_s(imgs[0]) >= 0.5 - 1e-3  # stamped at least the latency before it was published


@pytest.mark.parametrize("bad", ["-0.1", "-1.0", "350.0", "5.5"], ids=["negative", "negative-1", "ms-typo", "just-over"])
def test_an_out_of_range_transport_latency_is_refused_before_the_camera_is_opened(tmp_path, open_video, bad):
    calls = open_video(False)
    cal = write_cal(tmp_path)
    with pytest.raises(RuntimeError, match="transport_latency_s"):
        try:
            start(["-p", f"calibration_file:={cal}", "-p", "device:=/dev/video0", "-p", f"transport_latency_s:={bad}"])
        finally:
            rclpy.shutdown()
    assert calls == []  # fail closed, and without touching the camera


def test_a_placeholder_calibration_is_refused_by_default_and_nothing_is_published(tmp_path, open_video, log):
    calls = open_video(False)
    cal = write_cal(tmp_path, placeholder=True)
    with pytest.raises(RuntimeError, match="PLACEHOLDER"):
        try:
            start(["-p", f"calibration_file:={cal}", "-p", "device:=/dev/video0"])
        finally:
            rclpy.shutdown()
    assert calls == []  # fail closed before the camera is opened: no Image, no CameraInfo
    assert log.count("error", "allow_placeholder_calibration:=true") == 1  # why, and how to override, in the log
    assert log.count("warning", "placeholder") == 0


def test_a_placeholder_calibration_with_the_override_publishes_and_is_warned_about_periodically(
    tmp_path, stream, log, monkeypatch
):
    monkeypatch.setattr("ugv_bringup.nodes.camera_driver._REPORT_PERIOD_S", 0.1)  # production: 10 s
    cal = write_cal(tmp_path, placeholder=True)
    args = ["-p", f"calibration_file:={cal}", "-p", f"device:={_URL}", "-p", "allow_placeholder_calibration:=true"]
    with running(args, stream) as (node, probe, ex):
        assert log.count("warning", "placeholder") == 1  # at start-up, before any timer has fired
        imgs: list = []
        infos: list = []
        probe.create_subscription(Image, "/camera/image_raw", imgs.append, _LIVE)
        probe.create_subscription(CameraInfo, "/camera/camera_info", infos.append,
                                  QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE,
                                             durability=DurabilityPolicy.TRANSIENT_LOCAL))
        assert spin_until(ex, lambda: node.count_subscribers("/camera/image_raw") == 1)
        stream.push(9)
        assert spin_until(ex, lambda: imgs and infos)  # the override really publishes
        assert spin_until(ex, lambda: log.count("warning", "placeholder") >= 3)  # and again on every report period
    assert log.count("error", "placeholder") == 0


def test_a_real_calibration_is_never_warned_about(tmp_path, stream, log, monkeypatch):
    monkeypatch.setattr("ugv_bringup.nodes.camera_driver._REPORT_PERIOD_S", 0.1)
    cal = write_cal(tmp_path)
    with running(["-p", f"calibration_file:={cal}", "-p", f"device:={_URL}"], stream) as (node, probe, ex):
        spin(ex, 0.6)  # several report periods
    assert log.count("warning", "placeholder") == 0
    assert log.count("error", "placeholder") == 0  # and never refused: the default needs no override for a real file


def test_dropped_frames_are_reported_only_when_the_count_changed(tmp_path, stream, log, monkeypatch):
    monkeypatch.setattr("ugv_bringup.nodes.camera_driver._REPORT_PERIOD_S", 0.1)
    cal = write_cal(tmp_path)
    with running(["-p", f"calibration_file:={cal}", "-p", f"device:={_URL}"], stream) as (node, probe, ex):
        for shade in range(10):
            stream.push(shade)
        assert stream.wait_calls(11)  # nine frames were superseded
        assert spin_until(ex, lambda: log.count("info", "dropped") >= 1)
        spin(ex, 0.5)  # more report periods, nothing new dropped
        assert log.count("info", "dropped") == 1
        assert any("9" in m for lv, m in log.records if lv == "info" and "dropped" in m.lower())


# --- a stream that stalls without erroring must not be log-silent ----------------------------------------------


def _stall_warnings(log: LogRecorder) -> list[str]:
    return [m for lv, m in log.records if lv == "warning" and "no new frame" in m.lower()]


def test_a_blocked_read_is_reported_as_a_stall_every_report_period(tmp_path, stream, log, monkeypatch):
    monkeypatch.setattr("ugv_bringup.nodes.camera_driver._REPORT_PERIOD_S", 0.1)  # production: 10 s
    monkeypatch.setattr("ugv_bringup.nodes.camera_driver._STALL_WARN_S", 0.3)  # production: 2 s
    cal = write_cal(tmp_path)
    with running(["-p", f"calibration_file:={cal}", "-p", f"device:={_URL}"], stream) as (node, probe, ex):
        # the fake never delivers and never fails: read() just blocks, so the failed-read counter cannot move
        assert spin_until(ex, lambda: len(_stall_warnings(log)) >= 2)  # and the warning repeats while it lasts
        assert node._reader.failed_reads == 0
        assert all("blocked" in m for m in _stall_warnings(log))
    assert log.count("error", "not delivering") == 0


def test_failing_reads_are_reported_as_a_stall_with_the_reason(tmp_path, stream, log, monkeypatch):
    monkeypatch.setattr("ugv_bringup.nodes.camera_driver._REPORT_PERIOD_S", 0.1)
    monkeypatch.setattr("ugv_bringup.nodes.camera_driver._STALL_WARN_S", 0.3)
    stream.unblock()  # read() now returns (False, None) at once, over and over
    cal = write_cal(tmp_path)
    with running(["-p", f"calibration_file:={cal}", "-p", f"device:={_URL}"], stream) as (node, probe, ex):
        assert spin_until(ex, lambda: len(_stall_warnings(log)) >= 1)
        assert all("failing" in m and "read returned no frame" in m for m in _stall_warnings(log))


def test_a_stall_ends_with_an_info_when_frames_resume_and_a_flowing_stream_is_never_reported(
    tmp_path, stream, log, monkeypatch
):
    monkeypatch.setattr("ugv_bringup.nodes.camera_driver._REPORT_PERIOD_S", 0.1)
    monkeypatch.setattr("ugv_bringup.nodes.camera_driver._STALL_WARN_S", 0.5)
    cal = write_cal(tmp_path)
    with running(["-p", f"calibration_file:={cal}", "-p", f"device:={_URL}"], stream) as (node, probe, ex):
        assert spin_until(ex, lambda: len(_stall_warnings(log)) >= 1)
        assert log.count("info", "resumed") == 0  # still stalled
        stream.push(3)
        assert spin_until(ex, lambda: log.count("info", "resumed") >= 1)
        stalls = len(_stall_warnings(log))
        for _ in range(25):  # frames keep coming, far inside the threshold: no further warning, no second INFO
            stream.push(4)
            ex.spin_once(timeout_sec=0.02)
        assert len(_stall_warnings(log)) == stalls
        assert log.count("info", "resumed") == 1


def test_failed_reads_log_not_delivering_once_per_outage_and_again_after_frames_return(
    tmp_path, stream, log, monkeypatch
):
    monkeypatch.setattr("ugv_bringup.nodes.camera_driver._MAX_FAILED_READS_BEFORE_ERROR", 3)  # production: 30
    stream.unblock()  # every read() returns (False, None)
    cal = write_cal(tmp_path)
    with running(["-p", f"calibration_file:={cal}", "-p", f"device:={_URL}"], stream) as (node, probe, ex):
        imgs: list = []
        probe.create_subscription(Image, "/camera/image_raw", imgs.append, _LIVE)
        assert spin_until(ex, lambda: node.count_subscribers("/camera/image_raw") == 1)
        assert spin_until(ex, lambda: log.count("error", "not delivering frames") == 1)
        assert any("read returned no frame" in m for lv, m in log.records if lv == "error")
        spin(ex, 0.5)  # the same outage goes on: still one line, not one per tick
        assert log.count("error", "not delivering frames") == 1
        stream.push(1)  # frames return ...
        assert spin_until(ex, lambda: len(imgs) >= 1)
        # ... and when the stream dies again that is a new outage with its own line
        assert spin_until(ex, lambda: log.count("error", "not delivering frames") == 2)


def test_a_stamp_that_would_be_negative_is_clamped_to_zero_and_logged_once(tmp_path, stream, log):
    cal = write_cal(tmp_path)
    args = ["-p", f"calibration_file:={cal}", "-p", f"device:={_URL}", "-p", "transport_latency_s:=0.5"]
    with running(args, stream) as (node, probe, ex):
        first = node._stamp(0.1)  # a sim clock near zero with a large latency
        assert (first.sec, first.nanosec) == (0, 0)
        node._stamp(0.2)
        assert log.count("warning", "clamped") == 1  # once, not once per frame
        ok = node._stamp(100.0)
        assert (ok.sec, ok.nanosec) == (99, 500_000_000)  # the normal case is untouched
        assert log.count("warning", "clamped") == 1


def test_a_dropped_url_stream_is_reopened_and_frames_flow_again(tmp_path, stream, log, monkeypatch):
    monkeypatch.setattr("ugv_bringup.nodes.camera_driver._REPORT_PERIOD_S", 0.1)
    cal = write_cal(tmp_path)
    with running(["-p", f"calibration_file:={cal}", "-p", f"device:={_URL}"], stream) as (node, probe, ex):
        imgs: list = []
        probe.create_subscription(Image, "/camera/image_raw", imgs.append, _LIVE)
        assert spin_until(ex, lambda: node.count_subscribers("/camera/image_raw") == 1)
        stream.unblock()  # the phone left: every read fails at once (OpenCV never recovers such a capture itself)
        assert spin_until(ex, lambda: stream.opens >= 1)  # reopened after repeated failed reads
        assert not imgs  # nothing was published meanwhile, no old frame repeated
        stream.push(5)  # the phone is back
        assert spin_until(ex, lambda: len(imgs) >= 1)
        assert spin_until(ex, lambda: log.count("info", "reopened") >= 1)


def test_a_stream_that_comes_back_at_another_size_is_refused_on_reopen(tmp_path, stream, log, monkeypatch):
    monkeypatch.setattr("ugv_bringup.nodes.camera_driver._REPORT_PERIOD_S", 0.1)
    monkeypatch.setattr("ugv_bringup.nodes.camera_driver._STALL_WARN_S", 0.3)
    cal = write_cal(tmp_path)
    with running(["-p", f"calibration_file:={cal}", "-p", f"device:={_URL}"], stream) as (node, probe, ex):
        imgs: list = []
        probe.create_subscription(Image, "/camera/image_raw", imgs.append, _LIVE)
        stream.size = (480, 640)  # a phone held upright
        stream.unblock()
        assert spin_until(ex, lambda: stream.opens >= 1)
        assert spin_until(ex, lambda: any("came back at 480x640" in m for m in _stall_warnings(log)))
        assert spin_until(ex, lambda: stream.released)  # the wrong-size capture is closed again, never read from
        assert not imgs
