"""Camera driver: one capture (V4L2 device or network stream), published for both the system and the UI
(no second pipeline).

Publishes (frame_id = optical frame; one CameraInfo per image, same stamp):
  /camera/image_raw           sensor_msgs/Image            rgb8 - Dev 1 perception, Dev 2 RTAB-Map
  /camera/camera_info         sensor_msgs/CameraInfo       reliable + transient local - Dev 1, Dev 2, safety
  /image_raw/compressed       sensor_msgs/CompressedImage  jpeg, rate-limited - the web UI (rosbridge)
  /camera_info                sensor_msgs/CameraInfo       same message, with each compressed frame - the web UI
The stamp is the time the frame arrived from the capture (taken after the blocking read), not the sensor's
exposure time, which neither V4L2 nor a network stream through OpenCV gives us, minus `transport_latency_s`
(default 0): the measured delay between the phone capturing a frame and it arriving here through a tunnel.

Network streams (any `device` that is not an index or a /dev/... path: `http://...`, `rtsp://...`, a file) are
read on their own thread (`LatestFrameReader`) and only the newest frame is published: a stalled stream that
then delivers a burst of old frames yields one frame, never a replay, and while nothing new arrives the driver
publishes nothing, so the safety arbiter sees the camera as silent. V4L2 devices are read directly, one read
per tick, as before. Two consequences: a video file given as `device` is not paced (it is drained at decode
speed and only the newest frames are published; use a bag for replay), and for a network camera `fps` should be
set above the stream's own rate, otherwise frames are superseded in steady state and `dropped` stops being a
clean stall/burst indicator.

A stream that stalls without erroring leaves a blocked read, which counts no failure, so the driver watches the
time since the last frame it took: after more than 2 s it logs a WARN "no new frame for N s" every report
period (10 s) while that lasts, and an INFO when frames resume.
A URL stream (`scheme://...`) whose reads keep failing is opened again every 20 failed reads, with a 1 s
backoff while that fails (OpenCV never recovers a dropped network stream on its own, and a phone drops it every
time its screen locks); a stream that comes back at another size is refused. Files and V4L2 are not reopened.

Fails closed: no valid calibration, or a camera whose resolution differs from it, means no frames at all
(a driver that publishes a fake K would corrupt DA3 depth and RTAB-Map geometry).

Calibration mode (`calibration_mode:=true`): to calibrate a real camera you first need its images, but the
driver will not run without a calibration. In this mode it publishes ONLY raw images on image_topic, with no
CameraInfo and no UI stream, at the requested width x height, so `camera_calibration` can run. Dev 1 and Dev 2
reject frames without a CameraInfo and the safety arbiter sees the camera as silent, so nothing can use them.

A calibration file flagged `placeholder: true` (stand-in numbers, not a calibration of this camera) is refused like a
fake K: the driver logs an ERROR saying why and how to override, publishes nothing and exits. Only with
`allow_placeholder_calibration:=true` (bring-up on a cart or by hand, never an autonomous run that counts) does it
load, and then the driver logs a WARN at start-up and every 10 s. The driver also logs the frames dropped by the
network reader every 10 s while that count changes.

Params: calibration_file (required unless calibration_mode) device frame_id fps image_topic info_topic
        compressed_topic ui_info_topic compressed_rate_hz jpeg_quality calibration_mode width height
        transport_latency_s (seconds, finite, 0 to 5) allow_placeholder_calibration (bool, default false)
        (empty compressed_topic disables the UI stream; width/height only apply in calibration mode)
"""

from __future__ import annotations

import sys

import cv2
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from ugv_localization.camera import load_calibration

from ugv_bringup.camera_core import (
    CaptureError, LatestFrameReader, RatePacer, camera_info_fields, check_capture_size, check_placeholder_calibration,
    check_transport_latency, frame_stamp_ns, load_calibration_or_refuse, parse_device,
)

_MAX_FAILED_READS_BEFORE_ERROR = 30
_REPORT_PERIOD_S = 10.0  # placeholder-calibration warning, dropped-frame and stall reports
_STALL_WARN_S = 2.0  # no frame taken for longer than this (network streams) is reported as a stall


def _live_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST, depth=1,
        reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE,
    )


def _info_qos() -> QoSProfile:
    # Dev 1 subscribes transient-local, which only matches a transient-local publisher.
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST, depth=1,
        reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


class CameraDriver(Node):
    def __init__(self) -> None:
        super().__init__("camera_driver")
        cal_path = self.declare_parameter("calibration_file", "").value
        device = self.declare_parameter("device", "/dev/video0").value
        self._frame_id = self.declare_parameter("frame_id", "camera_optical_frame").value
        fps = float(self.declare_parameter("fps", 30.0).value)
        image_topic = self.declare_parameter("image_topic", "/camera/image_raw").value
        info_topic = self.declare_parameter("info_topic", "/camera/camera_info").value
        ui_image_topic = self.declare_parameter("compressed_topic", "/image_raw/compressed").value
        ui_info_topic = self.declare_parameter("ui_info_topic", "/camera_info").value
        ui_rate = float(self.declare_parameter("compressed_rate_hz", 5.0).value)
        self._jpeg_quality = int(self.declare_parameter("jpeg_quality", 80).value)
        calibrating = bool(self.declare_parameter("calibration_mode", False).value)
        req_w = int(self.declare_parameter("width", 640).value)
        req_h = int(self.declare_parameter("height", 480).value)
        latency = self.declare_parameter("transport_latency_s", 0.0).value
        allow_placeholder = bool(self.declare_parameter("allow_placeholder_calibration", False).value)

        if not fps > 0.0 or not 1 <= self._jpeg_quality <= 100:
            raise RuntimeError("fps must be > 0 and jpeg_quality in 1..100")
        try:
            self._latency_s = check_transport_latency(latency)
        except CaptureError as exc:
            raise RuntimeError(str(exc)) from exc

        self._info_fields = None
        self._placeholder_file = ""
        if calibrating:
            if req_w <= 0 or req_h <= 0:
                raise RuntimeError("width and height must be > 0")
            width, height = req_w, req_h
            ui_image_topic = ""  # raw images only: no UI stream, no CameraInfo
        else:
            if not cal_path:
                raise RuntimeError(
                    "calibration_file parameter is required; to calibrate this camera first, "
                    "run with calibration_mode:=true"
                )
            try:
                cal = load_calibration_or_refuse(load_calibration, cal_path)
            except CaptureError as exc:
                raise RuntimeError(str(exc)) from exc
            try:
                check_placeholder_calibration(cal_path, cal.placeholder, allow_placeholder)
            except CaptureError as exc:
                self.get_logger().error(str(exc))  # in the node's log too, not only on stderr at exit
                raise RuntimeError(str(exc)) from exc
            self._info_fields = camera_info_fields(cal)
            width, height = cal.width, cal.height
            if cal.placeholder:
                self._placeholder_file = cal_path

        try:
            src = parse_device(device)
        except CaptureError as exc:
            raise RuntimeError(str(exc)) from exc
        v4l2 = isinstance(src, int) or str(src).startswith("/dev/")
        self._cap = cv2.VideoCapture(src, cv2.CAP_V4L2) if v4l2 else cv2.VideoCapture(src)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera {device!r}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        try:
            check_capture_size(
                width, height,
                int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        except CaptureError as exc:
            self._cap.release()
            raise RuntimeError(str(exc)) from exc
        self._size = (width, height)

        self._pub_image = self.create_publisher(Image, image_topic, _live_qos())
        self._pub_info = self._pub_ui_image = self._pub_ui_info = self._pacer = None
        if not calibrating:
            self._pub_info = self.create_publisher(CameraInfo, info_topic, _info_qos())
        if ui_image_topic:
            self._pub_ui_image = self.create_publisher(CompressedImage, ui_image_topic, _live_qos())
            self._pub_ui_info = self.create_publisher(CameraInfo, ui_info_topic, _info_qos())
            self._pacer = RatePacer(ui_rate, slack_s=0.5 / fps)
        self._failed = 0
        self._dead_logged = False
        self._reported_dropped = 0
        self._last_frame_s = self._now_s()  # when a frame was last taken (start-up until the first one)
        self._stalled = False
        self._clamp_logged = False
        # Network streams stall and then deliver a burst: read them on a thread that keeps only the newest frame.
        # A dropped URL stream (a phone that locked its screen) is reopened; a file is not (it would replay).
        self._src = src
        reopen = self._reopen if not v4l2 and "://" in str(src) else None
        self._reader = None if v4l2 else LatestFrameReader(self._cap.read, self._now_s, reopen=reopen)
        self._reported_reopens = 0
        self.create_timer(1.0 / fps, self._tick)
        self.create_timer(_REPORT_PERIOD_S, self._report)
        if calibrating:
            self.get_logger().warning(
                f"CALIBRATION MODE: camera {device!r} {width}x{height} -> {image_topic} raw images only, "
                "no CameraInfo (nothing downstream can use this)"
            )
        else:
            self.get_logger().info(
                f"camera {device!r} {width}x{height} @ {fps:g} fps -> {image_topic}, {info_topic}"
                + (f", {ui_image_topic}, {ui_info_topic}" if ui_image_topic else "")
                + (f"; stamp = arrival - {self._latency_s:g} s" if self._latency_s else "")
            )
        self._report()  # the placeholder warning at start-up, not only after the first period
        if self._reader is not None:
            self._reader.start()

    def _reopen(self) -> bool:
        """On the reader thread, after repeated failed reads: open the URL again. A stream that comes back at
        another size is refused like at start-up (never published)."""
        self._cap.release()
        if not self._cap.open(self._src):
            return False
        got = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if got != self._size:
            self._cap.release()
            raise CaptureError(f"stream came back at {got[0]}x{got[1]}, calibration is {self._size[0]}x{self._size[1]}")
        return True

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _stamp(self, arrival_s: float):
        if arrival_s < self._latency_s and not self._clamp_logged:
            self._clamp_logged = True  # frame_stamp_ns clamps at 0; say so once instead of stamping silently
            self.get_logger().warning(
                f"frame stamp would be negative (arrival {arrival_s:.3f} s, transport_latency_s "
                f"{self._latency_s:g} s): clamped to 0; check the unit (seconds) and the clock (sim time?)"
            )
        return Time(nanoseconds=frame_stamp_ns(arrival_s, self._latency_s)).to_msg()

    def _report(self) -> None:
        if self._placeholder_file:
            self.get_logger().warning(
                f"calibration {self._placeholder_file!r} is a PLACEHOLDER (placeholder: true), not a calibration of "
                "this camera, loaded only because allow_placeholder_calibration is true: depth scale and map geometry "
                "are wrong until it is replaced (config/cameras/README.md)"
            )
        if self._reader is not None:
            total = self._reader.dropped
            if total != self._reported_dropped:
                self.get_logger().info(
                    f"network camera: {total - self._reported_dropped} frames dropped (superseded by a newer frame "
                    f"before they were published) in the last {_REPORT_PERIOD_S:g} s, {total} in total"
                )
                self._reported_dropped = total
            reopens = self._reader.reopens
            if reopens != self._reported_reopens:
                self.get_logger().info(f"network camera reopened ({reopens} times since start)")
                self._reported_reopens = reopens
            self._report_stall()

    def _report_stall(self) -> None:
        """A tunnel that stalls without erroring leaves a blocked read(): it counts no failure, so without this the
        driver would be silent on the topics (right) and in the log (not helpful)."""
        silent_s = self._now_s() - self._last_frame_s
        if silent_s <= _STALL_WARN_S:
            return
        self._stalled = True
        if self._reader.failed_reads:
            why = f"reads are failing: {self._reader.last_error}"
        else:
            why = "the read is blocked, nothing is arriving from the stream"
        self.get_logger().warning(f"no new frame for {silent_s:.1f} s ({why}); publishing nothing")

    def _info(self, stamp) -> CameraInfo:
        f = self._info_fields
        m = CameraInfo()
        m.header.stamp = stamp
        m.header.frame_id = self._frame_id
        m.width, m.height = f.width, f.height
        m.distortion_model = f.distortion_model
        m.d, m.k, m.r, m.p = list(f.d), list(f.k), list(f.r), list(f.p)
        return m

    def _tick(self) -> None:
        if self._reader is not None:
            item = self._reader.take()
            if item is None:
                # No new frame since the last tick (stalled or dead stream): publish nothing, never the old frame
                # again. Silence is what lets the safety arbiter see a dead camera.
                if self._reader.failed_reads >= _MAX_FAILED_READS_BEFORE_ERROR and not self._dead_logged:
                    self._dead_logged = True
                    self.get_logger().error(f"camera is not delivering frames ({self._reader.last_error})")
                return
            self._dead_logged = False
            bgr, arrival_s = item
            if self._stalled:
                self._stalled = False
                self.get_logger().info(f"camera frames resumed after {arrival_s - self._last_frame_s:.1f} s without one")
            self._last_frame_s = arrival_s
        else:
            ok, bgr = self._cap.read()
            arrival_s = self._now_s()  # after the blocking read: when the frame actually arrived
            if not ok or bgr is None:
                # Publish nothing: silence is what lets the safety arbiter see a dead camera.
                self._failed += 1
                if self._failed == _MAX_FAILED_READS_BEFORE_ERROR:
                    self.get_logger().error("camera is not delivering frames")
                return
            self._failed = 0
        stamp = self._stamp(arrival_s)
        if (bgr.shape[1], bgr.shape[0]) != self._size:
            self.get_logger().error("frame size changed after start; dropping frames", throttle_duration_sec=5.0)
            return

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image()
        img.header.stamp = stamp
        img.header.frame_id = self._frame_id
        img.height, img.width = rgb.shape[0], rgb.shape[1]
        img.encoding = "rgb8"
        img.is_bigendian = 0
        img.step = rgb.shape[1] * 3
        img.data = rgb.tobytes()
        self._pub_image.publish(img)
        if self._pub_info is not None:
            self._pub_info.publish(self._info(stamp))

        if self._pacer is not None and self._pacer.due(self.get_clock().now().nanoseconds / 1e9):
            ok_jpg, jpg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
            if ok_jpg:
                c = CompressedImage()
                c.header.stamp = stamp
                c.header.frame_id = self._frame_id
                c.format = "jpeg"
                c.data = jpg.tobytes()
                self._pub_ui_image.publish(c)
                self._pub_ui_info.publish(self._info(stamp))

    def close(self) -> None:
        if self._reader is not None:
            self._reader.stop()
            if self._reader.is_alive:
                # Blocked inside the network read: releasing the capture under it is unsafe, and the process is
                # about to exit, which ends the (daemon) thread.
                self.get_logger().warning("camera read is still blocked in the network; leaving the capture open")
                return
        self._cap.release()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    code = 0
    try:
        node = CameraDriver()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as exc:
        print(f"camera_driver: {exc}", file=sys.stderr)
        code = 1
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.try_shutdown()
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
