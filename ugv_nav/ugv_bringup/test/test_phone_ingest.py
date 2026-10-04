"""webcam_stream.py --phone: frames from a phone browser, the 640x480 gate, one sender, and the stream ending
when the phone leaves so the camera driver reopens.

scripts/ is not a package (the script runs on the Windows host), so it is loaded by path. The WebSocket server
itself needs the websockets package (Windows host only); its logic is PhoneIngest, tested here directly.
"""

from __future__ import annotations

import http.client
import importlib.util
import json
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "webcam_stream.py"
_spec = importlib.util.spec_from_file_location("webcam_stream_phone", _SCRIPT)
ws = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ws)


def jpeg(w: int, h: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.zeros((h, w, 3), np.uint8))
    assert ok
    return buf.tobytes()


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def ingest(clock):
    return ws.PhoneIngest(ws.Latest(), 640, 480, stall_s=2.0, clock=clock)


@pytest.mark.parametrize(("w", "h"), [(640, 480), (480, 640), (1280, 720), (1, 1)])
def test_jpeg_size_reads_the_frame_header(w, h) -> None:
    assert ws.jpeg_size(jpeg(w, h)) == (w, h)


@pytest.mark.parametrize("data", [b"", b"\xff\xd8", b"not a jpeg", b"\xff\xd8\xff\xd9", b"\xff\xd8\xff\xe0\x00"])
def test_jpeg_size_is_none_for_anything_unreadable(data) -> None:
    assert ws.jpeg_size(data) is None


def test_only_640x480_frames_reach_the_stream(ingest) -> None:
    phone = object()
    assert ingest.claim(phone)
    assert ingest.frame(phone, jpeg(640, 480)) == {"ok": True, "w": 640, "h": 480}
    portrait = ingest.frame(phone, jpeg(480, 640))
    assert portrait["ok"] is False and "480x640" in portrait["error"]
    assert ingest.frame(phone, jpeg(1280, 720))["ok"] is False  # never scaled: K is only valid at 640x480
    assert ingest.frame(phone, b"garbage")["error"].startswith("frame is not a JPEG")
    s = ingest.status()
    assert (s["accepted"], s["rejected"], s["last_rejected"]) == (1, 3, "not a JPEG")


def test_a_second_phone_is_refused_while_one_streams(ingest) -> None:
    first, second = object(), object()
    assert ingest.claim(first)
    assert not ingest.claim(second)
    assert ingest.frame(second, jpeg(640, 480))["ok"] is False  # its frames never interleave
    assert ingest.status()["refused_senders"] == 1
    ingest.release(second)  # releasing as the wrong sender changes nothing
    assert ingest.status()["connected"]
    ingest.release(first)
    assert ingest.claim(second)  # free again


def test_a_new_phone_takes_over_from_a_silent_sender(ingest, clock) -> None:
    """A half-open connection (the phone changed network, the tunnel lost the socket) must not lock phones out."""
    old, new = object(), object()
    ingest.claim(old)
    ingest.frame(old, jpeg(640, 480))
    clock.t += 1.0
    assert not ingest.claim(new)  # the old one is still live
    ingest.frame(old, jpeg(480, 640))  # any message counts as alive, also a rejected frame
    clock.t += 1.9
    assert not ingest.claim(new)
    clock.t += 0.2
    assert ingest.claim(new)  # silent for more than stall_s: taken over
    assert ingest.is_sender(new) and not ingest.is_sender(old)
    assert ingest.frame(old, jpeg(640, 480))["ok"] is False
    assert ingest.status()["taken_over"] == 1
    ingest.release(old)  # the old handler's cleanup leaves the new phone alone
    assert ingest.is_sender(new)


def test_a_sender_that_never_sends_can_be_taken_over(ingest, clock) -> None:
    ingest.claim("idle")
    clock.t += 2.5
    assert ingest.claim("phone")


def test_ready_needs_a_fresh_frame_from_a_connected_phone(ingest, clock) -> None:
    phone = object()
    assert not ingest.ready()
    ingest.claim(phone)
    assert not ingest.ready()  # connected, nothing sent yet
    ingest.frame(phone, jpeg(640, 480))
    assert ingest.ready()
    clock.t += 2.5
    assert not ingest.ready()  # stale: older than stall_s
    ingest.frame(phone, jpeg(640, 480))
    assert ingest.ready()
    ingest.release(phone)
    assert not ingest.ready()


def test_a_stall_cuts_the_stream_once_and_a_new_frame_resumes_it(ingest) -> None:
    latest, phone = ingest._latest, object()
    ingest.claim(phone)
    ingest.frame(phone, jpeg(640, 480))
    gen = latest.generation
    ingest.stalled(phone)
    assert latest.generation == gen + 1 and not ingest.ready()
    ingest.stalled(phone)
    assert latest.generation == gen + 1  # one cut per stall, not one per timeout
    ingest.frame(phone, jpeg(640, 480))
    assert ingest.ready()


def test_leaving_cuts_the_stream_and_drops_the_last_frame(ingest) -> None:
    latest, phone = ingest._latest, object()
    ingest.claim(phone)
    ingest.frame(phone, jpeg(640, 480))
    gen = latest.generation
    ingest.release(phone)
    assert latest.generation == gen + 1
    assert latest.wait_newer(0, timeout=0.0)[1] is None  # nothing old is served to the next client


def test_cut_wakes_a_waiting_reader() -> None:
    latest = ws.Latest()
    out = []
    t = threading.Thread(target=lambda: out.append(latest.wait_newer(0, timeout=5.0, gen=latest.generation)))
    t.start()
    latest.cut()
    t.join(2.0)
    assert not t.is_alive() and out == [(0, None)]


@pytest.fixture
def bridge(clock):
    latest = ws.Latest()
    ingest = ws.PhoneIngest(latest, 640, 480, stall_s=2.0, clock=clock)
    server = ws.make_server("127.0.0.1", 0, ws.make_handler(latest, phone=ingest))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield ingest, server.server_address[1]
    server.shutdown()
    server.server_close()


def get(port: int, path: str) -> http.client.HTTPResponse:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path)
    return conn.getresponse()


def test_the_stream_is_503_until_a_phone_frame_exists(bridge) -> None:
    ingest, port = bridge
    assert get(port, "/cam.mjpg").status == 503
    ingest.claim("phone")
    assert get(port, "/cam.mjpg").status == 503  # connected but no frame: the driver's open must fail fast


def test_the_stream_serves_phone_frames_and_ends_when_the_phone_leaves(bridge) -> None:
    ingest, port = bridge
    ingest.claim("phone")
    frame = jpeg(640, 480)
    ingest.frame("phone", frame)
    resp = get(port, "/cam.mjpg")
    assert resp.status == 200
    head = resp.read(200)
    assert head.startswith(b"--ugvframe") and b"image/jpeg" in head
    threading.Timer(0.2, ingest.release, args=("phone",)).start()
    rest = resp.read()  # returns only because the response ends: EOF for the driver, which then reopens
    assert frame[-50:] in head + rest


def test_status_reports_the_counts(bridge) -> None:
    ingest, port = bridge
    ingest.claim("phone")
    ingest.frame("phone", jpeg(480, 640))
    resp = get(port, "/phone/status")
    assert resp.status == 200
    body = json.loads(resp.read())
    assert body["connected"] and not body["ready"] and body["rejected"] == 1 and body["size"] == [640, 480]
