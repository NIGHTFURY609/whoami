"""Map routes and the map/pose events over HTTP, with a fake Robot (no ROS).

Route tests use fastapi's TestClient: real requests through the real app, bodies decoded with the codec's
test decoders. The SSE stream never ends, which TestClient cannot serve, so those tests run uvicorn on a
free port in a thread, like test_api_ros.py does against a real ROS graph.
"""

from __future__ import annotations

import inspect
import json
import socket
import threading
import time

import numpy as np
import pytest

httpx = pytest.importorskip("httpx")
uvicorn = pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient  # noqa: E402

import mapread  # noqa: E402
from ugv_api import mapcodec as codec  # noqa: E402
from ugv_api import state as k  # noqa: E402
from ugv_api.app import create_app  # noqa: E402
from ugv_api.goals import GoalRegistry  # noqa: E402
from ugv_api.mapstore import MapStore  # noqa: E402
from ugv_api.schemas import MapStatus, Pose  # noqa: E402
from ugv_api.state import StateStore  # noqa: E402
from ugv_api.watches import Timeouts  # noqa: E402

PROBLEM = "application/problem+json"
OCTET = "application/octet-stream"
LAYERS = ("trajectory", "grid", "live")
NOW_NS = 1_000 * 1_000_000_000
class FakeRobot:
    """Implements app.Robot with a clock the test moves by hand."""

    def __init__(self) -> None:
        self.timeouts = Timeouts()
        self.requested_mode: str | None = None
        self.t_ns = NOW_NS

    def get_name(self) -> str:
        return "ugv_api"

    def now_ns(self) -> int:
        return self.t_ns

    @property
    def estop_asserted(self) -> bool:
        return False

    def set_estop(self, asserted: bool) -> None:
        pass

    def set_mode(self, mode: str, timeout_s: float = 3.0) -> None:
        pass

    def action_ready(self) -> bool:
        return False

    def send_goal(self, rec) -> None:
        pass

    def cancel_goal(self, goal_id: str) -> bool:
        return False


class Rig:
    def __init__(self, maps: MapStore | None, **tunables) -> None:
        self.robot, self.state, self.maps = FakeRobot(), StateStore(), maps
        self.app = create_app(self.robot, self.state, GoalRegistry(), maps=maps, **tunables)
        self.client = TestClient(self.app)

    def get(self, path: str):
        return self.client.get(f"/api/v1{path}")


@pytest.fixture
def rig() -> Rig:
    return Rig(MapStore(epoch=1234))


# ---------------------------------------------------------------------------------- source builders


def grid_source() -> dict:
    cells = np.arange(12, dtype=np.int8).reshape(3, 4) * 8 - 1
    return {"cells": cells, "resolution": 0.25, "origin_xy": (-0.5, 1.0), "origin_yaw": 0.5}


def trajectory_source() -> np.ndarray:
    return np.array([[0, 0, 0, 0, 0, 0, 1], [3, 4, 0, 0, 0, 1, 0], [3, 4, 12, 0.5, 0.5, 0.5, 0.5]], np.float32)


def live_source(n: int = 6) -> dict:
    return {"xyz": np.arange(3 * n, dtype=np.float32).reshape(n, 3)}


SOURCES = {
    "trajectory": trajectory_source, "grid": grid_source, "live": live_source,
}
assert tuple(SOURCES) == LAYERS


# --------------------------------------------------------------------------------- before any data


@pytest.mark.parametrize("layer", LAYERS)
def test_every_layer_route_is_a_503_problem_before_any_data(rig, layer):
    r = rig.get(f"/map/{layer}")
    assert r.status_code == 503
    assert r.headers["content-type"].startswith(PROBLEM)
    body = r.json()
    assert body["status"] == 503 and body["title"] and layer in body["detail"]
    assert body["instance"] == f"/api/v1/map/{layer}"


@pytest.mark.parametrize("layer", LAYERS)
def test_every_layer_route_is_503_when_the_app_has_no_map_store(layer):
    r = Rig(None).get(f"/map/{layer}")
    assert r.status_code == 503 and r.headers["content-type"].startswith(PROBLEM)


def test_status_before_data_has_every_layer_at_zero(rig):
    r = rig.get("/map")
    assert r.status_code == 200
    assert r.json() == {"epoch": 1234, "seq": {name: 0 for name in LAYERS}, "stats": {}}


def test_status_without_a_store_is_all_zero_with_epoch_zero():
    r = Rig(None).get("/map")
    assert r.status_code == 200
    assert r.json() == {"epoch": 0, "seq": {name: 0 for name in LAYERS}, "stats": {}}


def test_an_unknown_layer_is_a_404_problem(rig):
    r = rig.get("/map/terrain")
    assert r.status_code == 404 and r.headers["content-type"].startswith(PROBLEM)


# ----------------------------------------------------------------------- after put: decoded bodies


def get_binary(rig: Rig, layer: str):
    r = rig.get(f"/map/{layer}")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == OCTET
    assert r.headers["cache-control"] == "no-store"
    assert int(r.headers["content-length"]) == len(r.content)
    return r


def test_live_cloud_has_no_colour_and_the_store_epoch_seq_and_stamp(rig):
    rig.maps.put("live", live_source(6), 7.0)
    d = mapread.cloud(get_binary(rig, "live").content)
    assert (d["epoch"], d["seq"], d["stamp_s"]) == (1234, 1, 7.0)
    assert d["count"] == 6 and d["source_count"] == 6 and not d["has_rgb"]
    np.testing.assert_array_equal(d["xyz"], live_source(6)["xyz"])  # 3 m apart: one point per voxel, all kept
    assert d["spacing_m"] == pytest.approx(0.02)  # the default live_spacing_m


def test_live_cloud_keeps_one_point_per_voxel_then_cuts_evenly_to_its_budget():
    rig = Rig(MapStore(), live_spacing_m=0.5)
    pts = np.array([[0, 0, 1], [0.1, 0.1, 1.1], [2, 0, 1], [2.2, 0, 1.2], [4, 0, 1]], np.float32)
    rig.maps.put("live", {"xyz": pts}, 1.0)
    d = mapread.cloud(get_binary(rig, "live").content)
    assert d["source_count"] == 5 and d["count"] == 3  # the second point of each shared voxel is dropped
    np.testing.assert_array_equal(d["xyz"], pts[[0, 2, 4]])  # first of each voxel, in input order
    cut = Rig(MapStore(), live_point_budget=4)  # Dev 1's cloud is full resolution: the live budget cuts it
    cut.maps.put("live", live_source(10), 7.0)
    d = mapread.cloud(get_binary(cut, "live").content)
    assert d["count"] == 4 and d["source_count"] == 10
    np.testing.assert_array_equal(d["xyz"], live_source(10)["xyz"][[0, 3, 6, 9]])  # an even stride, ends kept


def test_trajectory_decodes(rig):
    rig.maps.put("trajectory", trajectory_source(), 3.0)
    d = mapread.trajectory(get_binary(rig, "trajectory").content)
    assert d["count"] == 3 and d["length_m"] == pytest.approx(17.0)
    np.testing.assert_array_equal(d["poses"], trajectory_source())


def test_grid_decodes(rig):
    rig.maps.put("grid", grid_source(), 6.0)
    d = mapread.grid(get_binary(rig, "grid").content)
    np.testing.assert_array_equal(d["cells"], grid_source()["cells"])
    assert (d["resolution_m"], d["origin_x"], d["origin_y"], d["origin_yaw"]) == (0.25, -0.5, 1.0, 0.5)


def test_each_layer_has_its_own_route_and_none_serves_another_layers_data(rig):
    rig.maps.put("grid", grid_source(), 1.0)
    for layer in LAYERS:
        assert rig.get(f"/map/{layer}").status_code == (200 if layer == "grid" else 503)


def test_a_source_the_codec_rejects_is_a_500_problem_and_does_not_poison_the_layer(rig):
    rig.maps.put("trajectory", np.zeros((3, 5), np.float32), 1.0)  # not (N, 7)
    r = rig.get("/map/trajectory")
    assert r.status_code == 500 and r.headers["content-type"].startswith(PROBLEM)
    assert "(N, 7)" in r.json()["detail"]
    rig.maps.put("trajectory", trajectory_source(), 2.0)
    assert mapread.trajectory(rig.get("/map/trajectory").content)["count"] == 3


# ------------------------------------------------------------------------- seq, memoisation, status


def test_status_reports_the_new_seq_after_a_put(rig):
    rig.maps.put("live", live_source(), 1.0)
    rig.maps.put("live", live_source(), 2.0)
    rig.maps.put("grid", grid_source(), 2.0)
    assert rig.get("/map").json()["seq"] == {**{name: 0 for name in LAYERS}, "live": 2, "grid": 1}


def test_the_body_header_agrees_with_the_status(rig):
    for i in range(3):
        rig.maps.put("grid", grid_source(), float(i))
    status = rig.get("/map").json()
    d = mapread.grid(rig.get("/map/grid").content)
    assert (d["epoch"], d["seq"]) == (status["epoch"], status["seq"]["grid"]) == (1234, 3)


def test_repeated_gets_encode_once_and_a_new_put_encodes_again(rig, monkeypatch):
    calls: list[int] = []
    real = codec.encode_trajectory

    def counting(poses, **kw):
        calls.append(kw["seq"])
        return real(poses, **kw)

    monkeypatch.setattr(codec, "encode_trajectory", counting)
    rig.maps.put("trajectory", trajectory_source(), 1.0)
    bodies = [rig.get("/map/trajectory").content for _ in range(3)]
    assert calls == [1] and bodies[0] == bodies[1] == bodies[2]
    rig.maps.put("trajectory", trajectory_source(), 2.0)
    again = [rig.get("/map/trajectory").content for _ in range(2)]
    assert calls == [1, 2] and again[0] == again[1]
    assert mapread.trajectory(again[0])["seq"] == 2


def test_the_status_is_cheap_and_does_not_encode(rig, monkeypatch):
    def forbidden(*_a, **_kw):
        raise AssertionError("GET /map must not encode a layer")

    for name in ("encode_cloud", "encode_trajectory", "encode_grid"):
        monkeypatch.setattr(codec, name, forbidden)
    for layer in LAYERS:
        rig.maps.put(layer, SOURCES[layer](), 1.0)
    assert rig.get("/map").status_code == 200


def test_stats_pass_through_with_their_snake_case_keys_and_scalar_types(rig):
    rig.maps.put_stats("mapping", {"keyframes": 12, "path_length_m": 3.2, "calibration_placeholder": False, "mode": "mapping",
                                   "last_update_age_s": None, "nested": {"a": 1}})
    stats = rig.get("/map").json()["stats"]
    assert stats == {"keyframes": 12, "path_length_m": 3.2, "calibration_placeholder": False, "mode": "mapping",
                     "last_update_age_s": None}
    assert type(stats["keyframes"]) is int and stats["calibration_placeholder"] is False


def test_status_json_has_exactly_the_contract_keys(rig):
    body = rig.get("/map").json()
    assert set(body) == {"epoch", "seq", "stats"}
    assert tuple(body["seq"]) == LAYERS


# --------------------------------------------------------------------------------------- demand beat


def test_get_map_is_the_demand_heartbeat_and_the_layer_routes_are_not(rig):
    now_s = NOW_NS / 1e9
    assert not rig.maps.wanted(now_s, 5.0)
    rig.maps.put("live", live_source(), 1.0)
    rig.get("/map/live")
    rig.get("/map/pose")
    rig.get("/map/grid")  # a 503, still no touch
    assert not rig.maps.wanted(now_s, 5.0)

    rig.get("/map")
    assert rig.maps.wanted(now_s, 5.0)
    rig.robot.t_ns += 10 * 1_000_000_000  # the operator closes the map view: no more GET /map
    assert not rig.maps.wanted(rig.robot.now_ns() / 1e9, 5.0)


# ---------------------------------------------------------------------------------------------- pose


NO_POSE = {"available": False, "x": None, "y": None, "z": None, "qx": None, "qy": None, "qz": None, "qw": None,
           "ageS": None}


def test_pose_is_unavailable_until_a_transform_is_stored(rig):
    r = rig.get("/map/pose")
    assert r.status_code == 200 and r.json() == NO_POSE


def test_pose_with_only_a_stamp_is_unavailable(rig):
    """What the TF poller stores today: a stamp and no value."""
    rig.state.put(k.TF_MAP_BASE, None, NOW_NS, NOW_NS - 50_000_000)
    assert rig.get("/map/pose").json() == NO_POSE


def test_pose_is_the_stored_transform_with_its_age_from_the_stamp(rig):
    rig.state.put(k.TF_MAP_BASE, (1.0, 2.0, 0.5, 0.0, 0.0, 0.7071, 0.7071), NOW_NS - 400_000_000, NOW_NS - 50_000_000)
    body = rig.get("/map/pose").json()
    assert set(body) == set(NO_POSE)
    assert body["available"] is True
    assert (body["x"], body["y"], body["z"]) == (1.0, 2.0, 0.5)
    assert (body["qx"], body["qy"], body["qz"], body["qw"]) == (0.0, 0.0, 0.7071, 0.7071)
    assert body["ageS"] == pytest.approx(0.05)  # now - stamp, like the tf watch, not now - received


def test_pose_age_falls_back_to_receipt_when_there_is_no_stamp(rig):
    rig.state.put(k.TF_MAP_BASE, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), NOW_NS - 250_000_000)
    assert rig.get("/map/pose").json()["ageS"] == pytest.approx(0.25)


@pytest.mark.parametrize("value", [
    (1.0, 2.0, 3.0), "pose", 7, (1.0, 2.0, 0.0, 0.0, 0.0, 0.0, float("nan")),
    (float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (1.0, 2.0, 0.0, 0.0, 0.0, 0.0, "w"),
])
def test_a_malformed_or_non_finite_transform_is_unavailable_not_an_error(rig, value):
    rig.state.put(k.TF_MAP_BASE, value, NOW_NS, NOW_NS)
    r = rig.get("/map/pose")
    assert r.status_code == 200 and r.json() == NO_POSE


def test_pose_works_without_a_map_store():
    rig = Rig(None)
    rig.state.put(k.TF_MAP_BASE, (1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0), NOW_NS, NOW_NS)
    assert rig.get("/map/pose").json()["available"] is True


# ------------------------------------------------------------------------------- schemas and routes


def test_models_forbid_extra_fields_like_every_other_resource():
    ok = {"epoch": 1, "seq": {name: 0 for name in LAYERS}, "stats": {}}
    assert MapStatus.model_validate(ok).epoch == 1
    for bad in ({**ok, "extra": 1}, {**ok, "seq": {**ok["seq"], "cloud": 0}}, {**ok, "epoch": -1},
                {**ok, "epoch": 2**32}, {**ok, "seq": {**ok["seq"], "live": -1}}):
        with pytest.raises(ValueError):
            MapStatus.model_validate(bad)
    with pytest.raises(ValueError):
        Pose.model_validate({**NO_POSE, "extra": 1})


def test_the_seq_model_has_the_layers_in_the_store_order():
    assert tuple(MapStatus.model_fields["seq"].annotation.model_fields) == MapStore.LAYERS == LAYERS


def test_openapi_lists_the_map_routes(rig):
    paths = rig.client.get("/api/v1/openapi.json").json()["paths"]
    for p in ("/api/v1/map", "/api/v1/map/pose", *(f"/api/v1/map/{name}" for name in LAYERS)):
        assert p in paths, p


def test_map_handlers_are_sync_so_encoding_runs_in_the_threadpool(rig):
    routes = [r for r in rig.app.routes if getattr(r, "path", "").startswith("/api/v1/map")]
    assert len(routes) == 2 + len(LAYERS)
    assert not [r.path for r in routes if inspect.iscoroutinefunction(r.endpoint)]


@pytest.mark.parametrize("tunable,value", [
    ("live_spacing_m", 0.0), ("live_spacing_m", float("nan")), ("live_point_budget", -1),
])
def test_bad_tunables_fail_at_construction(tunable, value):
    with pytest.raises(ValueError, match=tunable):
        create_app(FakeRobot(), StateStore(), GoalRegistry(), maps=MapStore(), **{tunable: value})


def test_existing_callers_without_a_store_still_build_and_serve_health():
    rig = Rig(None)
    assert rig.get("/health").json()["status"] == "ok"


# --------------------------------------------------------------------------------------------- SSE


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Served:
    """The app on a real socket in a thread."""

    def __init__(self, rig: Rig) -> None:
        self.rig = rig
        port = _free_port()
        self.server = uvicorn.Server(uvicorn.Config(rig.app, host="127.0.0.1", port=port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        assert self.server.started, "uvicorn did not start"
        self.client = httpx.Client(base_url=f"http://127.0.0.1:{port}/api/v1", timeout=10.0)

    def close(self) -> None:
        self.client.close()
        self.server.should_exit = True
        self.thread.join(timeout=5)

    def first_cycle(self) -> list[tuple[str, dict]]:
        """(event, data) for the first six events of the stream, in arrival order."""
        got: list[tuple[str, dict]] = []
        with self.client.stream("GET", "/telemetry/stream") as r:
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
            event = None
            for line in r.iter_lines():
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: ") and event:
                    got.append((event, json.loads(line[6:])))
                    if len(got) == 6:
                        break
        return got


@pytest.fixture
def served():
    s = Served(Rig(MapStore(epoch=77)))
    yield s
    s.close()


def test_sse_appends_map_and_pose_after_the_four_existing_events_in_that_order(served):
    served.rig.maps.put("grid", grid_source(), 1.0)
    served.rig.maps.put_stats("mapping", {"keyframes": 3})
    served.rig.state.put(k.TF_MAP_BASE, (1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0), NOW_NS, NOW_NS)
    cycle = served.first_cycle()
    assert [name for name, _ in cycle] == ["safety", "command", "localization", "navigation", "map", "pose"]
    data = dict(cycle)
    assert data["map"] == {"epoch": 77, "seq": {**{n: 0 for n in LAYERS}, "grid": 1}, "stats": {"keyframes": 3}}
    assert data["map"] == served.client.get("/map").json()  # the GET twin says the same
    assert data["pose"]["available"] is True and data["pose"]["x"] == 1.0
    assert data["pose"] == served.client.get("/map/pose").json()  # the fake clock is fixed, so the ages agree


def test_sse_map_event_does_not_touch_the_demand_heartbeat(served):
    now_s = NOW_NS / 1e9
    served.first_cycle()  # an open sidebar: telemetry flows, the map view is closed
    assert not served.rig.maps.wanted(now_s, 5.0)
    served.client.get("/map")
    assert served.rig.maps.wanted(now_s, 5.0)


def test_sse_without_a_store_reports_epoch_zero_and_zero_seqs():
    s = Served(Rig(None))
    try:
        data = dict(s.first_cycle())
    finally:
        s.close()
    assert data["map"] == {"epoch": 0, "seq": {name: 0 for name in LAYERS}, "stats": {}}
    assert data["pose"] == NO_POSE
