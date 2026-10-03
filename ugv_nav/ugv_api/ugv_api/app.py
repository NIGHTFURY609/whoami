"""FastAPI app for /api/v1. Errors are RFC 9457 problem details (application/problem+json).

Resources (operator items of architecture.md only):
  GET  /api/v1/health
  GET  /api/v1/safety/status                 §12 watch table + Dev 5 arbiter status
  GET  /api/v1/safety/e-stop                 §3.1 level 1
  PUT  /api/v1/safety/e-stop                 {asserted}
  GET  /api/v1/base/command                  final /cmd_vel (read only)
  GET  /api/v1/localization                  pose validity + last requested mode
  PUT  /api/v1/localization/mode             {mode: mapping|localize} (§10)
  GET  /api/v1/navigation                    Nav2 heartbeat/status + active goal
  POST /api/v1/navigation/goals              {x, y, yaw, frameId: "map"} (§11) -> 202
  GET  /api/v1/navigation/goals/{id}
  DELETE /api/v1/navigation/goals/{id}       cancel -> 202
  GET  /api/v1/map                           MapStatus: epoch, per-layer seq, stats (the map view's demand heartbeat)
  GET  /api/v1/map/pose                      map -> base_link from TF
  GET  /api/v1/map/{trajectory|grid|live}         Binary format v1, 503 problem until first data
  GET  /api/v1/telemetry/stream              text/event-stream: safety, command, localization, navigation, map, pose
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Callable
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ugv_api import __version__
from ugv_api import mapcodec as codec
from ugv_api import state as k
from ugv_api.errors import ServiceUnavailable
from ugv_api.goals import GoalRecord, GoalRegistry, GoalState
from ugv_api.mapstore import MapStore
from ugv_api.schemas import (
    ArbiterOut,
    BaseCommand,
    EStopIn,
    EStopOut,
    GoalIn,
    GoalOut,
    Health,
    Localization,
    MapSeq,
    MapStatus,
    ModeIn,
    ModeOut,
    Navigation,
    Pose,
    SafetyStatus,
    Vector3,
    WatchOut,
)
from ugv_api.state import StateStore
from ugv_api.watches import Timeouts, evaluate, gate_reasons

PREFIX = "/api/v1"
PROBLEM = "application/problem+json"
OCTET = "application/octet-stream"

logger = logging.getLogger(__name__)


class Robot(Protocol):
    """What the HTTP layer needs from the ROS side (implemented by ros_node.GatewayNode)."""

    timeouts: Timeouts
    requested_mode: str | None

    def get_name(self) -> str: ...
    def now_ns(self) -> int: ...
    @property
    def estop_asserted(self) -> bool: ...
    def set_estop(self, asserted: bool) -> None: ...
    def set_mode(self, mode: str, timeout_s: float = 3.0) -> None: ...
    def action_ready(self) -> bool: ...
    def send_goal(self, rec: GoalRecord) -> None: ...
    def cancel_goal(self, goal_id: str) -> bool: ...


class Problem(HTTPException):
    def __init__(self, status: int, title: str, detail: str, **extra) -> None:
        super().__init__(status_code=status, detail=detail)
        self.title = title
        self.extra = extra


def _problem(request: Request, status: int, title: str, detail, **extra) -> JSONResponse:
    body = {"type": "about:blank", "title": title, "status": status, "detail": detail, "instance": request.url.path}
    body.update(extra)
    return JSONResponse(body, status_code=status, media_type=PROBLEM)


def _goal_out(rec: GoalRecord) -> GoalOut:
    return GoalOut(
        id=rec.id, state=rec.state.value, frame_id=rec.frame_id, x=rec.x, y=rec.y, yaw=rec.yaw,
        distance_remaining=rec.distance_remaining, recoveries=rec.recoveries,
        error_code=rec.error_code, error_message=rec.error_message,
    )


def _pose_values(value: Any) -> tuple[float, ...] | None:
    """(x, y, z, qx, qy, qz, qw) as finite floats, or None for anything else (no value yet, malformed, NaN)."""
    try:
        values = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None
    return values if len(values) == 7 and all(math.isfinite(v) for v in values) else None


def _layer_encoders(
    *, live_spacing_m: float, live_budget: int
) -> dict[str, Callable[[Any, int, int, float], bytes]]:
    """layer -> encode(source, epoch, seq, stamp_s) -> bytes, in the form MapStore.blob calls it. The `source`
    each layer takes is what MapStore.put was given (documented per layer below); the ROS side builds exactly
    these from its messages. Calls go through the `codec` module so a test can count them."""

    def live(src: dict, epoch: int, seq: int, stamp_s: float) -> bytes:
        # {"xyz": (N, 3) float32} in the map frame: Dev 1's full-resolution depth cloud, thinned evenly to one
        # point per voxel of live_spacing_m and at most live_budget points
        return codec.encode_cloud(src["xyz"], None, epoch=epoch, seq=seq, stamp_s=stamp_s, budget=live_budget,
                                  spacing_m=live_spacing_m, thin=True)

    def trajectory(src: Any, epoch: int, seq: int, stamp_s: float) -> bytes:
        # (N, 7) float32: x y z qx qy qz qw
        return codec.encode_trajectory(src, epoch=epoch, seq=seq, stamp_s=stamp_s)

    def grid(src: dict, epoch: int, seq: int, stamp_s: float) -> bytes:
        # {"cells": (H, W) int8, "resolution", "origin_xy", "origin_yaw"}
        return codec.encode_grid(src["cells"], epoch=epoch, seq=seq, stamp_s=stamp_s, resolution=src["resolution"],
                                 origin_xy=src["origin_xy"], origin_yaw=src["origin_yaw"])

    return {"trajectory": trajectory, "grid": grid, "live": live}


def create_app(robot: Robot, store: StateStore, goals: GoalRegistry, *, telemetry_hz: float = 5.0,
               cors_origins: list[str] | None = None, maps: MapStore | None = None,
               live_point_budget: int = 150_000, live_spacing_m: float = 0.02) -> FastAPI:
    if not telemetry_hz > 0:
        raise ValueError("telemetry_hz must be > 0")
    if not (math.isfinite(live_spacing_m) and live_spacing_m > 0):
        raise ValueError("live_spacing_m must be finite and > 0")
    if live_point_budget < 0:
        raise ValueError("live_point_budget must be >= 0")
    encoders = _layer_encoders(live_spacing_m=live_spacing_m, live_budget=live_point_budget)
    app = FastAPI(
        title="UGV operator API",
        version=__version__,
        description="Dev 5 operator boundary (architecture.md §3.1, §10, §11, §12). Never commands /cmd_vel.",
        openapi_url=f"{PREFIX}/openapi.json",
        docs_url=f"{PREFIX}/docs",
        redoc_url=None,
    )
    if cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=["GET", "PUT", "POST", "DELETE"],
                           allow_headers=["Content-Type"], expose_headers=["Location"])

    @app.exception_handler(StarletteHTTPException)
    async def http_problem(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if isinstance(exc, Problem):
            return _problem(request, exc.status_code, exc.title, exc.detail, **exc.extra)
        title = {404: "Not Found", 405: "Method Not Allowed"}.get(exc.status_code, "HTTP error")
        return _problem(request, exc.status_code, title, exc.detail)

    @app.exception_handler(RequestValidationError)
    async def validation_problem(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{"loc": list(e.get("loc", ())), "msg": e.get("msg", "")} for e in exc.errors()]
        return _problem(request, 422, "Invalid request", "The request body does not match the schema.", errors=errors)

    # ---- views (shared by GET resources and the SSE stream) -----------------------------------
    def safety_view() -> SafetyStatus:
        now = robot.now_ns()
        watches = evaluate(store, now, robot.timeouts, estop_asserted=robot.estop_asserted)
        arb = store.get(k.SAFETY_STATUS)
        return SafetyStatus(
            ok=all(w.ok for w in watches),
            watches=[WatchOut(name=w.name, ok=w.ok, reason=w.reason, age_s=w.age_s) for w in watches],
            arbiter=ArbiterOut(
                status=str(arb.value) if arb else None, age_s=arb.age_s(now) if arb else None, present=arb is not None
            ),
            e_stop=estop_view(now),
        )

    def estop_view(now: int | None = None) -> EStopOut:
        now = robot.now_ns() if now is None else now
        seen = store.get(k.E_STOP)
        mine = robot.estop_asserted
        return EStopOut(
            asserted=mine or bool(seen and seen.value),
            asserted_by_gateway=mine,
            last_seen=bool(seen.value) if seen else None,
            last_seen_age_s=seen.age_s(now) if seen else None,
        )

    def command_view() -> BaseCommand:
        s = store.get(k.CMD_VEL)
        if s is None:
            return BaseCommand(available=False, linear=None, angular=None, age_s=None)
        (lx, ly, lz), (ax, ay, az) = s.value
        return BaseCommand(available=True, linear=Vector3(x=lx, y=ly, z=lz), angular=Vector3(x=ax, y=ay, z=az),
                           age_s=s.age_s(robot.now_ns()))

    def localization_view() -> Localization:
        pv, st = store.get(k.POSE_VALID), store.get(k.LOCALIZATION_STATUS)
        return Localization(
            pose_valid=bool(pv.value) if pv else None,
            pose_valid_age_s=pv.age_s(robot.now_ns()) if pv else None,
            status=str(st.value) if st else None,
            requested_mode=robot.requested_mode,
        )

    def navigation_view() -> Navigation:
        hb, st = store.get(k.NAV2_HEARTBEAT), store.get(k.NAV2_STATUS)
        active = goals.active()
        return Navigation(
            heartbeat=bool(hb.value) if hb else None,
            heartbeat_age_s=hb.age_s(robot.now_ns()) if hb else None,
            status=str(st.value) if st else None,
            action_server_ready=robot.action_ready(),
            active_goal=_goal_out(active) if active else None,
        )

    def map_view() -> MapStatus:
        if maps is None:  # no store wired: report "nothing received" under the fixed epoch 0
            return MapStatus(epoch=0, seq=MapSeq(**{name: 0 for name in MapStore.LAYERS}), stats={})
        return MapStatus(epoch=maps.epoch, seq=MapSeq(**maps.seqs()), stats=maps.stats())

    def pose_view() -> Pose:
        sample = store.get(k.TF_MAP_BASE)
        values = _pose_values(sample.value) if sample else None
        if sample is None or values is None:  # no transform yet, or the poller stored only a stamp
            return Pose(available=False, x=None, y=None, z=None, qx=None, qy=None, qz=None, qw=None, age_s=None)
        x, y, z, qx, qy, qz, qw = values
        return Pose(available=True, x=x, y=y, z=z, qx=qx, qy=qy, qz=qz, qw=qw,
                    age_s=sample.age_s(robot.now_ns(), use_stamp=True))  # now - stamp, like the tf watch

    # ---- resources ----------------------------------------------------------------------------
    @app.get(f"{PREFIX}/health", response_model=Health, tags=["system"])
    def health() -> Health:
        return Health(version=__version__, ros_node=robot.get_name(), inputs_seen=store.seen())

    @app.get(f"{PREFIX}/safety/status", response_model=SafetyStatus, tags=["safety"])
    def safety_status() -> SafetyStatus:
        return safety_view()

    @app.get(f"{PREFIX}/safety/e-stop", response_model=EStopOut, tags=["safety"])
    def get_estop() -> EStopOut:
        return estop_view()

    @app.put(f"{PREFIX}/safety/e-stop", response_model=EStopOut, tags=["safety"])
    def put_estop(body: EStopIn) -> EStopOut:
        robot.set_estop(body.asserted)
        return estop_view()

    @app.get(f"{PREFIX}/base/command", response_model=BaseCommand, tags=["base"])
    def base_command() -> BaseCommand:
        return command_view()

    @app.get(f"{PREFIX}/localization", response_model=Localization, tags=["localization"])
    def localization() -> Localization:
        return localization_view()

    @app.put(f"{PREFIX}/localization/mode", response_model=ModeOut, tags=["localization"])
    def put_mode(body: ModeIn) -> ModeOut:
        try:
            robot.set_mode(body.mode)
        except ServiceUnavailable as exc:
            raise Problem(503, "Service unavailable", str(exc)) from exc
        return ModeOut(
            mode=body.mode,
            note="RTAB-Map switched at runtime; restart localization with mode:=" + body.mode
            + " for the pose-validity relocalization gate to match (ugv_localization mode_cli).",
        )

    @app.get(f"{PREFIX}/navigation", response_model=Navigation, tags=["navigation"])
    def navigation() -> Navigation:
        return navigation_view()

    @app.post(f"{PREFIX}/navigation/goals", response_model=GoalOut, status_code=202, tags=["navigation"])
    def post_goal(body: GoalIn, response: Response) -> GoalOut:
        reasons = gate_reasons(evaluate(store, robot.now_ns(), robot.timeouts, estop_asserted=robot.estop_asserted))
        if reasons:
            raise Problem(409, "Goal gate closed", "A §12 safety watch has tripped; no goal is sent.", reasons=reasons)
        rec = goals.create(body.x, body.y, body.yaw, body.frame_id, robot.now_ns())
        try:
            robot.send_goal(rec)
        except ServiceUnavailable as exc:
            goals.update(rec.id, state=GoalState.FAILED, error_message=str(exc))
            raise Problem(503, "Service unavailable", str(exc)) from exc
        response.headers["Location"] = f"{PREFIX}/navigation/goals/{rec.id}"
        return _goal_out(goals.get(rec.id) or rec)

    @app.get(f"{PREFIX}/navigation/goals/{{goal_id}}", response_model=GoalOut, tags=["navigation"])
    def get_goal(goal_id: str) -> GoalOut:
        rec = goals.get(goal_id)
        if rec is None:
            raise Problem(404, "Not Found", f"No goal {goal_id}.")
        return _goal_out(rec)

    @app.delete(f"{PREFIX}/navigation/goals/{{goal_id}}", response_model=GoalOut, status_code=202, tags=["navigation"])
    def delete_goal(goal_id: str) -> GoalOut:
        rec = goals.get(goal_id)
        if rec is None:
            raise Problem(404, "Not Found", f"No goal {goal_id}.")
        if rec.terminal:
            raise Problem(409, "Goal finished", f"Goal {goal_id} is already {rec.state.value}.")
        if not robot.cancel_goal(goal_id):
            raise Problem(409, "Not cancelable", f"Goal {goal_id} is {rec.state.value} and has no live Nav2 handle yet.")
        return _goal_out(goals.get(goal_id) or rec)

    @app.get(f"{PREFIX}/map", response_model=MapStatus, tags=["map"])
    def map_status() -> MapStatus:
        if maps is not None:
            # The demand heartbeat: only an explicit GET counts. The SSE `map` event must not touch, or an
            # open sidebar would keep the heavy subscriptions alive with the map view closed.
            maps.touch(robot.now_ns() / 1e9)
        return map_view()

    @app.get(f"{PREFIX}/map/pose", response_model=Pose, tags=["map"])
    def map_pose() -> Pose:
        return pose_view()

    def add_layer_route(layer: str) -> None:
        encode = encoders[layer]

        def get_layer() -> Response:
            if maps is None:
                raise Problem(503, "Map layers unavailable", "This gateway was started without a map store.")
            try:
                body = maps.blob(layer, encode)  # encodes once per seq, on this threadpool thread
            except Exception as exc:  # a source the codec rejects is a gateway bug: say so, keep the traceback
                logger.exception("could not encode the %s layer", layer)
                raise Problem(500, "Layer encoding failed", f"The {layer} layer could not be encoded: {exc}") from exc
            if body is None:
                raise Problem(503, "No data yet", f"The {layer} layer has not received any data yet.")
            return Response(body, media_type=OCTET, headers={"Cache-Control": "no-store"})

        app.get(
            f"{PREFIX}/map/{layer}", name=f"map_{layer}", tags=["map"], response_class=Response,
            responses={
                200: {"content": {OCTET: {"schema": {"type": "string", "format": "binary"}}}},
                503: {"content": {PROBLEM: {}}, "description": "No data for this layer yet"},
            },
        )(get_layer)

    for layer_name in MapStore.LAYERS:
        add_layer_route(layer_name)

    @app.get(f"{PREFIX}/telemetry/stream", tags=["telemetry"],
             responses={200: {"content": {"text/event-stream": {}}}})
    async def telemetry(request: Request) -> StreamingResponse:
        period = 1.0 / telemetry_hz

        async def events():
            yield f"retry: {int(period * 2000)}\n\n"
            while not await request.is_disconnected():
                views = {
                    "safety": safety_view, "command": command_view,
                    "localization": localization_view, "navigation": navigation_view,
                    "map": map_view, "pose": pose_view,
                }
                for name, view in views.items():
                    data = json.dumps(view().model_dump(mode="json", by_alias=True), separators=(",", ":"))
                    yield f"event: {name}\ndata: {data}\n\n"
                await asyncio.sleep(period)

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app
