"""Request/response models for /api/v1. JSON uses camelCase; Python uses snake_case."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

Finite = Annotated[float, Field(allow_inf_nan=False)]


class Model(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class Health(Model):
    status: Literal["ok"] = "ok"
    version: str
    ros_node: str
    inputs_seen: list[str] = Field(description="Watched inputs received at least once")


class WatchOut(Model):
    name: Literal["camera", "perception", "localization", "tf", "nav2", "e_stop"]
    ok: bool
    reason: str
    age_s: float | None


class ArbiterOut(Model):
    status: str | None = Field(description="Last /ugv/safety_status text from the Dev 5 arbiter")
    age_s: float | None
    present: bool = Field(description="False until a safety arbiter has published /ugv/safety_status")


class EStopOut(Model):
    asserted: bool = Field(description="Asserted by this gateway or last seen true on /ugv/e_stop")
    asserted_by_gateway: bool
    last_seen: bool | None = Field(description="Last value seen on /ugv/e_stop from any publisher")
    last_seen_age_s: float | None


class EStopIn(Model):
    asserted: bool


class SafetyStatus(Model):
    ok: bool = Field(description="True only when every §12 watch is ok")
    watches: list[WatchOut]
    arbiter: ArbiterOut
    e_stop: EStopOut


class Vector3(Model):
    x: float
    y: float
    z: float


class BaseCommand(Model):
    available: bool = Field(description="False until /cmd_vel has been seen")
    linear: Vector3 | None
    angular: Vector3 | None
    age_s: float | None


Mode = Literal["mapping", "localize"]


class Localization(Model):
    pose_valid: bool | None
    pose_valid_age_s: float | None
    status: str | None = Field(description="Last /ugv/localization_status text")
    requested_mode: Mode | None = Field(description="Last mode set through this gateway (None if never)")


class ModeIn(Model):
    mode: Mode


class ModeOut(Model):
    mode: Mode
    note: str


class GoalIn(Model):
    x: Finite
    y: Finite
    yaw: Finite = 0.0
    frame_id: Literal["map"] = "map"


class GoalOut(Model):
    id: str
    state: Literal["pending", "rejected", "executing", "canceling", "succeeded", "aborted", "canceled", "failed"]
    frame_id: str
    x: float
    y: float
    yaw: float
    distance_remaining: float | None
    recoveries: int | None
    error_code: int | None
    error_message: str | None


class Navigation(Model):
    heartbeat: bool | None
    heartbeat_age_s: float | None
    status: str | None = Field(description="Last /ugv/nav2_status text")
    action_server_ready: bool
    active_goal: GoalOut | None


U32 = Annotated[int, Field(ge=0, le=0xFFFFFFFF)]
StatValue = bool | int | float | str | None


class MapSeq(Model):
    """Per-layer change counter. Field order is MapStore.LAYERS; the viewer requires every one."""

    trajectory: U32
    grid: U32
    live: U32


class MapStatus(Model):
    epoch: U32 = Field(description="Random per gateway process: a change means every layer's seq restarted")
    seq: MapSeq = Field(description="0 = that layer has received nothing yet")
    stats: dict[str, StatValue] = Field(
        description="Flat scalars, snake_case keys exactly as the ROS stats nodes publish them (not camelCased)"
    )


class Pose(Model):
    """map -> base_link from TF. When no transform has been seen: available false, every other field null."""

    available: bool
    x: float | None
    y: float | None
    z: float | None
    qx: float | None
    qy: float | None
    qz: float | None
    qw: float | None
    age_s: float | None
