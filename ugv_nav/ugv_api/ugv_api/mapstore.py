"""Latest source of each 3D map layer, held until an HTTP request asks for the bytes. Pure Python, no ROS.

The ROS side `put`s a layer's source (a numpy array or a small dict, see app.py for the shape each layer
takes) every time the underlying topic updates; HTTP threads ask for `blob(layer, encode)`. Encoding a
150 k point live scan costs real time, so it happens on the request thread, at most once per `seq`, and not at all
for a layer nobody asks for. `put` only swaps a reference: sources are treated as immutable once stored.

Threads: `put`, `put_stats` and `touch` come from the ROS executor thread; `blob`, `seq`, `seqs`, `stats` and
`wanted` from HTTP handler threads (and `wanted` from a ROS timer). Two locks, never held together in the
other order:

  _lock            guards every field below. Held only for dict reads and writes, never while encoding, so a
                   `put` never waits for an encode.
  _encode_locks[l] serialises encoding of layer l. A second caller for the same seq waits here, then finds
                   the cache filled and returns it instead of encoding again. Layers do not block each other.

`blob` encodes a snapshot (source, seq, stamp) taken under `_lock` and caches the result under the seq *of
that snapshot*. A `put` that lands during the encode therefore bumps `_latest` but never relabels the old
bytes: the caller in flight gets a consistent seq-N blob, and the next call sees seq N+1 != cached N and
encodes again.
"""

from __future__ import annotations

import math
import numbers
import secrets
import threading
from collections.abc import Callable
from typing import Any, NamedTuple

__all__ = ["MapStore"]

_UINT32_MAX = 0xFFFFFFFF


class _Entry(NamedTuple):
    source: object
    seq: int
    stamp_s: float


def _scalar(value: Any) -> tuple[bool, Any]:
    """(keep, value): number, string, boolean and null pass; anything nested is dropped. A non-finite float
    becomes null, because JSON has no NaN and the viewer rejects the whole event on one bad value."""
    if value is None or isinstance(value, (bool, str)):
        return True, value
    if isinstance(value, numbers.Integral):
        return True, int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        return True, number if math.isfinite(number) else None
    return False, None


class MapStore:
    LAYERS = ("trajectory", "grid", "live")

    def __init__(self, *, epoch: int | None = None) -> None:
        """`epoch` is a random uint32 chosen once, so a viewer can tell a restarted gateway (seq starts over)
        from a stale frame. It is never 0, which the app uses to say "no store". Pass one only in tests."""
        if epoch is None:
            epoch = secrets.randbelow(_UINT32_MAX) + 1
        if not 0 <= epoch <= _UINT32_MAX:
            raise ValueError(f"epoch must be a uint32, got {epoch}")
        self.epoch = int(epoch)
        self._lock = threading.Lock()
        self._encode_locks = {name: threading.Lock() for name in self.LAYERS}
        self._latest: dict[str, _Entry | None] = {name: None for name in self.LAYERS}
        self._seq = {name: 0 for name in self.LAYERS}
        self._blobs: dict[str, tuple[int, bytes] | None] = {name: None for name in self.LAYERS}
        self._stats: dict[str, dict[str, Any]] = {}
        self._touched: float | None = None

    def _check(self, layer: str) -> None:
        if layer not in self._seq:
            raise ValueError(f"unknown layer {layer!r}, expected one of {self.LAYERS}")

    # ---- layers -------------------------------------------------------------------------------
    def put(self, layer: str, source: object, stamp_s: float) -> None:
        """Make `source` the newest content of `layer` and bump its seq. Keeps the reference, never copies."""
        self._check(layer)
        with self._lock:
            seq = self._seq[layer] + 1
            if seq > _UINT32_MAX:  # a uint32 on the wire, and 0 means "nothing received": wrap past 0
                seq = 1
            self._seq[layer] = seq
            self._latest[layer] = _Entry(source, seq, float(stamp_s))

    def seq(self, layer: str) -> int:
        """0 = nothing received."""
        self._check(layer)
        with self._lock:
            return self._seq[layer]

    def seqs(self) -> dict[str, int]:
        """Every layer's seq, read at one instant."""
        with self._lock:
            return dict(self._seq)

    def blob(self, layer: str, encode: Callable[[object, int, int, float], bytes]) -> bytes | None:
        """Bytes of the newest source, or None before the first `put`. `encode(source, epoch, seq, stamp_s)`
        runs on the calling thread, once per seq: later calls return the same bytes object. An exception from
        `encode` propagates and leaves nothing cached."""
        self._check(layer)
        with self._encode_locks[layer]:
            with self._lock:
                entry = self._latest[layer]
                if entry is None:
                    return None
                cached = self._blobs[layer]
                if cached is not None and cached[0] == entry.seq:
                    return cached[1]
            data = encode(entry.source, self.epoch, entry.seq, entry.stamp_s)  # outside _lock: puts stay fast
            with self._lock:
                self._blobs[layer] = (entry.seq, data)
            return data

    # ---- statistics ---------------------------------------------------------------------------
    def put_stats(self, source: str, values: dict) -> None:
        """Replace everything `source` reported with `values` (empty clears it). Only scalars are kept."""
        clean: dict[str, Any] = {}
        for key, value in values.items():
            keep, scalar = _scalar(value)
            if keep and isinstance(key, str):
                clean[key] = scalar
        with self._lock:
            self._stats.pop(source, None)  # re-inserted last, so a key two sources report is the newest one's
            if clean:
                self._stats[source] = clean

    def stats(self) -> dict[str, Any]:
        """A flat dict merged across sources; on a shared key the source updated last wins."""
        with self._lock:
            merged: dict[str, Any] = {}
            for values in self._stats.values():
                merged.update(values)
            return merged

    # ---- demand heartbeat ---------------------------------------------------------------------
    def touch(self, now_s: float) -> None:
        """The map view asked for the status just now. `now_s` is any steady clock in seconds, the same one
        `wanted` is given."""
        with self._lock:
            self._touched = float(now_s)

    def wanted(self, now_s: float, idle_s: float) -> bool:
        """True while the last touch is at most `idle_s` old. Never touched: False. A clock that jumped by
        more than `idle_s` either way (a sim-time reset) reads as idle rather than wanted forever; a read a
        few microseconds before the touch it follows (two threads, one clock) is still wanted."""
        with self._lock:
            return self._touched is not None and abs(now_s - self._touched) <= idle_s
