"""MapStore: latest source per layer, one memoised encode per seq, safe against a writer on another thread."""

from __future__ import annotations

import threading
import time

import pytest

from ugv_api.mapstore import MapStore

LAYERS = ("trajectory", "grid", "live")


def encode_text(source, epoch, seq, stamp_s) -> bytes:
    return f"{epoch}|{seq}|{stamp_s}|{source}".encode()


class Counting:
    """An encode callback that records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, source, epoch, seq, stamp_s) -> bytes:
        self.calls.append((source, epoch, seq, stamp_s))
        return encode_text(source, epoch, seq, stamp_s)


# ------------------------------------------------------------------------------------------- basics


def test_layers_are_those_of_the_contract_in_order():
    assert MapStore.LAYERS == LAYERS


def test_epoch_is_a_uint32_fixed_for_the_life_of_the_store():
    store = MapStore()
    assert isinstance(store.epoch, int) and 0 <= store.epoch <= 0xFFFFFFFF
    store.put("trajectory", "a", 1.0)
    assert store.epoch == store.epoch


def test_epoch_is_random_per_store():
    assert len({MapStore().epoch for _ in range(20)}) > 1


def test_epoch_can_be_pinned_for_tests():
    assert MapStore(epoch=7).epoch == 7


@pytest.mark.parametrize("bad", [-1, 2**32])
def test_pinned_epoch_must_be_a_uint32(bad):
    with pytest.raises(ValueError):
        MapStore(epoch=bad)


def test_every_layer_starts_at_seq_zero_and_has_no_blob():
    store = MapStore()
    assert store.seqs() == {name: 0 for name in LAYERS}
    enc = Counting()
    for name in LAYERS:
        assert store.seq(name) == 0
        assert store.blob(name, enc) is None
    assert enc.calls == []  # nothing to encode, so the encoder is never asked


def test_put_bumps_that_layer_by_one_and_leaves_the_others():
    store = MapStore()
    store.put("grid", "g1", 1.0)
    store.put("grid", "g2", 2.0)
    store.put("live", "d1", 3.0)
    assert store.seq("grid") == 2 and store.seq("live") == 1
    assert store.seqs() == {**{name: 0 for name in LAYERS}, "grid": 2, "live": 1}


def test_seq_wraps_past_zero_because_zero_means_nothing_received():
    store = MapStore()
    store._seq["trajectory"] = 0xFFFFFFFF  # white box: 4 billion puts is too many to make
    store.put("trajectory", "x", 1.0)
    assert store.seq("trajectory") == 1


@pytest.mark.parametrize("call", [
    lambda s: s.put("map", "x", 0.0),
    lambda s: s.seq("map"),
    lambda s: s.blob("map", encode_text),
])
def test_unknown_layer_is_rejected(call):
    with pytest.raises(ValueError, match="unknown layer"):
        call(MapStore())


# ------------------------------------------------------------------------------------ memoised blob


def test_blob_hands_the_encoder_the_stored_reference_epoch_seq_and_stamp():
    store = MapStore(epoch=99)
    source = {"anything": [1, 2, 3]}
    store.put("trajectory", source, 12.5)
    enc = Counting()
    blob = store.blob("trajectory", enc)
    assert blob == encode_text(source, 99, 1, 12.5)
    assert enc.calls[0][0] is source  # put swaps a reference, it never copies


def test_blob_encodes_once_per_seq_and_returns_the_same_bytes_object():
    store = MapStore()
    store.put("trajectory", "v1", 1.0)
    enc = Counting()
    first = store.blob("trajectory", enc)
    assert store.blob("trajectory", enc) is first and store.blob("trajectory", enc) is first
    assert len(enc.calls) == 1


def test_a_new_put_means_a_new_encode_with_the_new_seq_and_stamp():
    store = MapStore(epoch=5)
    enc = Counting()
    store.put("trajectory", "v1", 1.0)
    store.blob("trajectory", enc)
    store.put("trajectory", "v2", 2.0)
    blob = store.blob("trajectory", enc)
    assert blob == encode_text("v2", 5, 2, 2.0)
    assert [c[2] for c in enc.calls] == [1, 2]


def test_layers_do_not_share_a_cache():
    store = MapStore()
    enc = Counting()
    store.put("live", "a", 1.0)
    store.put("grid", "b", 1.0)
    assert store.blob("live", enc) != store.blob("grid", enc)
    assert len(enc.calls) == 2


def test_an_encoder_that_raises_caches_nothing_and_leaves_the_store_usable():
    store = MapStore()
    store.put("live", "bad", 1.0)

    def boom(*_args):
        raise ValueError("cannot encode")

    with pytest.raises(ValueError, match="cannot encode"):
        store.blob("live", boom)
    enc = Counting()
    assert store.blob("live", enc) is not None and len(enc.calls) == 1  # retried, and the lock was released


# ----------------------------------------------------------------------------------------- threads


def test_two_concurrent_blob_calls_for_one_seq_encode_once():
    store = MapStore()
    store.put("trajectory", "v1", 1.0)
    inside, release = threading.Event(), threading.Event()
    calls: list[int] = []

    def slow(source, epoch, seq, stamp_s):
        calls.append(seq)
        inside.set()
        assert release.wait(5)
        return encode_text(source, epoch, seq, stamp_s)

    results: list[bytes | None] = []
    first = threading.Thread(target=lambda: results.append(store.blob("trajectory", slow)))
    second = threading.Thread(target=lambda: results.append(store.blob("trajectory", slow)))
    first.start()
    assert inside.wait(5)  # the first caller is inside its encode
    second.start()
    time.sleep(0.1)  # long enough for the second caller to reach the store
    release.set()
    first.join(5)
    second.join(5)
    assert calls == [1], "the second caller encoded as well"
    assert len(results) == 2 and results[0] is results[1]


def test_blobs_of_different_layers_encode_in_parallel():
    store = MapStore()
    store.put("trajectory", "a", 1.0)
    store.put("live", "b", 1.0)
    both = threading.Barrier(2, timeout=5)  # each encoder waits for the other: only parallel encodes pass

    def meet(source, epoch, seq, stamp_s):
        both.wait()
        return encode_text(source, epoch, seq, stamp_s)

    out: dict[str, bytes | None] = {}
    threads = [threading.Thread(target=lambda n=n: out.__setitem__(n, store.blob(n, meet))) for n in ("trajectory", "live")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert out["trajectory"] and out["live"], "one layer's encode blocked the other"


def test_a_put_during_an_encode_is_never_served_as_the_new_seq():
    store = MapStore(epoch=3)
    store.put("trajectory", "v1", 1.0)
    inside, release = threading.Event(), threading.Event()
    calls: list[tuple] = []

    def slow_first(source, epoch, seq, stamp_s):
        calls.append((source, seq))
        if len(calls) == 1:
            inside.set()
            assert release.wait(5)
        return encode_text(source, epoch, seq, stamp_s)

    results: list[bytes | None] = []
    reader = threading.Thread(target=lambda: results.append(store.blob("trajectory", slow_first)))
    reader.start()
    assert inside.wait(5)
    store.put("trajectory", "v2", 2.0)  # the ROS thread publishes while the HTTP thread is mid-encode
    release.set()
    reader.join(5)

    assert results == [encode_text("v1", 3, 1, 1.0)]  # the in-flight caller got a consistent snapshot of seq 1
    assert store.seq("trajectory") == 2
    fresh = store.blob("trajectory", slow_first)
    assert fresh == encode_text("v2", 3, 2, 2.0), "the seq-1 blob was served as seq 2"
    assert calls == [("v1", 1), ("v2", 2)]


def test_a_writer_and_many_readers_never_see_a_torn_snapshot():
    store = MapStore()
    stop = threading.Event()
    bad: list[str] = []

    def writer():
        for i in range(1, 3001):  # single writer, so the i-th put has seq i
            store.put("live", i, float(i))
        stop.set()

    def encode(source, epoch, seq, stamp_s):
        return f"{seq}|{source}|{stamp_s}".encode()

    finals: list[int] = []

    def reader():
        last = 0
        while True:
            finished = stop.is_set()  # read before the blob, so the last pass starts after the final put
            blob = store.blob("live", encode)
            if blob is not None:
                seq, source, stamp = blob.decode().split("|")
                if not (seq == source and float(stamp) == float(seq)):
                    bad.append(blob.decode())
                if int(seq) < last:
                    bad.append(f"went back from {last} to {seq}")
                last = int(seq)
            if finished:
                finals.append(last)
                return

    threads = [threading.Thread(target=reader) for _ in range(4)] + [threading.Thread(target=writer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not any(t.is_alive() for t in threads)
    assert not bad, bad[:3]
    assert store.seq("live") == 3000
    assert finals == [3000] * 4  # the last pass of every reader saw the newest put


# -------------------------------------------------------------------------------------- demand beat


def test_never_touched_is_not_wanted():
    assert MapStore().wanted(1000.0, 5.0) is False


def test_wanted_holds_for_idle_s_after_the_last_touch():
    store = MapStore()
    store.touch(100.0)
    assert store.wanted(100.0, 5.0)
    assert store.wanted(105.0, 5.0)  # exactly idle_s later still counts
    assert not store.wanted(105.001, 5.0)


def test_a_later_touch_extends_the_window():
    store = MapStore()
    store.touch(100.0)
    store.touch(104.0)
    assert store.wanted(108.0, 5.0)
    assert not store.wanted(109.5, 5.0)


def test_a_clock_that_jumps_back_reads_as_idle_but_a_hair_behind_still_counts():
    store = MapStore()
    store.touch(5000.0)
    assert store.wanted(4999.9995, 5.0)  # another thread read the same clock a moment earlier
    assert not store.wanted(10.0, 5.0)  # sim time was reset: do not stay wanted until it catches up


# ------------------------------------------------------------------------------------------- stats


def test_stats_start_empty():
    assert MapStore().stats() == {}


def test_stats_merge_across_sources():
    store = MapStore()
    store.put_stats("mapping", {"keyframes": 12, "loop_closures": 1})
    store.put_stats("perception", {"depth_hz": 3.2, "calibration_placeholder": False})
    assert store.stats() == {"keyframes": 12, "loop_closures": 1, "depth_hz": 3.2, "calibration_placeholder": False}


def test_a_source_replaces_its_own_previous_values():
    store = MapStore()
    store.put_stats("mapping", {"keyframes": 12, "db_bytes": 100})
    store.put_stats("mapping", {"keyframes": 13})
    assert store.stats() == {"keyframes": 13}  # db_bytes is gone: the source no longer reports it


def test_an_empty_update_clears_that_source_only():
    store = MapStore()
    store.put_stats("a", {"x": 1})
    store.put_stats("b", {"y": 2})
    store.put_stats("a", {})
    assert store.stats() == {"y": 2}


def test_on_a_key_collision_the_source_updated_last_wins():
    store = MapStore()
    store.put_stats("a", {"hz": 1.0})
    store.put_stats("b", {"hz": 2.0})
    assert store.stats() == {"hz": 2.0}
    store.put_stats("a", {"hz": 3.0})
    assert store.stats() == {"hz": 3.0}


def test_only_scalars_survive_and_nested_values_are_dropped_not_passed_through():
    store = MapStore()
    store.put_stats("s", {
        "n": 1, "x": 2.5, "text": "mapping", "flag": True, "none": None,
        "nested": {"a": 1}, "list": [1, 2], "tuple": (1, 2), "raw": b"bytes", "set": {1},
    })
    assert store.stats() == {"n": 1, "x": 2.5, "text": "mapping", "flag": True, "none": None}


def test_non_finite_floats_become_null_so_the_json_stays_valid():
    store = MapStore()
    store.put_stats("s", {"nan": float("nan"), "inf": float("inf"), "ninf": float("-inf"), "ok": 0.0})
    assert store.stats() == {"nan": None, "inf": None, "ninf": None, "ok": 0.0}


def test_non_string_keys_are_dropped():
    store = MapStore()
    store.put_stats("s", {1: "a", "k": "b"})
    assert store.stats() == {"k": "b"}


def test_stats_are_copied_in_and_out():
    store = MapStore()
    values = {"keyframes": 1}
    store.put_stats("s", values)
    values["keyframes"] = 99  # the ROS side reuses its dict
    got = store.stats()
    assert got == {"keyframes": 1}
    got["keyframes"] = 7  # a view mutating its result does not reach the store
    assert store.stats() == {"keyframes": 1}
