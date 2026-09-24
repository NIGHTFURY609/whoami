"""mapping vs localize → RTAB-Map params. Filesystem checks only (tmp_path)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ugv_localization.modes import Mode, ModeError, parse_mode, plan_mode


def _db(tmp_path: Path, content: bytes = b"sqlite") -> Path:
    p = tmp_path / "rtabmap.db"
    p.write_bytes(content)
    return p


def test_m1_parse_mode() -> None:
    assert parse_mode("mapping") is Mode.MAPPING
    assert parse_mode("localize") is Mode.LOCALIZE
    with pytest.raises(ModeError, match="mapping|localize"):
        parse_mode("localization")
    with pytest.raises(ModeError):
        parse_mode("")


def test_m2_mapping_params(tmp_path: Path) -> None:
    plan = plan_mode(Mode.MAPPING, str(tmp_path / "new.db"), fresh=False)
    params = plan.rtabmap_params()
    assert params["Mem/IncrementalMemory"] == "true"
    assert params["Mem/InitWMWithAllNodes"] == "false"
    assert plan.rtabmap_args == ()
    assert plan.database_path == str(tmp_path / "new.db")


def test_m3_mapping_fresh_deletes_db_on_start(tmp_path: Path) -> None:
    plan = plan_mode(Mode.MAPPING, str(_db(tmp_path)), fresh=True)
    assert plan.rtabmap_args == ("-d",)


def test_m4_localize_params_readonly_graph(tmp_path: Path) -> None:
    plan = plan_mode(Mode.LOCALIZE, str(_db(tmp_path)), fresh=False)
    params = plan.rtabmap_params()
    assert params["Mem/IncrementalMemory"] == "false"
    assert params["Mem/InitWMWithAllNodes"] == "true"
    assert plan.rtabmap_args == ()


def test_m5_localize_missing_db_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(ModeError, match="does not exist"):
        plan_mode(Mode.LOCALIZE, str(tmp_path / "missing.db"), fresh=False)


def test_m6_localize_empty_db_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(ModeError, match="empty"):
        plan_mode(Mode.LOCALIZE, str(_db(tmp_path, b"")), fresh=False)


def test_m7_localize_fresh_is_contradiction(tmp_path: Path) -> None:
    with pytest.raises(ModeError, match="fresh"):
        plan_mode(Mode.LOCALIZE, str(_db(tmp_path)), fresh=True)


def test_m8_relative_path_refused(tmp_path: Path) -> None:
    with pytest.raises(ModeError, match="absolute"):
        plan_mode(Mode.MAPPING, "maps/rtabmap.db", fresh=False)


def test_m9_wrong_suffix_refused(tmp_path: Path) -> None:
    with pytest.raises(ModeError, match=r"\.db"):
        plan_mode(Mode.MAPPING, str(tmp_path / "map.sqlite"), fresh=False)


def test_m10_mapping_parent_must_exist(tmp_path: Path) -> None:
    with pytest.raises(ModeError, match="parent"):
        plan_mode(Mode.MAPPING, str(tmp_path / "nope" / "rtabmap.db"), fresh=False)


def test_m11_localize_directory_not_file(tmp_path: Path) -> None:
    d = tmp_path / "dir.db"
    d.mkdir()
    with pytest.raises(ModeError, match="not a file"):
        plan_mode(Mode.LOCALIZE, str(d), fresh=False)


def test_m12_fresh_must_be_bool(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        plan_mode(Mode.MAPPING, str(tmp_path / "a.db"), fresh="true")  # type: ignore[arg-type]


def test_m13_mode_must_be_enum(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        plan_mode("mapping", str(tmp_path / "a.db"), fresh=False)  # type: ignore[arg-type]


def test_m14_params_are_copies(tmp_path: Path) -> None:
    plan = plan_mode(Mode.MAPPING, str(tmp_path / "a.db"), fresh=False)
    plan.rtabmap_params()["Mem/IncrementalMemory"] = "false"
    assert plan.rtabmap_params()["Mem/IncrementalMemory"] == "true"
