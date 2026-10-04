"""Camera calibration contract. Numeric tables only — no images, no invented camera for product use."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ugv_localization.camera import (
    CalibrationError,
    calibration_from_camera_info,
    calibration_to_yaml_dict,
    load_calibration,
    validate_intrinsics,
)

# Numeric contract table (shape/range checks), not a product calibration.
_K = (500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0)
_D5 = (0.01, -0.02, 0.0, 0.0, 0.0)
_R = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
_P = (500.0, 0.0, 320.0, 0.0, 0.0, 500.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def _write_yaml(tmp_path: Path, **over) -> Path:
    body = {
        "image_width": 640,
        "image_height": 480,
        "camera_name": "front_mono",
        "camera_matrix": {"rows": 3, "cols": 3, "data": list(_K)},
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {"rows": 1, "cols": 5, "data": list(_D5)},
        "rectification_matrix": {"rows": 3, "cols": 3, "data": list(_R)},
        "projection_matrix": {"rows": 3, "cols": 4, "data": list(_P)},
    }
    body.update(over)
    path = tmp_path / "cam.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def test_c1_valid_yaml_loads(tmp_path: Path) -> None:
    cal = load_calibration(_write_yaml(tmp_path))
    assert cal.width == 640 and cal.height == 480
    assert cal.k == _K
    assert cal.distortion_model == "plumb_bob"
    assert len(cal.d) == 5


def test_c2_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_calibration(tmp_path / "nope.yaml")


def test_c3_all_zero_k_is_uncalibrated() -> None:
    with pytest.raises(CalibrationError, match="fx"):
        validate_intrinsics(640, 480, (0.0,) * 9)


@pytest.mark.parametrize(
    "k, match",
    [
        ((500.0, 0.0, 700.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0), "cx"),
        ((500.0, 0.0, 320.0, 0.0, 500.0, -1.0, 0.0, 0.0, 1.0), "cy"),
        ((500.0, 0.0, 320.0, 0.0, -5.0, 240.0, 0.0, 0.0, 1.0), "fy"),
        ((500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 2.0), "bottom row"),
        ((500.0, 0.0, 320.0, 1.0, 500.0, 240.0, 0.0, 0.0, 1.0), "bottom row|k\\[3\\]"),
        ((float("nan"), 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0), "finite"),
    ],
)
def test_c4_bad_intrinsics(k, match) -> None:
    with pytest.raises(CalibrationError, match=match):
        validate_intrinsics(640, 480, k)


def test_c5_k_wrong_length() -> None:
    with pytest.raises(CalibrationError, match="9"):
        validate_intrinsics(640, 480, (500.0,) * 8)


@pytest.mark.parametrize("w,h", [(0, 480), (640, 0), (-1, 480)])
def test_c6_bad_dims(w, h) -> None:
    with pytest.raises(CalibrationError, match="width|height"):
        validate_intrinsics(w, h, _K)


def test_c7_distortion_length_must_match_model(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path, distortion_coefficients={"rows": 1, "cols": 4, "data": [0.0, 0.0, 0.0, 0.0]}
    )
    with pytest.raises(CalibrationError, match="plumb_bob"):
        load_calibration(path)


def test_c8_unknown_distortion_model(tmp_path: Path) -> None:
    with pytest.raises(CalibrationError, match="distortion_model"):
        load_calibration(_write_yaml(tmp_path, distortion_model="magic"))


def test_c9_matrix_rows_cols_must_match_data(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, camera_matrix={"rows": 3, "cols": 3, "data": [1.0] * 8})
    with pytest.raises(CalibrationError, match="camera_matrix"):
        load_calibration(path)


def test_c10_projection_must_be_sane(tmp_path: Path) -> None:
    bad_p = list(_P)
    bad_p[0] = 0.0
    with pytest.raises(CalibrationError, match="projection"):
        load_calibration(_write_yaml(tmp_path, projection_matrix={"rows": 3, "cols": 4, "data": bad_p}))


def test_c11_missing_key(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    del data["projection_matrix"]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(CalibrationError, match="projection_matrix"):
        load_calibration(path)


def test_c12_roundtrip_from_camera_info(tmp_path: Path) -> None:
    cal = calibration_from_camera_info(
        camera_name="front_mono",
        width=640,
        height=480,
        distortion_model="plumb_bob",
        d=_D5,
        k=_K,
        r=_R,
        p=_P,
    )
    out = tmp_path / "out.yaml"
    out.write_text(yaml.safe_dump(calibration_to_yaml_dict(cal)), encoding="utf-8")
    again = load_calibration(out)
    assert again == cal


def test_c13_camera_info_with_zero_k_refused() -> None:
    with pytest.raises(CalibrationError):
        calibration_from_camera_info(
            camera_name="x",
            width=640,
            height=480,
            distortion_model="plumb_bob",
            d=_D5,
            k=(0.0,) * 9,
            r=_R,
            p=_P,
        )


def test_c14_empty_distortion_allowed_for_ideal_sim_camera() -> None:
    # Gazebo cameras often publish D=[] with plumb_bob. Treat as zero distortion, not an error.
    cal = calibration_from_camera_info(
        camera_name="sim",
        width=640,
        height=480,
        distortion_model="plumb_bob",
        d=(),
        k=_K,
        r=_R,
        p=_P,
    )
    assert cal.d == (0.0, 0.0, 0.0, 0.0, 0.0)


# --- placeholder flag --------------------------------------------------------------------------------------------


def test_c15_placeholder_defaults_to_false_and_existing_files_load_as_before(tmp_path: Path) -> None:
    cal = load_calibration(_write_yaml(tmp_path))
    assert cal.placeholder is False


@pytest.mark.parametrize("flag", [True, False])
def test_c16_placeholder_is_read_from_the_optional_top_level_key(tmp_path: Path, flag: bool) -> None:
    cal = load_calibration(_write_yaml(tmp_path, placeholder=flag))
    assert cal.placeholder is flag
    assert cal.k == _K  # the flag does not change what is loaded


@pytest.mark.parametrize("bad", ["true", "yes please", 1, 0, None, [True], {"a": 1}])
def test_c17_placeholder_must_be_a_boolean(tmp_path: Path, bad) -> None:
    with pytest.raises(CalibrationError, match="placeholder"):
        load_calibration(_write_yaml(tmp_path, placeholder=bad))


def test_c18_placeholder_survives_a_yaml_roundtrip_and_is_absent_when_false(tmp_path: Path) -> None:
    flagged = load_calibration(_write_yaml(tmp_path, placeholder=True))
    again = tmp_path / "again.yaml"
    again.write_text(yaml.safe_dump(calibration_to_yaml_dict(flagged)), encoding="utf-8")
    assert load_calibration(again) == flagged and load_calibration(again).placeholder is True
    assert "placeholder" not in calibration_to_yaml_dict(load_calibration(_write_yaml(tmp_path)))


_CAMERAS = Path(__file__).resolve().parents[2] / "config" / "cameras"


@pytest.mark.skipif(not _CAMERAS.is_dir(), reason="config/cameras is not next to this test (installed copy)")
def test_c19_a_copy_of_another_cameras_intrinsics_must_be_flagged_as_a_placeholder() -> None:
    """The phone ships with the laptop webcam's K/D until it is calibrated. That is only honest while the file
    says so: a file carrying the laptop's K without `placeholder: true` is a silent lie about the camera."""
    laptop = load_calibration(_CAMERAS / "laptop_webcam_640x480.yaml")
    assert laptop.placeholder is False
    for path in sorted(_CAMERAS.glob("*.yaml")):
        cal = load_calibration(path)
        if path.name != "laptop_webcam_640x480.yaml" and cal.k == laptop.k and cal.d == laptop.d:
            assert cal.placeholder, f"{path.name} carries the laptop webcam's intrinsics but is not flagged"


@pytest.mark.skipif(not _CAMERAS.is_dir(), reason="config/cameras is not next to this test (installed copy)")
def test_c20_the_phone_calibration_is_640x480_and_either_its_own_or_the_flagged_laptop_copy() -> None:
    path = _CAMERAS / "phone_640x480.yaml"
    if not path.is_file():
        pytest.fail("phone_640x480.yaml is missing")
    cal = load_calibration(path)
    laptop = load_calibration(_CAMERAS / "laptop_webcam_640x480.yaml")
    assert cal.camera_name == "phone" and (cal.width, cal.height) == (640, 480)
    if cal.placeholder:  # a stand-in is exactly the laptop's numbers
        assert (cal.k, cal.d, cal.r, cal.p) == (laptop.k, laptop.d, laptop.r, laptop.p)
    else:  # calibrated 2026-10-03: the phone's own K, not the laptop's
        assert cal.k != laptop.k
