"""Testes da etapa 6 v2 (âncora pela aorta) do bounding box cardíaco."""

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

import utils.segmentation.aorta_anchor as aorta_anchor
from utils.processing.preprocessing import downscale_image_ndi
from utils.segmentation.aorta_anchor import (
    AORTA_ANCHOR_RECORD_COLUMNS,
    calibrate_bbox_offsets,
    calibrate_bbox_size_joint,
    process_block,
    process_case_for_aorta_anchor,
    validate_coverage,
)

DEFAULT_CONFIG = {
    "DOWNSCALE_METHOD": "opencv",
    "OPENCV_INTERPOLATION": "linear",
    "LCC_PER_SLICE": True,
    "DOWNSCALE_FACTORS": [2, 2, 1],
    "MIN_THRESHOLD": -300,
    "MAX_THRESHOLD_PERCENTILE": 99.7,
    "LOWER_THRESHOLD": {"method": "fixed", "fixed_hu": -300},
    "CIRCLE_DETECTION": {},
    "LEVEL_SET": {},
    "USE_GPU": False,
}


def _write_case(base_path, img_id, img_array, label_array):
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(img_array, affine), base_path / f"{img_id}.img.nii.gz")
    nib.save(nib.Nifti1Image(label_array, affine), base_path / f"{img_id}.label.nii.gz")


def _make_case_arrays(shape=(16, 16, 8)):
    img = np.full(shape, -1000.0, dtype=np.float32)
    img[4:12, 4:12, 2:6] = 100.0
    label = np.zeros(shape, dtype=np.uint8)
    label[6:8, 6:8, 3:5] = 1
    return img, label


def _fake_segment_aorta_ok(lcc_image, scaled_spacing, config):
    mask = np.zeros_like(lcc_image, dtype=np.uint8)
    mask[3:5, 3:5, 3:5] = 1  # âncora conhecida: centróide em (3.5, 3.5, 3.5)
    return mask, 12


def _fake_segment_aorta_not_found(lcc_image, scaled_spacing, config):
    return np.zeros_like(lcc_image, dtype=np.uint8), 0


def test_process_case_ok_path_computes_anchor_and_label_bounds(tmp_path, monkeypatch):
    monkeypatch.setattr(aorta_anchor, "segment_aorta", _fake_segment_aorta_ok)
    img, label = _make_case_arrays()
    _write_case(tmp_path, 1, img, label)

    record = process_case_for_aorta_anchor(1, tmp_path, "1-200", DEFAULT_CONFIG)

    assert record["status"] == "ok"
    assert record["error_message"] == ""
    assert record["anchor_y"] == pytest.approx(3.5)
    assert record["anchor_x"] == pytest.approx(3.5)
    assert record["anchor_z"] == pytest.approx(3.5)
    assert record["aorta_voxels"] == 8
    assert record["num_circles_detected"] == 12
    # label é reamostrado (nearest-neighbor) pelas etapas 1-4 antes dos bounds serem calculados
    # (mesma resolução reduzida de lcc_image/lcc_mask) — deriva o esperado com a mesma função de
    # downscale usada internamente, em vez de fixar índices calculados à mão.
    downscaled_label = downscale_image_ndi(label, DEFAULT_CONFIG["DOWNSCALE_FACTORS"], order=0)
    ys, xs, zs = np.nonzero(downscaled_label)
    assert record["label_y_min"] == int(ys.min())
    assert record["label_y_max"] == int(ys.max())
    assert record["label_x_min"] == int(xs.min())
    assert record["label_x_max"] == int(xs.max())
    assert record["label_z_min"] == int(zs.min())
    assert record["label_z_max"] == int(zs.max())
    assert record["processing_seconds"] >= 0


def test_process_case_records_error_when_aorta_not_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(aorta_anchor, "segment_aorta", _fake_segment_aorta_not_found)
    img, label = _make_case_arrays()
    _write_case(tmp_path, 1, img, label)

    record = process_case_for_aorta_anchor(1, tmp_path, "1-200", DEFAULT_CONFIG)

    assert record["status"] == "error"
    assert record["error_message"]
    assert record["anchor_y"] is None


def test_process_case_records_error_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(aorta_anchor, "segment_aorta", _fake_segment_aorta_ok)

    record = process_case_for_aorta_anchor(999, tmp_path, "1-200", DEFAULT_CONFIG)

    assert record["status"] == "error"
    assert record["error_message"]


def test_process_block_persists_aorta_anchor_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(aorta_anchor, "segment_aorta", _fake_segment_aorta_ok)
    img, label = _make_case_arrays()
    _write_case(tmp_path, 1, img, label)

    output_csv = tmp_path / "block_1-200.csv"
    df = process_block("1-200", [1], tmp_path, DEFAULT_CONFIG, output_csv, progress=False)

    assert list(df.columns) == AORTA_ANCHOR_RECORD_COLUMNS
    assert df.loc[df["img_id"] == 1, "status"].item() == "ok"
    saved = pd.read_csv(output_csv)
    assert list(saved.columns) == AORTA_ANCHOR_RECORD_COLUMNS


def _calibration_records():
    return pd.DataFrame(
        [
            {
                "img_id": 1,
                "status": "ok",
                "anchor_y": 100.0,
                "anchor_x": 100.0,
                "anchor_z": 100.0,
                "label_y_min": 90,
                "label_y_max": 105,
                "label_x_min": 95,
                "label_x_max": 110,
                "label_z_min": 40,
                "label_z_max": 102,
            },
            {
                "img_id": 2,
                "status": "ok",
                "anchor_y": 100.0,
                "anchor_x": 100.0,
                "anchor_z": 100.0,
                "label_y_min": 85,
                "label_y_max": 112,
                "label_x_min": 98,
                "label_x_max": 108,
                "label_z_min": 30,
                "label_z_max": 98,
            },
            {
                "img_id": 3,
                "status": "error",
                "anchor_y": None,
                "anchor_x": None,
                "anchor_z": None,
                "label_y_min": None,
                "label_y_max": None,
                "label_x_min": None,
                "label_x_max": None,
                "label_z_min": None,
                "label_z_max": None,
            },
        ]
    )


def test_calibrate_bbox_offsets_takes_conservative_min_max_per_axis():
    calibration = calibrate_bbox_offsets(_calibration_records(), downscale_factors=(2, 2, 1))

    assert calibration["num_cases_calibration"] == 2
    # y: min(90-100, 85-100) = -15 ; max(105-100, 112-100) = 12
    assert calibration["offset_min_reduced"]["y"] == pytest.approx(-15.0)
    assert calibration["offset_max_reduced"]["y"] == pytest.approx(12.0)
    # x: min(95-100, 98-100) = -5 ; max(110-100, 108-100) = 10
    assert calibration["offset_min_reduced"]["x"] == pytest.approx(-5.0)
    assert calibration["offset_max_reduced"]["x"] == pytest.approx(10.0)
    # z: min(40-100, 30-100) = -70 ; max(102-100, 98-100) = 2
    assert calibration["offset_min_reduced"]["z"] == pytest.approx(-70.0)
    assert calibration["offset_max_reduced"]["z"] == pytest.approx(2.0)
    # escala para resolução integral (fatores 2,2,1)
    assert calibration["offset_min_full_resolution"]["y"] == pytest.approx(-30.0)
    assert calibration["offset_max_full_resolution"]["x"] == pytest.approx(20.0)
    assert calibration["offset_max_full_resolution"]["z"] == pytest.approx(2.0)
    assert calibration["argmin_case_per_axis"]["y"] == 2  # 85-100=-15 é o mínimo
    assert calibration["argmax_case_per_axis"]["y"] == 2  # 112-100=12 é o máximo


def test_calibrate_bbox_offsets_raises_when_no_ok_cases():
    records_df = pd.DataFrame([{"img_id": 1, "status": "error"}])

    with pytest.raises(ValueError, match="Nenhum caso"):
        calibrate_bbox_offsets(records_df, downscale_factors=(2, 2, 1))


def test_validate_coverage_flags_case_outside_calibrated_bbox():
    calibration = calibrate_bbox_offsets(_calibration_records(), downscale_factors=(2, 2, 1))

    # Caso novo (fora da calibração): label_z_min = 20 fica abaixo do bbox calibrado
    # (anchor_z + offset_min_z = 100 - 70 = 30), então covered_z deve ser False.
    held_out = pd.DataFrame(
        [
            {
                "img_id": 42,
                "status": "ok",
                "anchor_y": 100.0,
                "anchor_x": 100.0,
                "anchor_z": 100.0,
                "label_y_min": 90,
                "label_y_max": 105,
                "label_x_min": 95,
                "label_x_max": 105,
                "label_z_min": 20,
                "label_z_max": 95,
            }
        ]
    )

    result = validate_coverage(held_out, calibration)

    row = result.iloc[0]
    assert bool(row["covered_y"]) is True
    assert bool(row["covered_x"]) is True
    assert bool(row["covered_z"]) is False
    assert bool(row["covered"]) is False


def test_calibrate_bbox_size_joint_at_100_uses_largest_volume_case_verbatim():
    calibration = calibrate_bbox_size_joint(
        _calibration_records(), downscale_factors=(2, 2, 1), percentile=100
    )

    # caso 2: extent y=27 (85-112), x=10 (98-108), z=68 (30-98) -> volume=18360
    # caso 1: extent y=15, x=15, z=62 -> volume=13950 (menor) -> caso 2 deve ser escolhido.
    assert calibration["method"] == "joint_per_patient"
    assert calibration["reference_case_img_id"] == 2
    assert calibration["num_cases_calibration"] == 2
    # offsets vêm inteiramente do caso 2, nunca misturados com o caso 1.
    assert calibration["offset_min_reduced"] == pytest.approx({"y": -15.0, "x": -2.0, "z": -70.0})
    assert calibration["offset_max_reduced"] == pytest.approx({"y": 12.0, "x": 8.0, "z": -2.0})


def test_calibrate_bbox_size_joint_at_50_uses_smaller_volume_case():
    calibration = calibrate_bbox_size_joint(
        _calibration_records(), downscale_factors=(2, 2, 1), percentile=50
    )

    # com 2 casos, percentile=50 (rank 0 em ordem ascendente de volume) escolhe o caso 1
    # (volume 13950), o de menor volume entre os dois.
    assert calibration["reference_case_img_id"] == 1
    assert calibration["offset_min_reduced"] == pytest.approx({"y": -10.0, "x": -5.0, "z": -60.0})
    assert calibration["offset_max_reduced"] == pytest.approx({"y": 5.0, "x": 10.0, "z": 2.0})


def test_calibrate_bbox_size_joint_never_exceeds_marginal_extent():
    # Invariante central do método: por nunca combinar o extremo de um eixo de um paciente com
    # o de outro eixo de outro paciente, o volume do bbox conjunto (de um único paciente real)
    # não deve exceder o volume do bbox marginal (calibrate_bbox_offsets).
    marginal = calibrate_bbox_offsets(_calibration_records(), downscale_factors=(2, 2, 1))
    joint = calibrate_bbox_size_joint(
        _calibration_records(), downscale_factors=(2, 2, 1), percentile=100
    )

    marginal_extent = marginal["bbox_extent_reduced"]
    joint_extent = joint["bbox_extent_reduced"]
    marginal_volume = marginal_extent["y"] * marginal_extent["x"] * marginal_extent["z"]
    joint_volume = joint_extent["y"] * joint_extent["x"] * joint_extent["z"]

    assert joint_volume <= marginal_volume


def test_calibrate_bbox_size_joint_scales_to_full_resolution():
    calibration = calibrate_bbox_size_joint(
        _calibration_records(), downscale_factors=(2, 2, 1), percentile=100
    )

    assert calibration["offset_min_full_resolution"]["y"] == pytest.approx(-30.0)
    assert calibration["offset_max_full_resolution"]["x"] == pytest.approx(16.0)
    assert calibration["offset_max_full_resolution"]["z"] == pytest.approx(-2.0)  # fator z=1


def test_calibrate_bbox_size_joint_raises_when_no_ok_cases():
    records_df = pd.DataFrame([{"img_id": 1, "status": "error"}])

    with pytest.raises(ValueError, match="Nenhum caso"):
        calibrate_bbox_size_joint(records_df, downscale_factors=(2, 2, 1))


def test_calibrate_bbox_size_joint_raises_on_invalid_percentile():
    with pytest.raises(ValueError, match="percentile"):
        calibrate_bbox_size_joint(_calibration_records(), downscale_factors=(2, 2, 1), percentile=0)

    with pytest.raises(ValueError, match="percentile"):
        calibrate_bbox_size_joint(
            _calibration_records(), downscale_factors=(2, 2, 1), percentile=101
        )


def test_calibrate_bbox_size_joint_is_compatible_with_validate_coverage():
    calibration = calibrate_bbox_size_joint(
        _calibration_records(), downscale_factors=(2, 2, 1), percentile=100
    )

    held_out = pd.DataFrame(
        [
            {
                "img_id": 42,
                "status": "ok",
                "anchor_y": 100.0,
                "anchor_x": 100.0,
                "anchor_z": 100.0,
                "label_y_min": 90,
                "label_y_max": 105,
                "label_x_min": 99,
                "label_x_max": 107,
                "label_z_min": 35,
                "label_z_max": 95,
            }
        ]
    )

    result = validate_coverage(held_out, calibration)
    assert "covered" in result.columns
    assert bool(result.iloc[0]["covered"]) is True


def test_validate_coverage_marks_case_inside_calibrated_bbox_as_covered():
    calibration = calibrate_bbox_offsets(_calibration_records(), downscale_factors=(2, 2, 1))

    held_out = pd.DataFrame(
        [
            {
                "img_id": 42,
                "status": "ok",
                "anchor_y": 100.0,
                "anchor_x": 100.0,
                "anchor_z": 100.0,
                "label_y_min": 90,
                "label_y_max": 105,
                "label_x_min": 97,
                "label_x_max": 108,
                "label_z_min": 35,
                "label_z_max": 100,
            }
        ]
    )

    result = validate_coverage(held_out, calibration)

    assert bool(result.iloc[0]["covered"]) is True
