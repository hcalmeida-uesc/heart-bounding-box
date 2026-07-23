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
    compute_per_case_envelope_size,
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


def _envelope_records():
    return pd.DataFrame(
        [
            {
                "img_id": 1,
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
        ]
    )


def test_compute_per_case_envelope_size_matches_hand_calculation():
    result = compute_per_case_envelope_size(_envelope_records(), downscale_factors=(2, 2, 1))

    assert list(result["img_id"]) == [1, 2]

    row1 = result.iloc[0]
    # caso 1: y=[90,105] extent=15; x=[95,110] extent=15; z=[40,102] extent=62 (anchor=100 em tudo)
    assert row1["offset_min_y"] == pytest.approx(-10.0)
    assert row1["offset_max_y"] == pytest.approx(5.0)
    assert row1["envelope_extent_y"] == pytest.approx(15.0)
    assert row1["offset_min_x"] == pytest.approx(-5.0)
    assert row1["offset_max_x"] == pytest.approx(10.0)
    assert row1["envelope_extent_x"] == pytest.approx(15.0)
    assert row1["offset_min_z"] == pytest.approx(-60.0)
    assert row1["offset_max_z"] == pytest.approx(2.0)
    assert row1["envelope_extent_z"] == pytest.approx(62.0)

    row2 = result.iloc[1]
    # caso 2: y=[85,112] extent=27; x=[98,108] extent=10; z=[30,98] extent=68
    assert row2["offset_min_y"] == pytest.approx(-15.0)
    assert row2["offset_max_y"] == pytest.approx(12.0)
    assert row2["envelope_extent_y"] == pytest.approx(27.0)
    assert row2["envelope_extent_x"] == pytest.approx(10.0)
    assert row2["envelope_extent_z"] == pytest.approx(68.0)


def test_compute_per_case_envelope_size_sign_consistency_when_label_encloses_anchor():
    # Caso típico: a âncora (100) cai dentro do intervalo do label em todos os eixos -> offset_min
    # negativo, offset_max positivo. (Note: isso não vale em geral -- ver o caso 2 de
    # _envelope_records() no teste de escala, onde o label em z fica inteiramente abaixo da
    # âncora e offset_max_z sai negativo; esse teste cobre especificamente o caso "engloba".)
    enclosing_records = pd.DataFrame(
        [
            {
                "img_id": 1,
                "anchor_y": 100.0,
                "anchor_x": 100.0,
                "anchor_z": 100.0,
                "label_y_min": 90,
                "label_y_max": 105,
                "label_x_min": 95,
                "label_x_max": 110,
                "label_z_min": 40,
                "label_z_max": 102,
            }
        ]
    )

    result = compute_per_case_envelope_size(enclosing_records, downscale_factors=(2, 2, 1))

    for axis in ("y", "x", "z"):
        assert (result[f"offset_min_{axis}"] < 0).all()
        assert (result[f"offset_max_{axis}"] > 0).all()
        # extent é sempre a diferença positiva entre os dois, nunca negativo.
        assert (result[f"envelope_extent_{axis}"] > 0).all()


def test_compute_per_case_envelope_size_allows_offset_max_negative_when_label_is_below_anchor():
    # Achado real do projeto (2026-07): a âncora da aorta costuma ficar acima de toda a árvore
    # coronária em z -- offset_max_z pode ser negativo (label inteiro abaixo da âncora). A função
    # não deve mascarar isso; envelope_extent continua positivo (é sempre max - min).
    result = compute_per_case_envelope_size(_envelope_records(), downscale_factors=(2, 2, 1))

    row2 = result.iloc[1]
    assert row2["offset_max_z"] == pytest.approx(-2.0)
    assert row2["offset_min_z"] == pytest.approx(-70.0)
    assert row2["envelope_extent_z"] == pytest.approx(68.0)


def test_compute_per_case_envelope_size_scales_full_resolution_per_axis():
    # Fatores diferentes por eixo (2, 3, 1) para expor um bug de aplicar o mesmo fator a todos.
    result = compute_per_case_envelope_size(_envelope_records(), downscale_factors=(2, 3, 1))

    row1 = result.iloc[0]
    assert row1["envelope_extent_full_res_y"] == pytest.approx(row1["envelope_extent_y"] * 2)
    assert row1["envelope_extent_full_res_x"] == pytest.approx(row1["envelope_extent_x"] * 3)
    # fator 1 em z: o downscale preserva o eixo axial, full_res deve ser igual ao reduzido.
    assert row1["envelope_extent_full_res_z"] == pytest.approx(row1["envelope_extent_z"])
    assert row1["envelope_extent_full_res_z"] == pytest.approx(62.0)


def test_compute_per_case_envelope_size_raises_on_missing_column():
    records_df = _envelope_records().drop(columns=["label_z_max"])

    with pytest.raises(ValueError, match="label_z_max"):
        compute_per_case_envelope_size(records_df, downscale_factors=(2, 2, 1))


def test_compute_per_case_envelope_size_raises_listing_all_missing_columns():
    records_df = _envelope_records().drop(columns=["anchor_x", "label_y_min"])

    with pytest.raises(ValueError) as exc_info:
        compute_per_case_envelope_size(records_df, downscale_factors=(2, 2, 1))

    assert "anchor_x" in str(exc_info.value)
    assert "label_y_min" in str(exc_info.value)
