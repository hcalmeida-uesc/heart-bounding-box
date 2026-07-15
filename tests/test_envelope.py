"""Testes da etapa 6 (envelope de dimensões máximas) do bounding box cardíaco."""

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

import utils.segmentation.envelope as envelope
from utils.segmentation.envelope import (
    ENVELOPE_RECORD_COLUMNS,
    compute_envelope,
    discover_dataset_blocks,
    process_block,
    process_case_for_envelope,
)


def _write_case(base_path, img_id, img_array, label_array):
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(img_array, affine), base_path / f"{img_id}.img.nii.gz")
    nib.save(nib.Nifti1Image(label_array, affine), base_path / f"{img_id}.label.nii.gz")


def _make_case_arrays(shape=(16, 16, 8)):
    # Fundo em -1000 HU (ar, abaixo do limiar inferior fixo de -300), com um bloco
    # central de tecido (100 HU) grande o bastante para dominar o percentil superior
    # e sobreviver ao threshold + LCC por fatia.
    img = np.full(shape, -1000.0, dtype=np.float32)
    img[4:12, 4:12, 2:6] = 100.0
    label = np.zeros(shape, dtype=np.uint8)
    label[6:8, 6:8, 3:5] = 1
    return img, label


DEFAULT_CONFIG = {
    "DOWNSCALE_METHOD": "opencv",
    "OPENCV_INTERPOLATION": "linear",
    "LCC_PER_SLICE": True,
    "DOWNSCALE_FACTORS": [2, 2, 1],
    "MIN_THRESHOLD": -300,
    "MAX_THRESHOLD_PERCENTILE": 99.7,
    "LOWER_THRESHOLD": {"method": "fixed", "fixed_hu": -300},
}


def test_discover_dataset_blocks_lists_only_folders_with_images(tmp_path):
    (tmp_path / "1-200").mkdir()
    (tmp_path / "1-200" / "1.img.nii.gz").touch()
    (tmp_path / "1-200" / "1.label.nii.gz").touch()
    (tmp_path / "1-200" / "2.img.nii.gz").touch()

    (tmp_path / "201-400").mkdir()
    (tmp_path / "201-400" / "201.img.nii.gz").touch()

    (tmp_path / "empty_dir").mkdir()  # sem *.img.nii.gz, não deve virar bloco

    blocks = discover_dataset_blocks(tmp_path)

    assert blocks == {"1-200": [1, 2], "201-400": [201]}


def test_discover_dataset_blocks_raises_when_nothing_found(tmp_path):
    (tmp_path / "empty_dir").mkdir()

    with pytest.raises(FileNotFoundError, match="Nenhum bloco"):
        discover_dataset_blocks(tmp_path)


def test_process_case_for_envelope_end_to_end(tmp_path):
    img, label = _make_case_arrays()
    _write_case(tmp_path, 1, img, label)

    record = process_case_for_envelope(1, tmp_path, "1-200", DEFAULT_CONFIG)

    assert record["status"] == "ok"
    assert record["error_message"] == ""
    assert record["img_id"] == 1
    assert record["block"] == "1-200"
    # Downscale 2x em Y/X, Z preservado: shape reduzido (8, 8, 8).
    assert record["volume_shape_y"] == 8
    assert record["volume_shape_x"] == 8
    assert record["volume_shape_z"] == 8
    assert record["lcc_voxels"] > 0
    assert record["extent_y"] > 0
    assert record["extent_x"] > 0
    assert record["extent_z"] > 0
    assert record["processing_seconds"] >= 0


def test_process_case_for_envelope_records_error_without_raising(tmp_path):
    # Nenhum arquivo criado em tmp_path para o img_id 999: load deve falhar,
    # e o erro deve ficar registrado no record em vez de propagar.
    record = process_case_for_envelope(999, tmp_path, "1-200", DEFAULT_CONFIG)

    assert record["status"] == "error"
    assert record["error_message"]
    assert record["extent_y"] is None


def test_process_block_resume_skips_already_ok_cases(tmp_path, monkeypatch):
    calls = []

    def fake_process_case(img_id, base_path, block, config):
        calls.append(img_id)
        return {
            "img_id": img_id,
            "block": block,
            "status": "ok",
            "error_message": "",
            "centroid_y": 1.0,
            "centroid_x": 1.0,
            "centroid_z": 1.0,
            "extent_y": 1,
            "extent_x": 1,
            "extent_z": 1,
            "volume_shape_y": 2,
            "volume_shape_x": 2,
            "volume_shape_z": 2,
            "lcc_voxels": 1,
            "processing_seconds": 0.0,
        }

    monkeypatch.setattr(envelope, "process_case_for_envelope", fake_process_case)

    output_csv = tmp_path / "block_1-2.csv"
    df1 = process_block("1-2", [1, 2], tmp_path, {}, output_csv, resume=True, progress=False)
    assert calls == [1, 2]
    assert set(df1["img_id"]) == {1, 2}
    assert (df1["status"] == "ok").all()

    df2 = process_block(
        "1-2", [1, 2, 3], tmp_path, {}, output_csv, resume=True, progress=False
    )
    assert calls == [1, 2, 3]  # só o id novo (3) foi reprocessado
    assert set(df2["img_id"]) == {1, 2, 3}


def test_process_block_retries_previously_errored_cases(tmp_path, monkeypatch):
    output_csv = tmp_path / "block.csv"
    error_row = {col: None for col in ENVELOPE_RECORD_COLUMNS}
    error_row.update(img_id=1, block="b", status="error", error_message="boom")
    pd.DataFrame([error_row]).reindex(columns=ENVELOPE_RECORD_COLUMNS).to_csv(
        output_csv, index=False
    )

    calls = []

    def fake_process_case(img_id, base_path, block, config):
        calls.append(img_id)
        return {
            "img_id": img_id,
            "block": block,
            "status": "ok",
            "error_message": "",
            "centroid_y": 1.0,
            "centroid_x": 1.0,
            "centroid_z": 1.0,
            "extent_y": 1,
            "extent_x": 1,
            "extent_z": 1,
            "volume_shape_y": 2,
            "volume_shape_x": 2,
            "volume_shape_z": 2,
            "lcc_voxels": 1,
            "processing_seconds": 0.0,
        }

    monkeypatch.setattr(envelope, "process_case_for_envelope", fake_process_case)

    df = process_block("b", [1], tmp_path, {}, output_csv, resume=True, progress=False)

    assert calls == [1]
    assert df.loc[df["img_id"] == 1, "status"].item() == "ok"


def test_compute_envelope_takes_max_per_axis_and_scales_to_full_resolution():
    records_df = pd.DataFrame(
        [
            {"img_id": 1, "status": "ok", "extent_y": 100, "extent_x": 90, "extent_z": 200},
            {"img_id": 2, "status": "ok", "extent_y": 80, "extent_x": 120, "extent_z": 180},
            {"img_id": 3, "status": "error", "extent_y": 999, "extent_x": 999, "extent_z": 999},
        ]
    )

    result = compute_envelope(records_df, downscale_factors=(2, 2, 1))

    assert result["num_cases_ok"] == 2
    assert result["num_cases_error"] == 1
    assert result["extent_reduced"] == {"y": 100, "x": 120, "z": 200}
    assert result["extent_full_resolution"] == {"y": 200, "x": 240, "z": 200}
    assert result["argmax_case_per_axis"] == {"y": 1, "x": 2, "z": 1}


def test_compute_envelope_raises_when_no_ok_cases():
    records_df = pd.DataFrame(
        [{"img_id": 1, "status": "error", "extent_y": None, "extent_x": None, "extent_z": None}]
    )

    with pytest.raises(ValueError, match="Nenhum caso"):
        compute_envelope(records_df, downscale_factors=(2, 2, 1))
