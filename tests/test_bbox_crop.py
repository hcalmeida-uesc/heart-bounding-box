"""Testes das etapas 7-8 do bounding box cardíaco (posicionamento + corte em resolução integral)."""

import numpy as np
import pandas as pd
import pytest

from utils.segmentation.bbox_crop import (
    compute_full_resolution_bbox,
    compute_percentile_offsets,
    compute_real_label_loss,
    crop_volume_and_update_affine,
)


def _offset_records():
    return pd.DataFrame(
        [
            {
                "anchor_y": 100.0, "anchor_x": 100.0, "anchor_z": 100.0,
                "label_y_min": 90, "label_y_max": 105,
                "label_x_min": 95, "label_x_max": 110,
                "label_z_min": 40, "label_z_max": 102,
            },
            {
                "anchor_y": 100.0, "anchor_x": 100.0, "anchor_z": 100.0,
                "label_y_min": 85, "label_y_max": 112,
                "label_x_min": 98, "label_x_max": 108,
                "label_z_min": 30, "label_z_max": 98,
            },
        ]
    )


def test_compute_percentile_offsets_matches_min_max_at_100():
    offset_min, offset_max = compute_percentile_offsets(_offset_records(), offset_percentile=100.0)

    # Mesmos offsets já verificados em calibrate_bbox_offsets para estes dados
    # (y: min(-10,-15)=-15, max(5,12)=12; x: min(-5,-2)=-5, max(10,8)=10; z: min(-60,-70)=-70, max(2,-2)=2).
    assert offset_min == pytest.approx({"y": -15.0, "x": -5.0, "z": -70.0})
    assert offset_max == pytest.approx({"y": 12.0, "x": 10.0, "z": 2.0})


def test_compute_percentile_offsets_raises_on_invalid_percentile():
    with pytest.raises(ValueError, match="offset_percentile"):
        compute_percentile_offsets(_offset_records(), offset_percentile=50.0)

    with pytest.raises(ValueError, match="offset_percentile"):
        compute_percentile_offsets(_offset_records(), offset_percentile=101.0)


def test_compute_percentile_offsets_raises_on_missing_column():
    records_df = _offset_records().drop(columns=["label_z_max"])

    with pytest.raises(ValueError, match="label_z_max"):
        compute_percentile_offsets(records_df, offset_percentile=95.0)


def test_compute_percentile_offsets_raises_on_empty_df():
    records_df = _offset_records().iloc[0:0]

    with pytest.raises(ValueError, match="vazio"):
        compute_percentile_offsets(records_df, offset_percentile=95.0)


def test_compute_full_resolution_bbox_floor_ceil_and_scaling():
    anchor_reduced = (10.0, 20.0, 30.0)
    offset_min_reduced = {"y": -2.3, "x": -1.1, "z": -5.0}
    offset_max_reduced = {"y": 2.7, "x": 3.9, "z": 5.0}

    bbox_min, bbox_max = compute_full_resolution_bbox(
        anchor_reduced, offset_min_reduced, offset_max_reduced,
        downscale_factors=(2, 2, 1), volume_shape_full=(1000, 1000, 1000),
    )

    # y: (10-2.3)*2=15.4 -> floor 15 ; (10+2.7)*2=25.4 -> ceil 26
    # x: (20-1.1)*2=37.8 -> floor 37 ; (20+3.9)*2=47.8 -> ceil 48
    # z: (30-5.0)*1=25.0 -> floor 25 ; (30+5.0)*1=35.0 -> ceil 35 (fator z=1)
    assert bbox_min == (15, 37, 25)
    assert bbox_max == (26, 48, 35)


def test_compute_full_resolution_bbox_clamps_to_volume_shape():
    anchor_reduced = (10.0, 20.0, 30.0)
    offset_min_reduced = {"y": -2.3, "x": -1.1, "z": -5.0}
    offset_max_reduced = {"y": 2.7, "x": 3.9, "z": 5.0}

    bbox_min, bbox_max = compute_full_resolution_bbox(
        anchor_reduced, offset_min_reduced, offset_max_reduced,
        downscale_factors=(2, 2, 1), volume_shape_full=(20, 40, 30),
    )

    # Sem clamp seria (15,37,25)-(26,48,35); shape_full=(20,40,30) corta o maximo em cada eixo.
    assert bbox_min == (15, 37, 25)
    assert bbox_max == (20, 40, 30)


def test_compute_full_resolution_bbox_raises_when_empty_after_clamp():
    # ancora bem fora do volume -> bbox inteiro fica fora, clamp produz min==max.
    anchor_reduced = (10_000.0, 10_000.0, 10_000.0)
    offset_min_reduced = {"y": -1.0, "x": -1.0, "z": -1.0}
    offset_max_reduced = {"y": 1.0, "x": 1.0, "z": 1.0}

    with pytest.raises(ValueError, match="vazio"):
        compute_full_resolution_bbox(
            anchor_reduced, offset_min_reduced, offset_max_reduced,
            downscale_factors=(2, 2, 1), volume_shape_full=(100, 100, 100),
        )


def test_crop_volume_and_update_affine_crops_and_updates_origin_with_negative_diagonal():
    # Affine com diagonal negativa no eixo 0 -- mesma assinatura do dataset real (aff2axcodes
    # confirmou ('L','A','S') com affine[0,0]<0 em todos os casos checados).
    volume = np.arange(6 * 5 * 4).reshape(6, 5, 4).astype(np.float32)
    affine = np.array(
        [
            [-2.0, 0.0, 0.0, 10.0],
            [0.0, 1.5, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    bbox_min = (1, 1, 1)
    bbox_max = (4, 3, 3)

    cropped, new_affine = crop_volume_and_update_affine(volume, affine, bbox_min, bbox_max)

    assert cropped.shape == (3, 2, 2)
    np.testing.assert_array_equal(cropped, volume[1:4, 1:3, 1:3])

    # rotacao/escala inalterada
    np.testing.assert_array_equal(new_affine[:3, :3], affine[:3, :3])
    # nova origem = affine[:3,:3] @ [1,1,1] + affine[:3,3]
    # eixo 0 (diagonal negativa): -2*1 + 10 = 8 (nao 10+2*(-2)=6 nem 10+1*2=12 -- confere o sinal)
    expected_translation = np.array([8.0, 21.5, 31.0])
    np.testing.assert_allclose(new_affine[:3, 3], expected_translation)


def test_crop_volume_and_update_affine_identity_affine_simple_case():
    volume = np.zeros((10, 10, 10), dtype=np.uint8)
    affine = np.eye(4)

    cropped, new_affine = crop_volume_and_update_affine(volume, affine, (2, 3, 4), (5, 6, 7))

    assert cropped.shape == (3, 3, 3)
    np.testing.assert_allclose(new_affine[:3, 3], [2.0, 3.0, 4.0])


def test_compute_real_label_loss_zero_when_fully_inside():
    label = np.zeros((10, 10, 10), dtype=np.uint8)
    label[2:5, 2:5, 2:5] = 1  # 27 voxels

    result = compute_real_label_loss(label, bbox_min=(0, 0, 0), bbox_max=(10, 10, 10))

    assert result["total_voxels"] == 27
    assert result["n_inside"] == 27
    assert result["n_outside"] == 0
    assert result["loss_fraction"] == pytest.approx(0.0)


def test_compute_real_label_loss_partial():
    label = np.zeros((10, 10, 10), dtype=np.uint8)
    label[2:5, 2:5, 2:5] = 1  # y,x,z em {2,3,4}, 27 voxels

    # bbox corta y em [0,4) -- mantem y=2,3 (2 de 3), perde y=4 (1 de 3) em todo x,z.
    result = compute_real_label_loss(label, bbox_min=(0, 0, 0), bbox_max=(4, 10, 10))

    assert result["total_voxels"] == 27
    assert result["n_inside"] == 18  # 2 (y) * 3 (x) * 3 (z)
    assert result["n_outside"] == 9
    assert result["loss_fraction"] == pytest.approx(9 / 27)


def test_compute_real_label_loss_raises_on_empty_label():
    label = np.zeros((5, 5, 5), dtype=np.uint8)

    with pytest.raises(ValueError, match="vazio"):
        compute_real_label_loss(label, bbox_min=(0, 0, 0), bbox_max=(5, 5, 5))
