"""Testes da ROI por âncora composta com caixa por regime de confiança (roi_composite_anchor.py)."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from utils.segmentation.roi_composite_anchor import (
    REGIME_FALLBACK,
    REGIME_PLAUSIBLE,
    compute_composite_roi,
    select_descending,
    summarize_aorta_geometry,
    validate_params,
    volume_reduction_pct,
)

AXES = ("latero_lateral", "antero_posterior", "cranio_caudal")


def _params() -> dict:
    reg = {
        "antero_posterior": {"intercepto": 5.0, "aorta_producao": 0.5, "ascendente": 0.25, "descendente": 0.25,
                             "centro_da_imagem": 0.0},
        "cranio_caudal": {"intercepto": 0.0, "aorta_producao": 0.5, "ascendente": 0.5, "descendente": 0.0,
                          "centro_da_imagem": 0.0},
    }
    return {
        "p": 97.0,
        "vies_vox": {"latero_lateral": 0.0, "antero_posterior": 10.0, "cranio_caudal": -5.0},
        "regressao_ancora": copy.deepcopy(reg),
        "regressao_sinal_S1": copy.deepcopy(reg),
        "limiares": {
            "S1_vox": {"antero_posterior": 50.0, "cranio_caudal": 50.0},
            "S2_d_mm": [40.0, 120.0],
            "S3_coordenadas_vox": {f"{n}_{k}": [-1000.0, 2000.0] for n in ("A1", "asc", "desc") for k in range(3)},
        },
        "regimes": {
            "aorta_detectada_e_plausivel": {
                "offset_inferior_vox": dict(zip(AXES, (-100.0, -80.0, -60.0))),
                "offset_superior_vox": dict(zip(AXES, (100.0, 80.0, 60.0))),
            },
            "aorta_nao_confiavel": {
                "offset_inferior_vox": dict(zip(AXES, (-110.0, -90.0, -70.0))),
                "offset_superior_vox": dict(zip(AXES, (110.0, 90.0, 70.0))),
            },
        },
    }


def _features() -> dict:
    return {"aorta_anchor": [250.0, 266.0, 145.0], "asc_center": [230.0, 280.0, 160.0],
            "desc_center": [260.0, 200.0, 150.0], "desc_found": True, "d_mm": 80.0}


SHAPE = (512, 512, 300)


def _active(out: dict) -> set:
    return {k for k, v in out["signals"].items() if v}


def test_plausible_regime_uses_regression_anchor():
    out = compute_composite_roi(_features(), SHAPE, _params())
    assert out["regime"] == REGIME_PLAUSIBLE
    assert _active(out) == set()
    # látero-lateral = centro; AP = 5 + 0,5·266 + 0,25·280 + 0,25·200 = 258; CC = 0,5·145 + 0,5·160 = 152,5
    assert out["anchor"] == pytest.approx([256.0, 258.0, 152.5])
    assert out["bbox_min"] == (156, 178, 92)
    assert out["bbox_max"] == (356, 338, 213)  # ceil(152,5 + 60) = 213


def test_fallback_regime_uses_center_plus_bias_and_its_own_box():
    f = _features()
    f["d_mm"] = 130.0
    out = compute_composite_roi(f, SHAPE, _params())
    assert out["regime"] == REGIME_FALLBACK
    assert out["anchor"] == pytest.approx([256.0, 266.0, 145.0])
    assert out["bbox_min"] == (146, 176, 75)
    assert out["bbox_max"] == (366, 356, 215)


def test_s1_alone():
    f = _features()
    f["desc_center"] = [260.0, 0.0, 150.0]  # AP previsto = 208, centro + viés = 266 → 58 > 50
    out = compute_composite_roi(f, SHAPE, _params())
    assert _active(out) == {"S1"}
    assert out["regime"] == REGIME_FALLBACK


@pytest.mark.parametrize("d_mm", [39.9, 120.1])
def test_s2_alone(d_mm):
    f = _features()
    f["d_mm"] = d_mm
    out = compute_composite_roi(f, SHAPE, _params())
    assert _active(out) == {"S2"}
    assert out["regime"] == REGIME_FALLBACK


def test_s2_limits_are_inclusive():
    for d in (40.0, 120.0):
        f = _features()
        f["d_mm"] = d
        assert compute_composite_roi(f, SHAPE, _params())["regime"] == REGIME_PLAUSIBLE


def test_s3_alone():
    p = _params()
    p["limiares"]["S3_coordenadas_vox"]["asc_0"] = [0.0, 200.0]
    out = compute_composite_roi(_features(), SHAPE, p)
    assert _active(out) == {"S3"}
    assert out["S3_coordinates_out"] == ["asc_0"]
    assert out["regime"] == REGIME_FALLBACK


def test_production_aorta_absent():
    f = _features()
    f["aorta_anchor"] = None
    out = compute_composite_roi(f, SHAPE, _params())
    # sem A1 a regressão não existe: S1 e S3 não são acionados por valor ausente
    assert _active(out) == {"sem_aorta_producao", "entrada_ausente"}
    assert out["regime"] == REGIME_FALLBACK
    assert out["anchor"] == pytest.approx([256.0, 266.0, 145.0])


def test_descending_absent():
    f = _features()
    f.update(desc_center=None, desc_found=False, d_mm=None)
    out = compute_composite_roi(f, SHAPE, _params())
    assert _active(out) == {"descendente_nao_encontrada", "entrada_ausente"}
    assert out["regime"] == REGIME_FALLBACK


def test_clamp_to_volume_keeps_unclamped_indices():
    f = _features()
    f.update(aorta_anchor=None)
    out = compute_composite_roi(f, (150, 150, 100), _params())
    # âncora de fallback = (75, 85, 45); caixa bruta = (-35, -5, -25) – (185, 175, 115)
    assert out["bbox_min_unclamped"] == [-35, -5, -25]
    assert out["bbox_max_unclamped"] == [185, 175, 115]
    assert out["bbox_min"] == (0, 0, 0)
    assert out["bbox_max"] == (150, 150, 100)


def test_conservative_rounding():
    f = _features()
    f["aorta_anchor"] = [250.0, 266.3, 145.0]  # AP = 258,15
    out = compute_composite_roi(f, SHAPE, _params())
    assert out["bbox_min"][1] == math.floor(258.15 - 80) and out["bbox_max"][1] == math.ceil(258.15 + 80)


def test_invalid_shape_and_params_raise():
    with pytest.raises(ValueError):
        compute_composite_roi(_features(), (0, 512, 300), _params())
    p = _params()
    del p["regimes"]
    with pytest.raises(KeyError):
        validate_params(p)
    p = _params()
    del p["limiares"]["S3_coordenadas_vox"]["desc_2"]
    with pytest.raises(ValueError):
        validate_params(p)
    validate_params(_params())


def test_volume_reduction():
    assert volume_reduction_pct((0, 0, 0), (50, 100, 100), (100, 100, 100)) == pytest.approx(50.0)


# ----------------------------------------------------------------------------- geometria
def _asc_table(n=30, c0=110.0, c1=140.0, r=12.0, hu=400.0):
    return np.array([[z, c0, c1, r, 50.0, 0.0, hu, 30.0, 0.0] for z in range(100, 100 + n)], np.float32)


def _desc_table(zs, c0=130.0, c1=50.0, r=8.0, hu=410.0):
    return np.array([[z, c0, c1, r, 40.0, hu, 30.0, 0.0] for z in zs], np.float32).reshape(-1, 8)


def test_geometry_with_descending_found():
    asc = _asc_table()
    desc = np.vstack([_desc_table(range(100, 130)), _desc_table(range(100, 130), c0=60.0, c1=40.0, hu=-100.0)])
    g = summarize_aorta_geometry(asc, desc, reduced_spacing=0.7, full_spacing=(0.35, 0.35, 0.5))
    assert g["asc_found"] and g["desc_found"]
    # janela de 20 mm da raiz = 40 fatias de 0,5 mm: todas as 30; centro × (2, 2, 1)
    assert g["asc_center"] == pytest.approx([220.0, 280.0, 114.5])
    assert g["desc_center"] == pytest.approx([260.0, 100.0, 114.5])
    assert g["n_desc_slices"] == 30  # o candidato com HU incompatível é descartado
    assert g["d_mm"] == pytest.approx(math.hypot(40 * 0.35, 180 * 0.35))


def test_geometry_descending_needs_min_slices_and_ascending():
    asc = _asc_table()
    g = summarize_aorta_geometry(asc, _desc_table(range(100, 105)), 0.7, (0.35, 0.35, 0.5))
    assert g["asc_found"] and not g["desc_found"] and g["desc_center"] is None and g["d_mm"] is None
    g = summarize_aorta_geometry(np.zeros((0, 9), np.float32), np.zeros((0, 8), np.float32), 0.7, (0.35, 0.35, 0.5))
    assert not g["asc_found"] and not g["desc_found"]


def test_select_descending_prefers_most_posterior_persistent_cluster():
    asc = _asc_table()
    a = _desc_table(range(100, 130), c0=130.0, c1=50.0)
    b = _desc_table(range(100, 130), c0=130.0, c1=90.0)
    ch, support = select_descending(asc, np.vstack([b, a]), 0.7)
    assert support == 30 and np.all(ch[:, 2] == 50.0)
