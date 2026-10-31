"""ROI cardíaca por âncora composta com caixa por regime de confiança (variante A8c).

Método congelado em ``output/bounding_box/apresentacao_proposta/a8c_frozen.json`` (ajustado nos
792 casos fora do teste; ver ``docs/a8_regime_box.md`` e ``docs/apresentacao_proposta.md``).
Determinístico e sem rótulo:

1. pré-processamento de produção (downscale ×(2, 2, 1), limiar de HU, maior componente por fatia);
2. rastreamento da aorta ascendente por Hough (configuração de produção) e, com os mesmos círculos,
   Level Set da aorta: o centróide da máscara é a âncora de produção (A1);
3. centro da ascendente: mediana dos círculos reais na janela de 20 mm a partir da raiz;
4. descendente: Hough com faixa de raio própria (5 a 14 mm) nas colunas posteriores à ascendente,
   seleção por persistência entre fatias (mesma lógica do Caminho 2);
5. regime: "aorta_nao_confiavel" se a aorta de produção não for detectada, a descendente não for
   encontrada, faltar alguma entrada ou algum sinal de plausibilidade (S1, S2, S3) for acionado;
   senão "aorta_plausivel";
6. âncora: eixo látero-lateral = centro da imagem; eixos ântero-posterior e crânio-caudal =
   regressão linear (regime plausível) ou centro da imagem + viés mediano (fallback);
7. caixa do regime: ``floor(âncora + offset_inferior)``, ``ceil(âncora + offset_superior)``
   (exclusivo), recortada a ``[0, shape]``.

Coordenadas em voxels de resolução integral, eixos LAS (0 látero-lateral, 1 ântero-posterior,
2 crânio-caudal). Falhas de detecção não levantam exceção: viram regime de fallback, com o motivo
registrado em ``signals``. Erros de entrada (parâmetros incompletos, shape inválido) levantam.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from numpy.typing import NDArray
from skimage import feature
from skimage.transform import hough_circle, hough_circle_peaks

from .aorta_anchor import detect_ascending_aorta_circles, segment_aorta
from .bounding_box import compute_geometric_centroid
from .pipeline_preprocessing import preprocess_ccta_image

AXES = ("latero_lateral", "antero_posterior", "cranio_caudal")
COEF_NAMES = ("intercepto", "aorta_producao", "ascendente", "descendente", "centro_da_imagem")
REGIME_PLAUSIBLE = "aorta_plausivel"
REGIME_FALLBACK = "aorta_nao_confiavel"
_REGIME_KEYS = {REGIME_PLAUSIBLE: "aorta_detectada_e_plausivel", REGIME_FALLBACK: "aorta_nao_confiavel"}
SIGNAL_NAMES = ("sem_aorta_producao", "descendente_nao_encontrada", "entrada_ausente", "S1", "S2", "S3")

# Descendente e ascendente (Caminho 2, docs/caminho2_aorta_geometry.md); fixos, não calibrados.
DESC_RADIUS_MM = (5.0, 14.0)
DESC_NUM_PEAKS = 25
DESC_PEAK_MIN_DISTANCE_PX = 4
ROOT_WINDOW_MM = 20.0
DISK_FRACTION = 0.7
MAX_DMEAN_HU = 120.0
MAX_STD_HU = 80.0
MAX_HOLE = 0.5
MIN_GAP_PX = 2.0
MIN_LEFT_MM = 0.0
SUPPORT_MM = 6.0
SUPPORT_FRAC = 0.5
TRACK_MM = 12.0
MIN_DESC_SLICES = 10
DOWNSCALE = np.array([2.0, 2.0, 1.0])


# ----------------------------------------------------------------------------- parâmetros
def load_frozen_params(path: str | Path) -> dict[str, Any]:
    """Lê e valida o JSON congelado. Levanta ``KeyError``/``ValueError`` se faltar algo."""
    params = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_params(params)
    return params


def validate_params(params: dict[str, Any]) -> None:
    for key in ("p", "vies_vox", "regressao_ancora", "regressao_sinal_S1", "limiares", "regimes"):
        if key not in params:
            raise KeyError(f"parâmetro congelado ausente: {key}")
    for reg in ("regressao_ancora", "regressao_sinal_S1"):
        for ax in AXES[1:]:
            missing = set(COEF_NAMES) - set(params[reg][ax])
            if missing:
                raise KeyError(f"{reg}/{ax}: coeficientes ausentes {sorted(missing)}")
    lim = params["limiares"]
    if set(AXES[1:]) - set(lim["S1_vox"]) or len(lim["S2_d_mm"]) != 2:
        raise ValueError("limiares S1/S2 incompletos")
    expected = {f"{n}_{k}" for n in ("A1", "asc", "desc") for k in range(3)}
    if set(lim["S3_coordenadas_vox"]) != expected:
        raise ValueError("limiares S3 devem cobrir as 9 coordenadas (A1, asc, desc × 3 eixos)")
    for key in _REGIME_KEYS.values():
        r = params["regimes"][key]
        for side in ("offset_inferior_vox", "offset_superior_vox"):
            if set(AXES) - set(r[side]):
                raise ValueError(f"regime {key}: {side} incompleto")


# ----------------------------------------------------------------------------- características
def _disk_stats(s: NDArray[Any], c0: float, c1: float, r: float) -> tuple[float, float, float]:
    rr = max(1.0, DISK_FRACTION * r)
    a0, a1 = np.ogrid[: s.shape[0], : s.shape[1]]
    m = (a0 - c0) ** 2 + (a1 - c1) ** 2 <= rr**2
    v = s[m]
    w = v[v > -299.5]  # -300 = fora do limiar do pré-processamento
    hole = 1.0 - w.size / max(v.size, 1)
    return (float(w.mean()) if w.size else np.nan, float(w.std()) if w.size else np.nan, hole)


def detect_descending_candidates(
    lcc_image: NDArray[Any], asc_circles: list, pixel_spacing: float, canny_sigma: float
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Tabelas dos círculos da ascendente e dos candidatos à descendente (resolução reduzida).

    asc:  z, c0, c1, r, accum, interp, mean_hu, std_hu, hole
    desc: z, c0, c1, r, accum, mean_hu, std_hu, hole
    """
    radii = np.arange(
        int(np.floor(DESC_RADIUS_MM[0] / pixel_spacing)), int(np.ceil(DESC_RADIUS_MM[1] / pixel_spacing)) + 1
    )
    asc, desc = [], []
    for c in asc_circles:
        z = int(c["slice_index"])
        s = lcc_image[:, :, z]
        # convenção do skimage: center_y = linha = eixo 0; center_x = coluna = eixo 1
        c0, c1, r = float(c["center_y"]), float(c["center_x"]), float(c["radius"])
        interp = bool(c.get("interpolated", False))
        asc.append([z, c0, c1, r, float(c.get("accum", np.nan)), float(interp), *_disk_stats(s, c0, c1, r)])
        if interp:
            continue
        hi1 = int(np.floor(c1))
        if hi1 < 2 * radii[0] + 2:
            continue
        roi = s[:, :hi1]  # colunas posteriores à ascendente (eixo 1 menor)
        edges = feature.canny(roi.astype(float), sigma=canny_sigma)
        h = hough_circle(edges, radii)
        acc, x1, x0, rr = hough_circle_peaks(
            h, radii, min_xdistance=DESC_PEAK_MIN_DISTANCE_PX, min_ydistance=DESC_PEAK_MIN_DISTANCE_PX,
            total_num_peaks=DESC_NUM_PEAKS,
        )
        for a_, p1, p0, r_ in zip(acc, x1, x0, rr):
            desc.append([z, float(p0), float(p1), float(r_), float(a_), *_disk_stats(s, p0, p1, r_)])
    return (np.array(asc, np.float32).reshape(-1, 9), np.array(desc, np.float32).reshape(-1, 8))


def _passing(asc: NDArray[Any], desc: NDArray[Any], rsp: float) -> NDArray[Any]:
    if len(desc) == 0:
        return desc
    am = {int(r[0]): r for r in asc}
    a = np.array([am[int(z)] for z in desc[:, 0]])
    gap = np.hypot(desc[:, 1] - a[:, 1], desc[:, 2] - a[:, 2]) - a[:, 3] - desc[:, 3]
    ok = (
        (np.abs(desc[:, 5] - a[:, 6]) <= MAX_DMEAN_HU) & (desc[:, 6] <= MAX_STD_HU) & (desc[:, 7] <= MAX_HOLE)
        & (gap >= MIN_GAP_PX) & (desc[:, 1] - a[:, 1] >= MIN_LEFT_MM / rsp)
    )
    return desc[ok]


def select_descending(asc: NDArray[Any], desc: NDArray[Any], rsp: float) -> tuple[NDArray[Any], int]:
    """Um candidato por fatia, em torno do agrupamento persistente mais posterior."""
    p = _passing(asc, desc, rsp)
    if len(p) == 0:
        return p, 0
    sup_px, trk_px = SUPPORT_MM / rsp, TRACK_MM / rsp
    zs = p[:, 0].astype(int)
    d = np.hypot(p[:, 1, None] - p[None, :, 1], p[:, 2, None] - p[None, :, 2])
    support = np.array([len(np.unique(zs[d[k] <= sup_px])) for k in range(len(p))])
    strong = np.where(support >= SUPPORT_FRAC * support.max())[0]
    best = strong[np.argmin(p[strong, 2])]
    near = d[best] <= sup_px
    mode = np.median(p[near, 1:3], axis=0)
    mode = np.median(p[np.hypot(p[:, 1] - mode[0], p[:, 2] - mode[1]) <= sup_px, 1:3], axis=0)
    chosen = []
    for z in np.unique(zs):
        q = p[zs == z]
        q = q[np.hypot(q[:, 1] - mode[0], q[:, 2] - mode[1]) <= trk_px]
        if len(q):
            chosen.append(q[np.argmax(q[:, 4])])
    return np.array(chosen).reshape(-1, p.shape[1]), int(support[best])


def summarize_aorta_geometry(
    asc: NDArray[Any], desc: NDArray[Any], reduced_spacing: float, full_spacing: tuple[float, float, float]
) -> dict[str, Any]:
    """Centros da ascendente e da descendente (voxels de resolução integral) e distância axial (mm).

    Ausências ficam como ``None``; ``asc_found``/``desc_found`` registram a falha."""
    fsp = np.asarray(full_spacing, float)
    out: dict[str, Any] = {"asc_found": False, "desc_found": False, "asc_center": None, "desc_center": None,
                           "d_mm": None, "n_asc_circles": int(len(asc)), "n_desc_slices": 0}
    if len(asc) == 0:
        return out
    real = asc[asc[:, 5] == 0]
    z_root = asc[:, 0].min()
    win = real[real[:, 0] <= z_root + ROOT_WINDOW_MM / fsp[2]]
    asc_c = np.array([np.median(win[:, 1]), np.median(win[:, 2]), np.median(win[:, 0])]) * DOWNSCALE
    out.update(asc_found=True, asc_center=asc_c.tolist())
    ch, _ = select_descending(asc, desc, reduced_spacing)
    out["n_desc_slices"] = int(len(ch))
    if len(ch) < MIN_DESC_SLICES:
        return out
    dc = np.array([np.median(ch[:, 1]), np.median(ch[:, 2]), np.median(ch[:, 0])]) * DOWNSCALE
    d0 = (dc[0] - asc_c[0]) * fsp[0]
    d1 = (dc[1] - asc_c[1]) * fsp[1]
    out.update(desc_found=True, desc_center=dc.tolist(), d_mm=float(np.hypot(d0, d1)))
    return out


# ----------------------------------------------------------------------------- caixa
def _vec(v: Any) -> NDArray[np.float64]:
    return np.full(3, np.nan) if v is None else np.asarray(v, float)


def compute_composite_roi(features: dict[str, Any], volume_shape_full: tuple[int, int, int],
                          params: dict[str, Any]) -> dict[str, Any]:
    """Regime, sinais, âncora e caixa a partir das características medidas (sem rótulo).

    ``features``: ``aorta_anchor`` (A1, voxels integrais, ou ``None``), ``asc_center``,
    ``desc_center`` (``None`` se ausentes), ``desc_found`` e ``d_mm``."""
    shape = np.asarray(volume_shape_full, float)
    if shape.shape != (3,) or np.any(shape <= 0):
        raise ValueError(f"shape inválido: {volume_shape_full}")
    a1, asc, desc = _vec(features.get("aorta_anchor")), _vec(features.get("asc_center")), _vec(
        features.get("desc_center"))
    d_mm = features.get("d_mm")
    d_mm = np.nan if d_mm is None else float(d_mm)
    a2 = shape / 2.0
    bias = np.array([params["vies_vox"][ax] for ax in AXES], float)
    lim = params["limiares"]

    def design(k: int) -> NDArray[np.float64]:
        return np.array([1.0, a1[k], asc[k], desc[k], a2[k]])

    def predict(reg: str) -> NDArray[np.float64]:
        out = a2.copy()
        for k in (1, 2):
            out[k] = design(k) @ np.array([params[reg][AXES[k]][c] for c in COEF_NAMES], float)
        return out

    with np.errstate(invalid="ignore"):
        pred_s1 = predict("regressao_sinal_S1")
        thr1 = np.array([lim["S1_vox"][ax] for ax in AXES[1:]], float)
        s1 = bool(np.any(np.abs(pred_s1[1:] - (a2[1:] + bias[1:])) > thr1))
        d_lo, d_hi = lim["S2_d_mm"]
        s2 = bool(d_mm < d_lo or d_mm > d_hi)
        coords = {f"{n}_{k}": arr[k] for n, arr in (("A1", a1), ("asc", asc), ("desc", desc)) for k in range(3)}
        s3_out = sorted(c for c, (lo, hi) in lim["S3_coordenadas_vox"].items() if coords[c] < lo or coords[c] > hi)
    signals = {
        "sem_aorta_producao": bool(np.isnan(a1).any()),
        "descendente_nao_encontrada": not bool(features.get("desc_found", False)),
        "entrada_ausente": bool(np.isnan(a1).any() or np.isnan(asc).any() or np.isnan(desc).any()),
        "S1": s1, "S2": s2, "S3": bool(s3_out),
    }
    regime = REGIME_FALLBACK if any(signals.values()) else REGIME_PLAUSIBLE
    anchor = a2.copy()
    if regime == REGIME_PLAUSIBLE:
        anchor[1:] = predict("regressao_ancora")[1:]
    else:
        anchor[1:] = a2[1:] + bias[1:]
    reg = params["regimes"][_REGIME_KEYS[regime]]
    olo = np.array([reg["offset_inferior_vox"][ax] for ax in AXES], float)
    ohi = np.array([reg["offset_superior_vox"][ax] for ax in AXES], float)
    raw_min = np.floor(anchor + olo)
    raw_max = np.ceil(anchor + ohi)
    bmin = np.clip(raw_min, 0, shape)
    bmax = np.clip(raw_max, 0, shape)
    if np.any(bmax <= bmin):
        raise ValueError(f"caixa vazia após o clamp: {bmin.tolist()} – {bmax.tolist()} (shape {shape.tolist()})")
    return {
        "regime": regime,
        "signals": signals,
        "S3_coordinates_out": s3_out,
        "anchor": anchor.tolist(),
        "bbox_min_unclamped": [int(v) for v in raw_min],
        "bbox_max_unclamped": [int(v) for v in raw_max],
        "bbox_min": tuple(int(v) for v in bmin),
        "bbox_max": tuple(int(v) for v in bmax),
    }


def measure_composite_features(nii_img: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Características sem rótulo de uma imagem: âncora de produção (A1), ascendente, descendente."""
    prep = preprocess_ccta_image(nii_img, config)
    lcc, ssp = prep["lcc_image"], prep["scaled_spacing"]
    full_spacing = tuple(float(v) for v in prep["spacing"])
    psp = float(ssp[0] + ssp[1]) / 2.0
    circles = detect_ascending_aorta_circles(lcc, ssp, config)
    asc, desc = detect_descending_candidates(lcc, circles, psp, config["CIRCLE_DETECTION"]["canny_sigma"])
    geom = summarize_aorta_geometry(asc, desc, float(ssp[0]), full_spacing)
    mask, n_circ = segment_aorta(lcc, ssp, config, detected_circles=circles)
    a1 = None
    if mask.any():
        a1 = (np.asarray(compute_geometric_centroid(mask), float) * DOWNSCALE).tolist()
    return {**geom, "aorta_anchor": a1, "aorta_voxels": int(mask.sum()), "num_circles_detected": int(n_circ),
            "volume_shape_full": tuple(int(v) for v in nii_img.shape), "full_spacing": full_spacing,
            "_asc_table": asc, "_desc_table": desc}


def composite_roi_from_image(image_path: str | Path, config: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Caixa da ROI de uma imagem NIfTI (só a imagem; nenhum rótulo é lido)."""
    nii = nib.load(str(image_path))
    feats = measure_composite_features(nii, config)
    roi = compute_composite_roi(feats, feats["volume_shape_full"], params)
    return {**roi, "features": {k: v for k, v in feats.items() if not k.startswith("_")}}


def volume_reduction_pct(bbox_min: tuple[int, int, int], bbox_max: tuple[int, int, int],
                         volume_shape_full: tuple[int, int, int]) -> float:
    vol = math.prod(int(b) - int(a) for a, b in zip(bbox_min, bbox_max))
    return (1.0 - vol / math.prod(volume_shape_full)) * 100.0


__all__ = [
    "AXES", "REGIME_FALLBACK", "REGIME_PLAUSIBLE", "SIGNAL_NAMES",
    "compute_composite_roi", "composite_roi_from_image", "detect_descending_candidates",
    "load_frozen_params", "measure_composite_features", "select_descending", "summarize_aorta_geometry",
    "validate_params", "volume_reduction_pct",
]
