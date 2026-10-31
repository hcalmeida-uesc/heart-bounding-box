"""Gera o dataset ``cropped_option3``: ROI por âncora composta com caixa por regime (variante A8c).

Uso:
    uv run python src/generate_cropped_option3.py                  # 1000 casos, retomável
    uv run python src/generate_cropped_option3.py --fresh          # ignora progresso salvo
    uv run python src/generate_cropped_option3.py --ids 1,2,3      # só alguns casos

A caixa de cada caso vem de :func:`utils.segmentation.roi_composite_anchor.compute_composite_roi`
com os parâmetros congelados (``a8c_frozen.json``, ajustados nos 792 casos fora do teste). As
características sem rótulo vêm dos caches das mesmas rotinas de produção, como nas Opções 1 e 2
(que reaproveitaram ``all_cases.csv`` sem refazer Hough + Level Set):

- âncora de produção (A1): ``output/bounding_box/aorta_anchor/all_cases.csv`` (× (2, 2, 1));
- círculos da ascendente e candidatos à descendente: ``c2_cache/<id>.npz`` (``c2_extract.py``),
  resumidos por :func:`summarize_aorta_geometry`.

O caminho completo a partir da imagem (:func:`composite_roi_from_image`) é conferido contra este,
caso a caso, em uma amostra (``scripts/check_option3_from_image.py``).

Recorte por slicing puro (``crop_volume_and_update_affine``: mesmo dtype, mesmo spacing, origem do
affine deslocada para a mesma posição física), imagem e label com a mesma caixa (perda parcial
embutida, como na Opção 1). O label só é lido depois da caixa calculada, para medir a perda.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

from utils.segmentation.bbox_crop import compute_real_label_loss, crop_volume_and_update_affine
from utils.segmentation.roi_composite_anchor import (
    SIGNAL_NAMES,
    compute_composite_roi,
    load_frozen_params,
    summarize_aorta_geometry,
    volume_reduction_pct,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("D:/Doutorado/dataset")
ORIGINAL_ROOT = DATASET_ROOT / "original"
OUTPUT_ROOT = DATASET_ROOT / "cropped_option3"
MANIFEST_DIR = DATASET_ROOT / "cropped_manifests"
PARAMS_PATH = REPO_ROOT / "output" / "bounding_box" / "apresentacao_proposta" / "a8c_frozen.json"
ALL_CASES_CSV = REPO_ROOT / "output" / "bounding_box" / "aorta_anchor" / "all_cases.csv"
C2_CACHE = REPO_ROOT / "output" / "bounding_box" / "caminho2_aorta_geometry" / "scripts" / "c2_cache"
SPLIT_CSV = Path("D:/Doutorado/Coronary_Segmentation_deep_learning/data_list/split_original.csv")
DOWNSCALE = np.array([2.0, 2.0, 1.0])


def block_of(img_id: int) -> str:
    lo = (img_id - 1) // 200 * 200 + 1
    return f"{lo}-{lo + 199}"


def _fmt(v: Any) -> str:
    return "" if v is None else ";".join(f"{x:.6g}" for x in v)


def process_case(img_id: int, a1: list[float] | None, split: str, fold: Any, params: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    block = block_of(img_id)
    rec: dict[str, Any] = {"img_id": img_id, "bloco": block, "split": split, "fold": fold, "status": "ok",
                           "erro": ""}
    try:
        src = ORIGINAL_ROOT / block
        nii_img = nib.load(str(src / f"{img_id}.img.nii.gz"))
        shape = tuple(int(s) for s in nii_img.shape)
        fsp = tuple(float(v) for v in nii_img.header.get_zooms()[:3])
        cache = np.load(C2_CACHE / f"{img_id}.npz")
        geom = summarize_aorta_geometry(cache["asc"], cache["desc"], fsp[0] * DOWNSCALE[0], fsp)
        feats = {"aorta_anchor": a1, **geom}
        roi = compute_composite_roi(feats, shape, params)
        bmin, bmax = roi["bbox_min"], roi["bbox_max"]

        # só agora o label: perda e contagem de voxels (não entra na caixa)
        nii_lbl = nib.load(str(src / f"{img_id}.label.nii.gz"))
        img = np.asarray(nii_img.dataobj)
        lbl = np.asarray(nii_lbl.dataobj)
        loss = compute_real_label_loss((lbl == 1).astype(np.uint8), bmin, bmax)
        img_c, aff_i = crop_volume_and_update_affine(img, nii_img.affine, bmin, bmax)
        lbl_c, aff_l = crop_volume_and_update_affine(lbl, nii_lbl.affine, bmin, bmax)
        out = OUTPUT_ROOT / block
        out.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(img_c, aff_i), str(out / f"{img_id}.img.nii.gz"))
        nib.save(nib.Nifti1Image(lbl_c, aff_l), str(out / f"{img_id}.label.nii.gz"))

        rec.update(
            regime=roi["regime"],
            sinais=";".join(s for s in SIGNAL_NAMES if roi["signals"][s]),
            **{f"sinal_{s}": bool(roi["signals"][s]) for s in SIGNAL_NAMES},
            S3_coordenadas=";".join(roi["S3_coordinates_out"]),
            **{f"ancora_{k}": roi["anchor"][k] for k in range(3)},
            **{f"bbox_min_bruto_{k}": roi["bbox_min_unclamped"][k] for k in range(3)},
            **{f"bbox_max_bruto_{k}": roi["bbox_max_unclamped"][k] for k in range(3)},
            **{f"bbox_min_{k}": bmin[k] for k in range(3)},
            **{f"bbox_max_{k}": bmax[k] for k in range(3)},
            **{f"shape_original_{k}": shape[k] for k in range(3)},
            **{f"shape_recorte_{k}": int(img_c.shape[k]) for k in range(3)},
            reducao_volume_pct=volume_reduction_pct(bmin, bmax, shape),
            perda_coronarias_pct=loss["loss_fraction"] * 100.0,
            label_voxels_antes=loss["total_voxels"],
            label_voxels_depois=int((lbl_c == 1).sum()),
            aorta_producao=_fmt(a1), ascendente=_fmt(geom["asc_center"]), descendente=_fmt(geom["desc_center"]),
            d_mm=geom["d_mm"], n_circulos_ascendente=geom["n_asc_circles"], n_fatias_descendente=geom["n_desc_slices"],
            spacing=_fmt(fsp), dtype_imagem=str(img_c.dtype), dtype_label=str(lbl_c.dtype),
        )
    except Exception as exc:  # noqa: BLE001 - erro de caso real fica registrado, o lote segue
        rec.update(status="error", erro=repr(exc))
    rec["segundos"] = time.perf_counter() - t0
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--ids", default="")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    params = load_frozen_params(PARAMS_PATH)
    split = pd.read_csv(SPLIT_CSV).rename(columns={"ID": "img_id"})
    ac = pd.read_csv(ALL_CASES_CSV).set_index("img_id")
    ids = [int(x) for x in args.ids.split(",")] if args.ids else sorted(int(i) for i in split.img_id)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    progress = MANIFEST_DIR / "manifest_option3_progress.csv"
    done = pd.read_csv(progress) if (progress.exists() and not args.fresh) else pd.DataFrame()
    if len(done):
        done = done[done.status == "ok"]
    todo = [i for i in ids if i not in set(done.get("img_id", []))]
    sp = split.set_index("img_id")
    print(f"a processar: {len(todo)} (já feitos: {len(done)})", flush=True)

    def a1_of(i: int) -> list[float] | None:
        r = ac.loc[i]
        if r.status != "ok":
            return None
        return (np.array([r.anchor_y, r.anchor_x, r.anchor_z], float) * DOWNSCALE).tolist()

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_case, i, a1_of(i), str(sp.loc[i, "split"]),
                          (None if pd.isna(sp.loc[i, "k_fold"]) else int(sp.loc[i, "k_fold"])), params) for i in todo]
        for k, f in enumerate(as_completed(futs)):
            rows.append(f.result())
            if k % 50 == 0 or k == len(futs) - 1:
                pd.concat([done, pd.DataFrame(rows)]).to_csv(progress, index=False)
                print(f"{k + 1}/{len(futs)}  {time.time() - t0:.0f}s", flush=True)
    allr = pd.concat([done, pd.DataFrame(rows)]).drop_duplicates("img_id", keep="last").sort_values("img_id")
    allr.to_csv(progress, index=False)
    if not args.ids:
        allr.to_csv(MANIFEST_DIR / "manifest_option3.csv", index=False)
        (MANIFEST_DIR / "a8c_frozen.json").write_text(PARAMS_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"ok={int((allr.status == 'ok').sum())} erros={int((allr.status != 'ok').sum())} "
          f"tempo desta execução={time.time() - t0:.0f}s ({os.cpu_count()} CPUs, {args.workers} processos)")


if __name__ == "__main__":
    main()
