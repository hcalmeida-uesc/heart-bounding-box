"""Aplica o bbox calibrado (etapas 7-8) ao dataset local completo, nas duas opções decididas na
reunião de 2026-07-29 (ver docs/STATUS.md), para comparação em treino real.

Uso:
    uv run python src/apply_bbox_full_dataset.py                # roda as duas opções
    uv run python src/apply_bbox_full_dataset.py --fresh        # ignora progresso salvo

Reaproveita só funções já testadas (``aorta_anchor.py``, ``bbox_crop.py``) sobre o
``all_cases.csv`` já processado (994 casos ``status="ok"``) -- não reprocessa segmentação (Hough +
Level Set). Para cada caso incluído em alguma das opções, carrega imagem/label em resolução
integral, recorta e salva com o mesmo nome/estrutura de diretórios do dataset original
(``<output_root>/<block>/<img_id>.{img,label}.nii.gz``).

**Opção 1 (perda parcial tolerada)**: limiar do critério C (``aorta_voxels``, exclusion_pct=5%)
calculado só no treino (``imagecas_splits_train90.json``) e aplicado a todos os splits; offsets do
bbox calibrados só no treino filtrado, percentil 95%. Todo caso do critério C entra na saída,
mesmo com perda parcial (perda real é sempre calculada e registrada no manifest, mas não filtra
inclusão).

**Opção 2 (perda zero garantida)**: limiar do critério C calculado sobre os 994 casos inteiros
(não só treino); offsets calibrados sobre toda a população filtrada (não só treino), percentil
98%. Só entra na saída o caso cujo label real (voxel a voxel, resolução integral) cabe inteiro no
bbox -- perda real computada diretamente na máscara em resolução integral (mais direta que o cache
de máscara em resolução reduzida usado no cálculo original de referência, ver docs/step6v2_aorta_
anchor.md, "Etapas 7-8 ... piloto em 5 casos", achado 4: as duas medidas divergem ~0,4-0,8pp; a
integral é a mais definitiva para o corte de fato aplicado).

Aviso de custo: ~994 casos candidatos, imagem completa lida/recortada/salva por caso incluído em
cada opção (I/O dominante, não processamento). Retomável por bloco (mesmo padrão de
run_aorta_anchor.py) -- reexecutar pula casos já com status="ok" no CSV de progresso.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

from utils.project.batch import run_resumable_batch
from utils.segmentation.aorta_anchor import (
    evaluate_percentile_bbox_tradeoff,
    select_cases_by_aorta_volume_percentile,
)
from utils.segmentation.bbox_crop import (
    compute_full_resolution_bbox,
    compute_percentile_offsets,
    compute_real_label_loss,
    crop_volume_and_update_affine,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

DOWNSCALE_FACTORS = (2.0, 2.0, 1.0)
CRITERION_COLUMN = "aorta_voxels"
EXCLUSION_PCT = 5.0
OPTION1_OFFSET_PERCENTILE = 95.0
OPTION2_OFFSET_PERCENTILE = 98.0
SPLITS_PATH = REPO_ROOT / "config" / "imagecas_splits_train90.json"
ALL_CASES_CSV = REPO_ROOT / "output" / "bounding_box" / "aorta_anchor" / "all_cases.csv"

RECORD_COLUMNS = [
    "img_id",
    "block",
    "status",
    "error_message",
    "in_option1_population",
    "in_option2_population",
    "option1_saved",
    "option1_loss_pct",
    "option2_saved",
    "option2_loss_pct",
]


def prepare_calibration(all_cases: pd.DataFrame) -> dict[str, Any]:
    """Calibra as duas opções sobre ``all_cases`` (994 casos ``status="ok"``), cruzando contra as
    funções já commitadas que produziram os números documentados em docs/step6v2_aorta_anchor.md.
    """
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))["splits"]
    train_ids = [int(x) for x in splits["train"]]
    val_ids = [int(x) for x in splits["val"]]
    test_ids = [int(x) for x in splits["test"]]

    # --- Opção 1: limiar do criterio C via treino, offsets via treino filtrado (percentil 95%) ---
    tradeoff = evaluate_percentile_bbox_tradeoff(
        records_df=all_cases,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        offset_percentile=OPTION1_OFFSET_PERCENTILE,
        downscale_factors=DOWNSCALE_FACTORS,
        exclude_via_criterion=CRITERION_COLUMN,
        exclusion_pct=EXCLUSION_PCT,
    )
    low1, high1 = tradeoff["low_threshold"], tradeoff["high_threshold"]

    ok = all_cases[all_cases["status"] == "ok"]
    train_ok = ok[ok["img_id"].isin(train_ids)]
    in_range1 = (ok[CRITERION_COLUMN] >= low1) & (ok[CRITERION_COLUMN] <= high1)
    population1 = ok.loc[in_range1]
    train_filtered1 = train_ok.loc[
        (train_ok[CRITERION_COLUMN] >= low1) & (train_ok[CRITERION_COLUMN] <= high1)
    ]
    offset_min1, offset_max1 = compute_percentile_offsets(train_filtered1, OPTION1_OFFSET_PERCENTILE)
    extent1_full = {
        axis: (offset_max1[axis] - offset_min1[axis]) * factor
        for axis, factor in zip(("y", "x", "z"), DOWNSCALE_FACTORS)
    }
    for axis in ("y", "x", "z"):
        diff = abs(extent1_full[axis] - tradeoff["bbox_extent_full_resolution"][axis])
        if diff > 1e-6:
            raise AssertionError(
                f"Opção 1: offsets recalculados divergem de evaluate_percentile_bbox_tradeoff "
                f"no eixo {axis} (diff={diff})."
            )
    print(
        f"[Opção 1] limiar aorta_voxels (treino) = [{low1:.1f}, {high1:.1f}]  "
        f"população={len(population1)}/{len(ok)}  "
        f"bbox_extent_full={extent1_full['y']:.1f}x{extent1_full['x']:.1f}x{extent1_full['z']:.1f}  "
        f"(esperado 408.5x357.2x243.1)"
    )

    # --- Opção 2: limiar do criterio C global, offsets sobre toda a população filtrada (98%) ---
    excluded2_df, population2 = select_cases_by_aorta_volume_percentile(all_cases, EXCLUSION_PCT)
    offset_min2, offset_max2 = compute_percentile_offsets(population2, OPTION2_OFFSET_PERCENTILE)
    extent2_full = {
        axis: (offset_max2[axis] - offset_min2[axis]) * factor
        for axis, factor in zip(("y", "x", "z"), DOWNSCALE_FACTORS)
    }
    print(
        f"[Opção 2] população filtrada (global) = {len(population2)}/{len(ok)}  "
        f"({len(excluded2_df)} excluídos pelo critério C)  "
        f"bbox_extent_full={extent2_full['y']:.1f}x{extent2_full['x']:.1f}x{extent2_full['z']:.1f}  "
        f"(esperado 436.2x379.0x257.4, N_final documentado=848)"
    )

    return {
        "downscale_factors": DOWNSCALE_FACTORS,
        "option1": {
            "offset_min": offset_min1,
            "offset_max": offset_max1,
            "bbox_extent_full_resolution": extent1_full,
            "aorta_voxels_low": low1,
            "aorta_voxels_high": high1,
            "population_img_ids": sorted(int(x) for x in population1["img_id"]),
        },
        "option2": {
            "offset_min": offset_min2,
            "offset_max": offset_max2,
            "bbox_extent_full_resolution": extent2_full,
            "population_img_ids": sorted(int(x) for x in population2["img_id"]),
        },
    }


def process_case_for_bbox_crop(
    img_id: int, base_path: str | Path, block: str, config: dict[str, Any]
) -> dict[str, Any]:
    """Recorta um caso (se aplicável) nas duas opções de bbox, reaproveitando ``bbox_crop.py``."""
    case_lookup: dict[int, dict[str, Any]] = config["case_lookup"]
    calibration = config["calibration"]
    in1 = img_id in config["option1_ids"]
    in2_candidate = img_id in config["option2_ids"]

    record: dict[str, Any] = {
        "img_id": img_id,
        "block": block,
        "status": "ok",
        "error_message": "",
        "in_option1_population": in1,
        "in_option2_population": in2_candidate,
        "option1_saved": False,
        "option1_loss_pct": None,
        "option2_saved": False,
        "option2_loss_pct": None,
    }
    if not (in1 or in2_candidate):
        return record

    try:
        row = case_lookup[img_id]
        anchor_reduced = (float(row["anchor_y"]), float(row["anchor_x"]), float(row["anchor_z"]))

        img_path = Path(base_path) / f"{img_id}.img.nii.gz"
        label_path = Path(base_path) / f"{img_id}.label.nii.gz"
        nii_img = nib.load(str(img_path))
        nii_label = nib.load(str(label_path))
        img_full = np.asarray(nii_img.dataobj)
        label_full_raw = np.asarray(nii_label.dataobj)
        label_full_binary = (label_full_raw == 1).astype(np.uint8)
        volume_shape_full = img_full.shape

        if in1:
            opt1 = calibration["option1"]
            bbox_min1, bbox_max1 = compute_full_resolution_bbox(
                anchor_reduced,
                opt1["offset_min"],
                opt1["offset_max"],
                calibration["downscale_factors"],
                volume_shape_full,
            )
            loss1 = compute_real_label_loss(label_full_binary, bbox_min1, bbox_max1)
            record["option1_loss_pct"] = loss1["loss_fraction"] * 100.0

            img_c1, aff_img1 = crop_volume_and_update_affine(
                img_full, nii_img.affine, bbox_min1, bbox_max1
            )
            label_c1, aff_lbl1 = crop_volume_and_update_affine(
                label_full_raw, nii_label.affine, bbox_min1, bbox_max1
            )
            out_dir1 = Path(config["option1_output_root"]) / block
            out_dir1.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(img_c1, aff_img1), str(out_dir1 / f"{img_id}.img.nii.gz"))
            nib.save(nib.Nifti1Image(label_c1, aff_lbl1), str(out_dir1 / f"{img_id}.label.nii.gz"))
            record["option1_saved"] = True

        if in2_candidate:
            opt2 = calibration["option2"]
            bbox_min2, bbox_max2 = compute_full_resolution_bbox(
                anchor_reduced,
                opt2["offset_min"],
                opt2["offset_max"],
                calibration["downscale_factors"],
                volume_shape_full,
            )
            loss2 = compute_real_label_loss(label_full_binary, bbox_min2, bbox_max2)
            record["option2_loss_pct"] = loss2["loss_fraction"] * 100.0

            if loss2["loss_fraction"] <= 1e-9:
                img_c2, aff_img2 = crop_volume_and_update_affine(
                    img_full, nii_img.affine, bbox_min2, bbox_max2
                )
                label_c2, aff_lbl2 = crop_volume_and_update_affine(
                    label_full_raw, nii_label.affine, bbox_min2, bbox_max2
                )
                out_dir2 = Path(config["option2_output_root"]) / block
                out_dir2.mkdir(parents=True, exist_ok=True)
                nib.save(
                    nib.Nifti1Image(img_c2, aff_img2), str(out_dir2 / f"{img_id}.img.nii.gz")
                )
                nib.save(
                    nib.Nifti1Image(label_c2, aff_lbl2), str(out_dir2 / f"{img_id}.label.nii.gz")
                )
                record["option2_saved"] = True
    except Exception as exc:  # noqa: BLE001 - erro de caso real, não deve interromper o lote
        record["status"] = "error"
        record["error_message"] = str(exc)

    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fresh", action="store_true", help="Ignora progresso salvo e reprocessa tudo."
    )
    args = parser.parse_args()

    dataset_root = REPO_ROOT / "dataset"
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset não encontrado em {dataset_root}.")

    all_cases = pd.read_csv(ALL_CASES_CSV)
    calibration = prepare_calibration(all_cases)

    manifest_dir = REPO_ROOT / "output" / "bounding_box" / "cropped_dataset_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = manifest_dir / "calibration_both_options.json"
    with calibration_path.open("w", encoding="utf-8") as fh:
        json.dump(calibration, fh, indent=2, ensure_ascii=False)
    print(f"Calibração salva em {calibration_path}")

    option1_output_root = REPO_ROOT / "output" / "bounding_box" / "cropped_dataset_option1"
    option2_output_root = REPO_ROOT / "output" / "bounding_box" / "cropped_dataset_option2"

    case_lookup = {
        int(row["img_id"]): row for _, row in all_cases.set_index("img_id", drop=False).iterrows()
    }
    cfg = {
        "calibration": calibration,
        "case_lookup": case_lookup,
        "option1_ids": set(calibration["option1"]["population_img_ids"]),
        "option2_ids": set(calibration["option2"]["population_img_ids"]),
        "option1_output_root": option1_output_root,
        "option2_output_root": option2_output_root,
    }

    all_records = []
    for block, block_df in all_cases.groupby("block"):
        img_ids = sorted(int(x) for x in block_df["img_id"])
        base_path = dataset_root / block
        output_csv = manifest_dir / f"block_{block}.csv"
        block_result = run_resumable_batch(
            process_case_for_bbox_crop,
            RECORD_COLUMNS,
            block,
            img_ids,
            base_path,
            cfg,
            output_csv,
            resume=not args.fresh,
            progress=True,
        )
        all_records.append(block_result)

    manifest = pd.concat(all_records, ignore_index=True).sort_values("img_id").reset_index(drop=True)
    manifest_csv = manifest_dir / "manifest_all.csv"
    manifest.to_csv(manifest_csv, index=False)

    n_errors = int((manifest["status"] == "error").sum())
    n1_saved = int(manifest["option1_saved"].sum())
    n2_saved = int(manifest["option2_saved"].sum())
    print(f"\nManifest salvo em {manifest_csv}")
    print(f"Opção 1: {n1_saved} casos salvos em {option1_output_root}")
    print(f"Opção 2: {n2_saved} casos salvos em {option2_output_root} (esperado ~848)")
    if n_errors:
        print(f"AVISO: {n_errors} caso(s) com erro -- ver error_message em {manifest_csv}")


if __name__ == "__main__":
    main()
