"""Verificações do dataset cropped_option3 (Passo 4) e checagem de origem das Opções 1 e 2.

1. 1000 pares gerados, nenhum erro;
2. releitura de todos os recortes: perda (label recortado × total do label original, contado no
   p0_cache) e redução de volume (shape do recorte × shape original) contra o manifesto;
3. métricas nos 792 e nos 208 (geral e por regime) contra referencia_exploratoria.csv;
4. 10 casos (seed 42, 5 por regime): spacing, dtype, valores idênticos ao original (logo, mesmo HU);
5. 5 casos: ponto do label recortado → mm pelo affine do recorte = mesmo ponto do original;
6. figura de controle com 8 casos (4 por regime);
mais: origem do affine nas Opções 1 e 2 (3 casos cada)."""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO = Path("D:/Doutorado/heart-bounding-box")
sys.path.insert(0, str(REPO / "src"))
from utils.segmentation.bbox_crop import compute_full_resolution_bbox  # noqa: E402

DS = Path("D:/Doutorado/dataset")
ORIG, CROP = DS / "original", DS / "cropped_option3"
OUT = REPO / "output/bounding_box/cropped_option3"
P0 = REPO / "output/bounding_box/p0_cache"
man = pd.read_csv(DS / "cropped_manifests/manifest_option3.csv")
ref = pd.read_csv(OUT / "referencia_exploratoria.csv").set_index("referencia")


def block_of(i):
    lo = (i - 1) // 200 * 200 + 1
    return f"{lo}-{lo + 199}"


def reread(i):
    b = block_of(i)
    ci, cl = nib.load(str(CROP / b / f"{i}.img.nii.gz")), nib.load(str(CROP / b / f"{i}.label.nii.gz"))
    oshape = nib.load(str(ORIG / b / f"{i}.img.nii.gz")).shape
    n_total = len(np.load(P0 / f"{i}.npz")["x"])
    n_in = int((np.asarray(cl.dataobj) == 1).sum())
    return dict(img_id=i, shape_img=ci.shape, shape_lbl=cl.shape, n_total=n_total, n_in=n_in,
                perda=(1 - n_in / n_total) * 100, rv=(1 - np.prod(ci.shape) / np.prod(oshape)) * 100)


def summary(d):
    loss = d.perda.to_numpy()
    return dict(n=len(d), rv_mean=d.rv.mean(), loss_mean=loss.mean(), frac_gt1=(loss > 1).mean() * 100,
                frac_gt5=(loss > 5).mean() * 100, frac_gt20=(loss > 20).mean() * 100, loss_max=loss.max())


if __name__ == "__main__":
    rep = {}
    # 1
    pairs = sum(1 for p in CROP.glob("*/*.img.nii.gz") if (p.parent / p.name.replace(".img.", ".label.")).exists())
    rep["1_pares"] = pairs
    rep["1_erros_manifesto"] = int((man.status != "ok").sum())
    rep["1_linhas_manifesto"] = len(man)
    # 2
    with ProcessPoolExecutor(max_workers=12) as ex:
        rr = pd.DataFrame(list(ex.map(reread, man.img_id.tolist(), chunksize=8)))
    mm = man.merge(rr, on="img_id")
    shp = mm[[f"shape_recorte_{k}" for k in range(3)]].apply(tuple, axis=1)
    rep["2_shape_igual_manifesto"] = int(((shp == mm.shape_img) & (shp == mm.shape_lbl)).sum())
    rep["2_max_dif_perda"] = float((mm.perda - mm.perda_coronarias_pct).abs().max())
    rep["2_max_dif_rv"] = float((mm.rv - mm.reducao_volume_pct).abs().max())
    rep["2_voxels_antes_iguais"] = int((mm.n_total == mm.label_voxels_antes).sum())
    rep["2_voxels_depois_iguais"] = int((mm.n_in == mm.label_voxels_depois).sum())
    mm.to_csv(OUT / "verificacao_releitura.csv", index=False)
    # 3
    met = []
    for cj, sel, rname in (("792", mm.split == "trainval", "congelado, 792"), ("208", mm.split == "test", "congelado, 208")):
        for reg, rs in (("todos", slice(None)), ("aorta_plausivel", "conf"), ("aorta_nao_confiavel", "fb")):
            d = mm[sel] if reg == "todos" else mm[sel & (mm.regime == reg)]
            s = summary(d)
            r = ref.loc[rname]
            pre = "" if reg == "todos" else f"{rs}_"
            row = dict(conjunto=cj, regime=reg, **s)
            for k in s:
                row[f"ref_{k}"] = r[f"{pre}{k}"]
                row[f"dif_{k}"] = s[k] - r[f"{pre}{k}"]
            met.append(row)
    met = pd.DataFrame(met)
    met.to_csv(OUT / "verificacao_metricas.csv", index=False)
    rep["3_max_dif_metricas"] = float(met[[c for c in met if c.startswith("dif_")]].abs().max().max())
    # 4 e 5
    rng = np.random.default_rng(42)
    s10 = sorted(rng.choice(man[man.regime == "aorta_plausivel"].img_id, 5, replace=False).tolist()) + \
        sorted(rng.choice(man[man.regime == "aorta_nao_confiavel"].img_id, 5, replace=False).tolist())
    c4 = []
    for i in s10:
        b, r = block_of(i), man.set_index("img_id").loc[i]
        o, c = nib.load(str(ORIG / b / f"{i}.img.nii.gz")), nib.load(str(CROP / b / f"{i}.img.nii.gz"))
        od, cd = np.asarray(o.dataobj), np.asarray(c.dataobj)
        lo = [int(r[f"bbox_min_{k}"]) for k in range(3)]
        hi = [int(r[f"bbox_max_{k}"]) for k in range(3)]
        sub = od[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        c4.append(dict(img_id=i, regime=r.regime, spacing_igual=tuple(o.header.get_zooms()) == tuple(c.header.get_zooms()),
                       dtype_orig=str(o.get_data_dtype()), dtype_recorte=str(c.get_data_dtype()), valores_identicos=bool(np.array_equal(sub, cd)),
                       hu_min_orig_caixa=int(sub.min()), hu_max_orig_caixa=int(sub.max()), hu_min_recorte=int(cd.min()), hu_max_recorte=int(cd.max()),
                       hu_min_volume=int(od.min()), hu_max_volume=int(od.max())))
    c4 = pd.DataFrame(c4)
    c4.to_csv(OUT / "verificacao_dtype_spacing_hu.csv", index=False)
    rep["4_ok"] = int((c4.spacing_igual & (c4.dtype_orig == c4.dtype_recorte) & c4.valores_identicos).sum())
    c5 = []
    for i in s10[:3] + s10[5:7]:
        b, r = block_of(i), man.set_index("img_id").loc[i]
        lo = np.array([int(r[f"bbox_min_{k}"]) for k in range(3)])
        cl, ol = nib.load(str(CROP / b / f"{i}.label.nii.gz")), nib.load(str(ORIG / b / f"{i}.label.nii.gz"))
        v = np.argwhere(np.asarray(cl.dataobj) == 1)
        v = v[len(v) // 2]
        mm_c = cl.affine @ np.r_[v, 1.0]
        mm_o = ol.affine @ np.r_[v + lo, 1.0]
        c5.append(dict(img_id=i, voxel_recorte=v.tolist(), voxel_original=(v + lo).tolist(), mm_recorte=mm_c[:3].round(4).tolist(),
                       mm_original=mm_o[:3].round(4).tolist(), dif_mm=float(np.abs(mm_c - mm_o).max()),
                       label_original_no_ponto=int(np.asarray(ol.dataobj)[tuple(v + lo)])))
    c5 = pd.DataFrame(c5)
    c5.to_csv(OUT / "verificacao_alinhamento.csv", index=False)
    rep["5_max_dif_mm"] = float(c5.dif_mm.max())
    rep["5_label_no_ponto"] = c5.label_original_no_ponto.tolist()
    # origem nas Opções 1 e 2
    cal = json.loads((DS / "cropped_manifests/calibration_both_options.json").read_text())
    ac = pd.read_csv(REPO / "output/bounding_box/aorta_anchor/all_cases.csv").set_index("img_id")
    mo = pd.read_csv(DS / "cropped_manifests/manifest_all.csv").set_index("img_id")
    co = []
    for opt, col in (("option1", "option1_saved"), ("option2", "option2_saved")):
        for i in mo[mo[col]].index[[0, 300, 600]]:
            b, a = block_of(i), ac.loc[i]
            o = nib.load(str(ORIG / b / f"{i}.img.nii.gz"))
            c = nib.load(str(DS / f"cropped_{opt}" / b / f"{i}.img.nii.gz"))
            bmin, bmax = compute_full_resolution_bbox((a.anchor_y, a.anchor_x, a.anchor_z), cal[opt]["offset_min"],
                                                      cal[opt]["offset_max"], cal["downscale_factors"], o.shape)
            exp = o.affine.copy()
            exp[:3, 3] = o.affine[:3, :3] @ np.array(bmin, float) + o.affine[:3, 3]
            co.append(dict(opcao=opt, img_id=int(i), shape_ok=tuple(np.subtract(bmax, bmin)) == c.shape,
                           dif_affine=float(np.abs(c.affine - exp).max()), origem_original=o.affine[:3, 3].round(3).tolist(),
                           origem_recorte=c.affine[:3, 3].round(3).tolist(), dtype=str(c.get_data_dtype()),
                           spacing_ok=tuple(o.header.get_zooms()) == tuple(c.header.get_zooms())))
    co = pd.DataFrame(co)
    co.to_csv(OUT / "verificacao_origem_opcoes_1_2.csv", index=False)
    rep["origem_opcoes_1_2_max_dif"] = float(co.dif_affine.max())
    # 6 figura
    rng = np.random.default_rng(42)
    s8 = [("Aorta detectada e plausível", int(i)) for i in sorted(rng.choice(man[(man.regime == "aorta_plausivel")].img_id, 4, replace=False))] + \
         [("Aorta não confiável", int(i)) for i in sorted(rng.choice(man[(man.regime == "aorta_nao_confiavel")].img_id, 4, replace=False))]
    fig, axs = plt.subplots(2, 4, figsize=(20, 12.5))
    for ax, (reg, i) in zip(axs.flat, s8):
        b, r = block_of(i), man.set_index("img_id").loc[i]
        lo = [int(r[f"bbox_min_{k}"]) for k in range(3)]
        hi = [int(r[f"bbox_max_{k}"]) for k in range(3)]
        z = (lo[2] + hi[2] - 1) // 2
        o = nib.load(str(ORIG / b / f"{i}.img.nii.gz"))
        sl = np.asarray(o.dataobj[:, :, z], dtype=np.float32)
        lab = np.asarray(nib.load(str(ORIG / b / f"{i}.label.nii.gz")).dataobj[:, :, z]) == 1
        ax.imshow(sl.T, origin="lower", cmap="gray", vmin=-150, vmax=450)
        ax.imshow(np.ma.masked_where(~lab.T, np.ones_like(sl.T)), origin="lower",
                  cmap=matplotlib.colors.ListedColormap(["#ff2d2d"]), vmin=0, vmax=1)
        ax.add_patch(Rectangle((lo[0] - 0.5, lo[1] - 0.5), hi[0] - lo[0], hi[1] - lo[1], fill=False, ec="#ffd400", lw=2))
        ax.set_title(f"{reg}\ncaso {i}, perda {r.perda_coronarias_pct:.1f}%".replace(".", ","), fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("cropped_option3 — corte axial no centro da caixa (imagem original, caixa em amarelo, label em vermelho)",
                 fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3)
    fig.savefig(REPO / "docs/img/cropped_option3_controle.png", dpi=90)
    rep["6_casos_figura"] = [i for _, i in s8]
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    (OUT / "verificacao_resumo.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(met.round(4).to_string(index=False))
