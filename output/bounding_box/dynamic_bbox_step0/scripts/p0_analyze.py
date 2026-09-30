"""Passo 0 bbox dinâmico: variantes 1-6 de Rv/perda por caso + espaçamento efetivo (--rl 2)."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
CACHE = HERE / "p0_cache"
FACT = (2.0, 2.0, 1.0)
GRID = (256, 256, 128)
AX = ("y", "x", "z")

df = pd.read_csv(HERE / "p0_cases.csv").sort_values("img_id").reset_index(drop=True)
n = len(df)
shape = df[["sy", "sx", "sz"]].to_numpy(float)
spacing = df[["spy", "spx", "spz"]].to_numpy(float)
vol = shape.prod(axis=1)

coords = []
for i in df.img_id:
    d = np.load(CACHE / f"{i}.npz")
    coords.append([np.sort(d[a].astype(np.int32)) for a in AX])
raw = []
for i in df.img_id:
    d = np.load(CACHE / f"{i}.npz")
    raw.append(np.stack([d[a].astype(np.int32) for a in AX], axis=1))

lab_min = np.array([[c[k][0] for k in range(3)] for c in coords], float)
lab_max = np.array([[c[k][-1] for k in range(3)] for c in coords], float)
lab_ext = lab_max - lab_min + 1
lab_ctr = (lab_min + lab_max + 1) / 2.0


def clip_box(lo, hi):
    lo = np.clip(np.floor(lo), 0, shape)
    hi = np.clip(np.ceil(hi), 0, shape)
    return lo, hi


def evaluate(lo, hi):
    ext = np.maximum(hi - lo, 0)
    rv = 1.0 - ext.prod(axis=1) / vol
    loss = np.empty(n)
    for i in range(n):
        p = raw[i]
        inside = np.all((p >= lo[i]) & (p < hi[i]), axis=1)
        loss[i] = 1.0 - inside.mean()
    return rv, loss, ext


res = {}
# 1 / 2: bbox fixo Opção 1 / Opção 2 (mesmos índices do corte de produção)
for k, tag in ((1, "b1"), (2, "b2")):
    lo = df[[f"{tag}_min_{a}" for a in AX]].to_numpy(float)
    hi = df[[f"{tag}_max_{a}" for a in AX]].to_numpy(float)
    res[k] = evaluate(lo, hi)

# 3: oráculo completo
res[3] = evaluate(lab_min, lab_max + 1)

# 4: oráculo com orçamento de perda — percentis q e 100-q das coordenadas, q único
target = res[1][1].mean()


def box_q(q):
    lo = np.empty((n, 3))
    hi = np.empty((n, 3))
    for i, c in enumerate(coords):
        for k in range(3):
            lo[i, k] = math.floor(np.percentile(c[k], q))
            hi[i, k] = math.floor(np.percentile(c[k], 100 - q)) + 1
    return clip_box(lo, hi)


def mean_loss_q(q):
    lo, hi = box_q(q)
    return evaluate(lo, hi)[1].mean()


a, b = 0.0, 10.0
while mean_loss_q(b) < target:
    b *= 2
for _ in range(40):
    mid = (a + b) / 2
    if mean_loss_q(mid) < target:
        a = mid
    else:
        b = mid
q_star = a if abs(mean_loss_q(a) - target) <= abs(mean_loss_q(b) - target) else b
res[4] = evaluate(*box_q(q_star))

# 5 / 6: oráculos de tamanho e de posição (p95 principal, p100 secundário)
anchor_full = df[["anchor_y", "anchor_x", "anchor_z"]].to_numpy(float) * np.array(FACT)
err = lab_ctr - anchor_full
bias = np.median(err, axis=0)
resid = np.abs(err - bias)
pred_ctr = anchor_full + bias
extra = {}
for pct, key5, key6 in ((95, "5", "6"), (100, "5b", "6b")):
    margin = np.percentile(resid, pct, axis=0)
    lo, hi = clip_box(pred_ctr - lab_ext / 2 - margin, pred_ctr + lab_ext / 2 + margin)
    res[key5] = evaluate(lo, hi)
    size = np.percentile(lab_ext, pct, axis=0)
    lo, hi = clip_box(lab_ctr - size / 2, lab_ctr + size / 2)
    res[key6] = evaluate(lo, hi)
    extra[pct] = dict(margin_vox=margin.tolist(), fixed_size_vox=size.tolist())


def stats(v):
    return dict(mean=float(np.mean(v)), median=float(np.median(v)),
                p5=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)))


names = {1: "1 piso: bbox fixo Opção 1", 2: "2 piso sec.: bbox fixo Opção 2", 3: "3 oráculo completo",
         4: f"4 oráculo c/ orçamento (q={q_star:.4f})", "5": "5 oráculo tamanho p95",
         "6": "6 oráculo posição p95", "5b": "5' oráculo tamanho p100", "6b": "6' oráculo posição p100"}
rows = []
for k, name in names.items():
    rv, loss, _ = res[k]
    s = stats(rv * 100)
    ls = stats(loss * 100)
    rows.append(dict(variante=name, rv_mean=s["mean"], rv_median=s["median"], rv_p5=s["p5"], rv_p95=s["p95"],
                     loss_mean=ls["mean"], loss_median=ls["median"], loss_p95=ls["p95"],
                     loss_max=float(loss.max() * 100), n_loss_pos=int((loss > 1e-12).sum())))
summary = pd.DataFrame(rows)

gap = res[4][0].mean() - res[1][0].mean()
frac = {k: (res[k][0].mean() - res[1][0].mean()) / gap for k in ("5", "6", "5b", "6b", 2, 3)}

sp_rows = []
for k in (1, 4):
    ext = res[k][2]
    eff = ext * spacing / np.array(GRID)
    for j, a in enumerate(AX):
        s = stats(eff[:, j])
        sp_rows.append(dict(variante=k, eixo=a, **s))
native = stats(spacing[:, 0]), stats(spacing[:, 2])
sp = pd.DataFrame(sp_rows)

# validação: perda da variante 1 vs manifesto
diff = np.abs(res[1][1] * 100 - df.option1_loss_pct.to_numpy())
nominal_rv1 = 1 - (408.5473626217631 * 357.1752662733064 * 243.07803316014537) / vol
cal = json.loads(Path("D:/Doutorado/dataset/cropped_manifests/calibration_both_options.json").read_text())
omin = np.array([cal["option1"]["offset_min"][a] for a in AX])
omax = np.array([cal["option1"]["offset_max"][a] for a in AX])
anc_red = df[["anchor_y", "anchor_x", "anchor_z"]].to_numpy(float)
lo_raw = np.floor((anc_red + omin) * np.array(FACT))
hi_raw = np.ceil((anc_red + omax) * np.array(FACT))
clipped_axis = (lo_raw < 0) | (hi_raw > shape)
n_clipped = int(clipped_axis.any(axis=1).sum())
n_clipped_by_axis = clipped_axis.sum(axis=0).tolist()

pd.set_option("display.width", 250)
print("N casos:", n)
print(summary.round(3).to_string(index=False))
print("\nq* =", q_star, " alvo perda média (%):", target * 100, " obtida:", res[4][1].mean() * 100)
print("espaço de ganho Rv_v4 - Rv_v1 (pp):", gap * 100)
print("fração recuperada:", {str(k): round(v, 4) for k, v in frac.items()})
print("\nespaçamento efetivo (mm):")
print(sp.round(4).to_string(index=False))
print("spacing nativo y/x:", native[0], " z:", native[1])
print("\nvalidação perda v1 vs manifesto: max |diff| (pp) =", diff.max())
print("Rv v1 nominal (sem clamp): mean %.3f median %.3f" % (nominal_rv1.mean() * 100, np.median(nominal_rv1) * 100))
print("casos com caixa v1 clampada em algum eixo:", n_clipped, " por eixo y/x/z:", n_clipped_by_axis)
print("margens/tamanhos:", json.dumps(extra))
print("bias centro-âncora (vox):", bias.tolist())
print("extensão label vox: median", np.median(lab_ext, axis=0).tolist(), " p95", np.percentile(lab_ext, 95, axis=0).tolist())

out = dict(n=n, q_star=q_star, target_loss_pct=target * 100, v4_loss_pct=res[4][1].mean() * 100,
           gap_pp=gap * 100, frac={str(k): v for k, v in frac.items()}, extra=extra, bias=bias.tolist())
summary.to_csv(HERE / "p0_summary.csv", index=False)
sp.to_csv(HERE / "p0_spacing.csv", index=False)
(HERE / "p0_meta.json").write_text(json.dumps(out, indent=2))
percase = df[["img_id"]].copy()
for k in (1, 2, 3, 4, "5", "6", "5b", "6b"):
    percase[f"rv_{k}"] = res[k][0] * 100
    percase[f"loss_{k}"] = res[k][1] * 100
percase.to_csv(HERE / "p0_percase.csv", index=False)
