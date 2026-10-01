"""Passo 0d, suplemento: varredura fina de p em (99,5, 100) passo 0,05, porque a grade de 0,5 pula
do último ponto com perda > 0 direto para p = 100 (caixa ~ volume inteiro nos métodos com A1)."""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import p0d_analyze as P

if __name__ == "__main__":
    cases = P.load_cases()
    grid = np.round(np.arange(99.55, 99.951, 0.05), 2)
    tasks = [(m, float(p)) for m in P.METHODS for p in grid]
    with ProcessPoolExecutor(max_workers=6, initializer=P.worker_init_full,
                             initargs=(list(cases.img_id),)) as ex:
        fine = pd.DataFrame(list(ex.map(P.sweep_task, tasks)))
    fine.to_csv(P.OUTDIR / "p0d_sweep_fine_99.5_100.csv", index=False)
    full = pd.concat([pd.read_csv(P.OUTDIR / "p0d_sweep.csv"), fine]).sort_values(["metodo", "p"])
    cons = {"(a) perda máx. <= 1%": full.loss_max <= 1.0,
            "(b) <= 2% dos casos com perda > 1%": full.frac_gt1 <= 2.0,
            "(c) perda média <= 0,156%": full.loss_mean <= 0.156}
    rows = []
    for cname, ok in cons.items():
        for m in P.METHODS:
            s = full[ok & (full.metodo == m)]
            b = s.loc[s.rv_mean.idxmax()]
            rows.append(dict(restricao=cname, **b.to_dict()))
    best = pd.DataFrame(rows)
    best.to_csv(P.OUTDIR / "p0d_best_under_constraints_fine.csv", index=False)
    pd.set_option("display.width", 300)
    print(best.round(3).to_string(index=False))
    print(fine[fine.p >= 99.9].round(3).to_string(index=False))
