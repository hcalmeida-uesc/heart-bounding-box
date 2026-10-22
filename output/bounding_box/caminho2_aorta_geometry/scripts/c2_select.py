"""Caminho 2: seleção da descendente a partir dos candidatos em cache (sem rótulo).

asc:  z, c0, c1, r, accum, interp, mean_hu, std_hu, hole
desc: z, c0, c1, r, accum, mean_hu, std_hu, hole
Eixo 0 = látero-lateral (cresce para a esquerda, LAS); eixo 1 = ântero-posterior (cresce para
anterior). Coordenadas em voxels reduzidos (fator 2 em eixo 0 e 1)."""
import numpy as np

MAX_DMEAN_HU = 120.0      # |média do disco − média da ascendente na mesma fatia| (mesmo pool de contraste)
MAX_STD_HU = 80.0         # homogeneidade do lúmen
MAX_HOLE = 0.5            # fração máxima do disco saturada/fora do limiar
MIN_GAP_PX = 2.0          # distância mínima entre as bordas dos dois círculos (círculos distintos)
MIN_LEFT_MM = 0.0         # descendente à esquerda do centro da ascendente (eixo 0 maior)
SUPPORT_MM = 6.0          # raio de vizinhança para contar suporte (fatias distintas)
SUPPORT_FRAC = 0.5        # agrupamentos com suporte >= 50% do máximo competem pelo mais posterior
TRACK_MM = 12.0           # tolerância por fatia em torno do modo (escolhe o maior acumulador dentro dela)
MIN_SLICES = 10           # mínimo de fatias válidas para aceitar a descendente


def passing(asc, desc, rsp):
    if len(desc) == 0:
        return desc
    am = {int(r[0]): r for r in asc}
    a = np.array([am[int(z)] for z in desc[:, 0]])
    gap = np.hypot(desc[:, 1] - a[:, 1], desc[:, 2] - a[:, 2]) - a[:, 3] - desc[:, 3]
    ok = ((np.abs(desc[:, 5] - a[:, 6]) <= MAX_DMEAN_HU) & (desc[:, 6] <= MAX_STD_HU) & (desc[:, 7] <= MAX_HOLE)
          & (gap >= MIN_GAP_PX) & (desc[:, 1] - a[:, 1] >= MIN_LEFT_MM / rsp))
    return desc[ok]


def select_desc(asc, desc, rsp):
    """Devolve (linhas escolhidas por fatia, modo (c0, c1), suporte do modo)."""
    p = passing(asc, desc, rsp)
    if len(p) == 0:
        return p, None, 0
    sup_px, trk_px = SUPPORT_MM / rsp, TRACK_MM / rsp
    zs = p[:, 0].astype(int)
    d = np.hypot(p[:, 1, None] - p[None, :, 1], p[:, 2, None] - p[None, :, 2])
    support = np.array([len(np.unique(zs[d[k] <= sup_px])) for k in range(len(p))])
    # entre os agrupamentos persistentes, o mais posterior (menor eixo 1): a descendente fica junto à coluna
    strong = np.where(support >= SUPPORT_FRAC * support.max())[0]
    best = strong[np.argmin(p[strong, 2])]
    near = d[best] <= sup_px
    mode = np.median(p[near, 1:3], axis=0)
    mode = np.median(p[np.hypot(p[:, 1] - mode[0], p[:, 2] - mode[1]) <= sup_px, 1:3], axis=0)
    chosen = []
    for z in np.unique(zs):
        q = p[zs == z]
        dist = np.hypot(q[:, 1] - mode[0], q[:, 2] - mode[1])
        q = q[dist <= trk_px]
        if len(q):
            chosen.append(q[np.argmax(q[:, 4])])
    chosen = np.array(chosen).reshape(-1, p.shape[1])
    return chosen, mode, int(support[best])
