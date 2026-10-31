"""Passo 5: split_option3.csv (mesmos img_id, split e folds de split_original.csv) e checksums SHA-256.

O split vai para o repositório Coronary_Segmentation_deep_learning (data_list/). Os checksums ficam em
cropped_option3/SHA256SUMS (formato de `sha256sum -c`, caminhos relativos à pasta do dataset), com uma
cópia em cropped_manifests/, para conferir depois da cópia para o servidor."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

DS = Path("D:/Doutorado/dataset")
CROP = DS / "cropped_option3"
DL = Path("D:/Doutorado/Coronary_Segmentation_deep_learning/data_list")
OUT = Path("D:/Doutorado/heart-bounding-box/output/bounding_box/cropped_option3")

orig = pd.read_csv(DL / "split_original.csv")
man = pd.read_csv(DS / "cropped_manifests/manifest_option3.csv")
assert set(man.img_id) == set(orig.ID) and (man.status == "ok").all()
orig.to_csv(DL / "split_option3.csv", index=False)
new = pd.read_csv(DL / "split_option3.csv")
assert new.equals(orig)
rep = {"split_option3_linhas": len(new), "split_option3_igual_original": bool(new.equals(orig))}
test = {name: set(pd.read_csv(DL / f).query("split == 'test'").ID) for name, f in
        (("original", "split_original.csv"), ("option1", "split_cropped_option1.csv"),
         ("option2", "split_cropped_option2.csv"), ("option3", "split_option3.csv"))}
rep["teste_n"] = {k: len(v) for k, v in test.items()}
rep["teste_identico_nos_quatro"] = all(v == test["original"] for v in test.values())
folds = {name: pd.read_csv(DL / f).set_index("ID") for name, f in (("option1", "split_cropped_option1.csv"),
                                                                     ("option2", "split_cropped_option2.csv"))}
for k, d in folds.items():
    o = orig.set_index("ID").loc[d.index]
    rep[f"split_fold_{k}_igual_original_nos_casos_dele"] = bool(o.fillna(-1).equals(d.fillna(-1)))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest(), p.relative_to(CROP).as_posix()


files = sorted(CROP.glob("*/*.nii.gz"))
with ThreadPoolExecutor(max_workers=8) as ex:
    sums = list(ex.map(sha, files))
txt = "".join(f"{h}  {p}\n" for h, p in sums)
(CROP / "SHA256SUMS").write_text(txt, encoding="utf-8", newline="\n")
(DS / "cropped_manifests/SHA256SUMS_option3").write_text(txt, encoding="utf-8", newline="\n")
rep["arquivos_com_checksum"] = len(sums)
rep["tamanho_total_GB"] = round(sum(f.stat().st_size for f in files) / 1e9, 2)
print(json.dumps(rep, indent=1, ensure_ascii=False))
(OUT / "split_e_checksums.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
