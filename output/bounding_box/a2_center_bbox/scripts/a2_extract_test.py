"""Bbox centrado no volume: estende p0_cache/ aos 208 casos de teste (só lê labels NIfTI)."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from p0c_extract import work

HERE = Path(__file__).parent
SPLIT = Path("D:/Doutorado/Coronary_Segmentation_deep_learning/data_list/split_original.csv")

if __name__ == "__main__":
    so = pd.read_csv(SPLIT)
    todo = sorted(so.loc[so.split == "test", "ID"])
    with ProcessPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(work, todo, chunksize=4))
    pd.DataFrame(rows).to_csv(HERE / "a2_test_cases.csv", index=False)
    print("ok", len(rows))
