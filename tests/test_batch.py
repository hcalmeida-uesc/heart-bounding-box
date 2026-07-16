"""Testes do runner de lote retomável genérico (project.batch.run_resumable_batch)."""

import pandas as pd

from utils.project.batch import run_resumable_batch

RECORD_COLUMNS = ["img_id", "block", "status", "value"]


def _fake_process_case_factory(calls):
    def fake_process_case(img_id, base_path, block, config):
        calls.append(img_id)
        return {"img_id": img_id, "block": block, "status": "ok", "value": img_id * 10}

    return fake_process_case


def test_resumable_batch_skips_already_ok_cases(tmp_path):
    calls = []
    output_csv = tmp_path / "block.csv"

    df1 = run_resumable_batch(
        _fake_process_case_factory(calls),
        RECORD_COLUMNS,
        "b",
        [1, 2],
        tmp_path,
        {},
        output_csv,
        resume=True,
        progress=False,
    )
    assert calls == [1, 2]
    assert set(df1["img_id"]) == {1, 2}
    assert output_csv.exists()

    df2 = run_resumable_batch(
        _fake_process_case_factory(calls),
        RECORD_COLUMNS,
        "b",
        [1, 2, 3],
        tmp_path,
        {},
        output_csv,
        resume=True,
        progress=False,
    )
    assert calls == [1, 2, 3]  # só o id novo (3) foi processado
    assert set(df2["img_id"]) == {1, 2, 3}


def test_resumable_batch_ignores_saved_progress_when_resume_false(tmp_path):
    calls = []
    output_csv = tmp_path / "block.csv"

    run_resumable_batch(
        _fake_process_case_factory(calls),
        RECORD_COLUMNS,
        "b",
        [1, 2],
        tmp_path,
        {},
        output_csv,
        resume=True,
        progress=False,
    )
    run_resumable_batch(
        _fake_process_case_factory(calls),
        RECORD_COLUMNS,
        "b",
        [1, 2],
        tmp_path,
        {},
        output_csv,
        resume=False,
        progress=False,
    )
    assert calls == [1, 2, 1, 2]  # reprocessa tudo quando resume=False


def test_resumable_batch_retries_previously_errored_cases(tmp_path):
    output_csv = tmp_path / "block.csv"
    error_row = {"img_id": 1, "block": "b", "status": "error", "value": None}
    pd.DataFrame([error_row]).reindex(columns=RECORD_COLUMNS).to_csv(output_csv, index=False)

    calls = []
    df = run_resumable_batch(
        _fake_process_case_factory(calls),
        RECORD_COLUMNS,
        "b",
        [1],
        tmp_path,
        {},
        output_csv,
        resume=True,
        progress=False,
    )
    assert calls == [1]
    assert df.loc[df["img_id"] == 1, "status"].item() == "ok"


def test_resumable_batch_persists_output_matching_record_columns(tmp_path):
    output_csv = tmp_path / "block.csv"
    run_resumable_batch(
        _fake_process_case_factory([]),
        RECORD_COLUMNS,
        "b",
        [5],
        tmp_path,
        {},
        output_csv,
        resume=True,
        progress=False,
    )
    saved = pd.read_csv(output_csv)
    assert list(saved.columns) == RECORD_COLUMNS
