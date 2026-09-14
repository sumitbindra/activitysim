import pytest

from asim_harness import ledger
from asim_harness import manifest as mf


def _manifest(run_id, label="l", status="running", created="2026-01-01T00:00:00+00:00", **extra):
    m = {"run_id": run_id, "label": label, "status": status, "created_at": created,
         "settings_overrides": {"inherit_settings": True}, "sample_size": None}
    m.update(extra)
    return m


def test_append_upsert_and_list_order(tmp_root):
    ledger.append(ledger.row_from_manifest(_manifest("20260101-000000-aaaaaa", created="2026-01-01T00:00:00+00:00")))
    ledger.append(ledger.row_from_manifest(_manifest("20260102-000000-bbbbbb", created="2026-01-02T00:00:00+00:00")))
    rows = ledger.list_runs()
    assert [r["run_id"] for r in rows] == ["20260102-000000-bbbbbb", "20260101-000000-aaaaaa"]
    ledger.upsert(ledger.row_from_manifest(_manifest("20260101-000000-aaaaaa", status="succeeded", duration_s=3.0)))
    rows = ledger.read_rows()
    assert len(rows) == 2
    assert rows[0]["status"] == "succeeded" and rows[0]["duration_s"] == 3.0
    assert ledger.list_runs(limit=1)[0]["run_id"] == "20260102-000000-bbbbbb"


def test_row_from_manifest_picks_up_scorecard_and_error():
    row = ledger.row_from_manifest(_manifest("r", settings_overrides={"resume_after": "x", "households_sample_size": 9}),
                                   scorecard={"passed": False}, error={"failed_step": "cdap_simulate"})
    assert row["scorecard_pass"] is False
    assert row["failed_step"] == "cdap_simulate"
    assert row["resume_after"] == "x"
    assert row["sample_size"] is None  # explicit manifest value wins over the override


def test_reindex_from_manifests_and_resolve_prefix(tmp_root):
    runs = tmp_root / "runs"
    for rid, created in [("20260103-000000-cccccc", "2026-01-03T00:00:00+00:00"),
                         ("20260101-000000-aaaaaa", "2026-01-01T00:00:00+00:00")]:
        d = runs / rid
        d.mkdir()
        mf.write_manifest(d, _manifest(rid, created=created))
    (runs / "not-a-run").mkdir()
    rows = ledger.reindex()
    assert [r["run_id"] for r in rows] == ["20260101-000000-aaaaaa", "20260103-000000-cccccc"]
    assert ledger.resolve_run_id("20260103") == "20260103-000000-cccccc"
    assert ledger.resolve_run_id("20260101-000000-aaaaaa") == "20260101-000000-aaaaaa"
    with pytest.raises(KeyError, match="ambiguous"):
        ledger.resolve_run_id("2026")
    with pytest.raises(KeyError, match="no run"):
        ledger.resolve_run_id("zzz")


def test_read_rows_tolerates_torn_line(tmp_root):
    ledger.append({"run_id": "a"})
    with open(ledger.paths.ledger_path(), "a") as f:
        f.write('{"run_id": "b"')
    assert [r["run_id"] for r in ledger.read_rows()] == ["a"]
