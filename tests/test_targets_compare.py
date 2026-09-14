import pytest
import yaml

from asim_harness import compare, ledger, summarize, targets
from asim_harness import manifest as mf
from asim_harness.jsonio import write_json


def _summary(run_id="r1", walk=0.6, bike=0.1, autos=(0.5, 0.4, 0.1)):
    other = round(1 - walk - bike, 6)
    return {
        "run_id": run_id,
        "counts": {"households": 100, "persons": 200, "tours": 300, "trips": 1000,
                   "tours_by_purpose": {"work": 200}, "trips_by_purpose": {"work": 700, "home": 300}},
        "auto_ownership_share": {"0": autos[0], "1": autos[1], "2": autos[2]},
        "cdap_share": {"H": 0.2, "M": 0.5, "N": 0.3},
        "tour_mode_share": {"overall": {"WALK": walk, "BIKE": bike, "DRIVEALONEFREE": other},
                            "by_purpose": {"work": {"WALK": walk, "BIKE": bike, "DRIVEALONEFREE": other}}},
        "trip_mode_share": {"overall": {"WALK": walk, "BIKE": bike, "DRIVEALONEFREE": other},
                            "by_purpose": {"work": {"WALK": walk, "BIKE": bike, "DRIVEALONEFREE": other},
                                           "home": {"WALK": 1.0}}},
        "tour_frequency": {"mandatory": {"0": 0.5, "1": 0.5}, "non_mandatory": {"0": 0.7, "1": 0.3}},
        "checks": {"all_passed": True},
    }


def _make_run(tmp_root, run_id, summary):
    d = tmp_root / "runs" / run_id
    d.mkdir(parents=True)
    mf.write_manifest(d, {"run_id": run_id, "label": run_id, "status": "succeeded", "created_at": run_id,
                          "settings_overrides": {"inherit_settings": True}, "sample_size": None})
    write_json(d / "summary.json", summary)
    return d


def test_bootstrap_then_check_is_full_pass(tmp_root):
    _make_run(tmp_root, "20260101-000000-aaaaaa", _summary("20260101-000000-aaaaaa"))
    doc = targets.bootstrap("20260101")
    path = tmp_root / "targets" / "prototype_mtc.yaml"
    assert path.exists() and doc["source_run"] == "20260101-000000-aaaaaa"
    loaded = yaml.safe_load(path.read_text())
    assert loaded["metrics"]["trip_mode_share.overall"]["tolerance_abs"] == 0.02
    assert loaded["metrics"]["auto_ownership_share"]["tolerance_abs"] == 0.03
    assert loaded["metrics"]["cdap_share"]["tolerance_abs"] == 0.03
    assert loaded["metrics"]["trip_mode_share.by_purpose.work"]["n"] == 700
    card = targets.check_run("20260101")
    assert card["passed"] is True and card["n_failed"] == 0
    assert (tmp_root / "runs" / "20260101-000000-aaaaaa" / "scorecard.json").exists()
    assert ledger.reindex()[0]["scorecard_pass"] is True
    with pytest.raises(targets.TargetsError, match="already exists"):
        targets.bootstrap("20260101")
    targets.bootstrap("20260101", force=True)


def test_score_partial_fail_with_deltas(tmp_root):
    base = _summary("base")
    _make_run(tmp_root, "20260101-000000-aaaaaa", base)
    targets.bootstrap("20260101")
    drifted = _summary("20260102-000000-bbbbbb", walk=0.65, bike=0.05, autos=(0.52, 0.39, 0.09))
    _make_run(tmp_root, "20260102-000000-bbbbbb", drifted)
    card = targets.check_run("20260102")
    assert card["passed"] is False
    m = card["metrics"]
    assert m["trip_mode_share.overall"]["passed"] is False
    assert m["trip_mode_share.overall"]["deltas"] == {"BIKE": -0.05, "DRIVEALONEFREE": 0.0, "WALK": 0.05}
    assert m["trip_mode_share.overall"]["max_abs_delta"] == 0.05
    assert m["trip_mode_share.overall"]["worst_category"] in ("WALK", "BIKE")
    assert m["auto_ownership_share"]["passed"] is True  # 0.02 < 0.03 tolerance
    assert m["trip_mode_share.by_purpose.home"]["passed"] is True
    assert m["cdap_share"]["passed"] is True
    assert set(card["failed_metrics"]) == {"trip_mode_share.overall", "trip_mode_share.by_purpose.work",
                                           "tour_mode_share.overall", "tour_mode_share.by_purpose.work"}
    compact = targets.compact(card)
    assert compact["n_failed"] == 4 and compact["failed"][0]["metric"] in card["failed_metrics"]
    text = targets.text(card)
    assert "FAIL" in text and "trip_mode_share.overall" in text


def test_score_missing_metric_and_new_category():
    base = _summary()
    t = {"model": "m", "source_run": "x", "_path": "p", "metrics": {
        "trip_mode_share.by_purpose.univ": {"tolerance_abs": 0.02, "targets": {"WALK": 1.0}},
        "trip_mode_share.overall": {"tolerance_abs": 0.02, "targets": {"WALK": 0.6, "BIKE": 0.1, "DRIVEALONEFREE": 0.29, "TAXI": 0.01}},
    }}
    card = targets.score(base, t)
    assert card["metrics"]["trip_mode_share.by_purpose.univ"]["status"] == "missing"
    assert card["metrics"]["trip_mode_share.overall"]["deltas"]["TAXI"] == -0.01
    assert card["metrics"]["trip_mode_share.overall"]["deltas"]["DRIVEALONEFREE"] == pytest.approx(0.01)
    assert card["metrics"]["trip_mode_share.overall"]["passed"] is True
    assert card["passed"] is False and card["n_failed"] == 1


def test_load_targets_errors(tmp_root):
    with pytest.raises(targets.TargetsError, match="bootstrap"):
        targets.load_targets()
    p = tmp_root / "targets" / "prototype_mtc.yaml"
    p.parent.mkdir()
    p.write_text("model: x\n")
    with pytest.raises(targets.TargetsError, match="metrics"):
        targets.load_targets()


def test_compare_verdicts(tmp_root):
    a = _summary("a")
    b = _summary("b", walk=0.603, bike=0.097)
    b["trip_mode_share"]["by_purpose"]["univ"] = {"WALK": 1.0}
    del b["tour_frequency"]["non_mandatory"]
    result = compare.compare_summaries(a, b)
    m = result["metrics"]
    assert m["trip_mode_share.overall"]["verdict"] == "unchanged"
    assert m["trip_mode_share.by_purpose.univ"]["verdict"] == "missing" and m["trip_mode_share.by_purpose.univ"]["missing_in"] == "a"
    assert m["tour_frequency.non_mandatory"]["verdict"] == "missing" and m["tour_frequency.non_mandatory"]["missing_in"] == "b"
    c = _summary("c", walk=0.7, bike=0.0)
    result = compare.compare_summaries(a, c)
    assert result["metrics"]["trip_mode_share.overall"]["verdict"] == "drifted"
    assert result["metrics"]["trip_mode_share.overall"]["deltas"]["WALK"] == pytest.approx(0.1)
    assert result["n_drifted"] == 4 and result["counts"]["trips"] == {"a": 1000, "b": 1000}
    compact = compare.compact(result)
    assert compact["drifted"][0]["metric"] in result["metrics"]
    assert "drifted" in compare.text(result)
    _make_run(tmp_root, "20260101-000000-aaaaaa", a)
    _make_run(tmp_root, "20260103-000000-cccccc", c)
    assert compare.compare_runs("20260101", "20260103")["run_b"] == "c"


def test_check_reads_only_summary(tmp_root, monkeypatch):
    """check must never touch the raw tables: summarize_output is not called."""
    _make_run(tmp_root, "20260101-000000-aaaaaa", _summary("20260101-000000-aaaaaa"))
    targets.bootstrap("20260101")
    monkeypatch.setattr(summarize, "summarize_output", lambda *a, **k: (_ for _ in ()).throw(AssertionError("raw tables read")))
    assert targets.check_run("20260101")["passed"] is True


def test_scorecard_keeps_shares_and_targets_and_compact_detail(tmp_root):
    _make_run(tmp_root, "20260101-000000-aaaaaa", _summary("20260101-000000-aaaaaa"))
    targets.bootstrap("20260101")
    drifted = _summary("20260102-000000-bbbbbb", walk=0.65, bike=0.05)
    _make_run(tmp_root, "20260102-000000-bbbbbb", drifted)
    card = targets.check_run("20260102")
    m = card["metrics"]["trip_mode_share.overall"]
    assert m["actual"]["WALK"] == 0.65 and m["target"]["WALK"] == 0.6 and m["deltas"]["WALK"] == 0.05
    default = targets.compact(card)
    assert default["detail"] == "failed"
    failed = {r["metric"]: r for r in default["failed"]}
    assert failed["trip_mode_share.overall"]["categories"]["WALK"] == {"delta": 0.05, "share": 0.65, "target": 0.6}
    assert "DRIVEALONEFREE" not in failed["trip_mode_share.overall"]["categories"]  # |delta| < 0.0005 dropped
    assert all("categories" not in r for r in default["passed_metrics"])
    everything = targets.compact(card, detail="all")
    assert everything["detail"] == "all"
    assert all("categories" in r for r in everything["passed_metrics"] + everything["failed"])
    one = targets.compact(card, metric="trip_mode_share.overall")
    assert [r["metric"] for r in one["failed"] + one["passed_metrics"]] == ["trip_mode_share.overall"]
    assert set(one["failed"][0]["categories"]) == {"WALK", "BIKE", "DRIVEALONEFREE"}
    prefix = targets.compact(card, metric="tour_frequency")
    assert {r["metric"] for r in prefix["passed_metrics"]} == {"tour_frequency.mandatory", "tour_frequency.non_mandatory"}
    with pytest.raises(KeyError, match="no metric"):
        targets.compact(card, metric="nope")
    with pytest.raises(ValueError):
        targets.compact(card, detail="some")
    # old scorecards without actual/target still render
    old = {k: v for k, v in card.items()}
    old["metrics"] = {n: {k: v for k, v in r.items() if k not in ("actual", "target")} for n, r in card["metrics"].items()}
    assert targets.compact(old, detail="all")["failed"][0]["categories"]["WALK"] == {"delta": 0.05}
