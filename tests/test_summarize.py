import csv
import json
from collections import Counter

import pytest

from asim_harness import summarize
from asim_harness.jsonio import dumps


@pytest.fixture(scope="module")
def summary(fixtures_dir):
    return summarize.summarize_output(fixtures_dir / "output")


def _rows(fixtures_dir, table):
    with open(fixtures_dir / "output" / f"final_{table}.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_counts_match_fixture(summary, fixtures_dir):
    for table in ("households", "persons", "tours", "trips"):
        assert summary["counts"][table] == len(_rows(fixtures_dir, table))


def test_shares_are_plain_counting(summary, fixtures_dir):
    trips = _rows(fixtures_dir, "trips")
    modes = Counter(r["trip_mode"] for r in trips)
    for mode, n in modes.items():
        assert summary["trip_mode_share"]["overall"][mode] == pytest.approx(n / len(trips), abs=1e-6)
    work = [r for r in trips if r["purpose"] == "work"]
    walk = sum(1 for r in work if r["trip_mode"] == "WALK")
    assert summary["trip_mode_share"]["by_purpose"]["work"]["WALK"] == pytest.approx(walk / len(work), abs=1e-6)
    hh = _rows(fixtures_dir, "households")
    autos = Counter(r["auto_ownership"] for r in hh)
    assert summary["auto_ownership_share"] == {k: pytest.approx(v / len(hh), abs=1e-6) for k, v in autos.items()}
    assert summary["counts"]["trips_by_purpose"]["work"] == len(work)


def test_tour_frequency_counts_persons_with_zero_tours(summary, fixtures_dir):
    persons = _rows(fixtures_dir, "persons")
    tours = _rows(fixtures_dir, "tours")
    mand = Counter(r["person_id"] for r in tours if r["tour_category"] == "mandatory")
    zero = sum(1 for p in persons if mand.get(p["person_id"], 0) == 0)
    assert summary["tour_frequency"]["mandatory"]["0"] == pytest.approx(zero / len(persons), abs=1e-6)
    assert sum(summary["tour_frequency"]["non_mandatory"].values()) == pytest.approx(1.0, abs=1e-5)


def test_checks_pass_and_json_round_trip(summary):
    assert summary["checks"]["all_passed"] is True
    text = dumps(summary)
    assert json.loads(text)["counts"] == summary["counts"]
    for path in summarize.share_metric_paths(summary):
        vec = summarize.get_metric(summary, path)
        assert sum(vec.values()) == pytest.approx(1.0, abs=1e-5), path


def test_share_metric_paths_and_metric_n(summary):
    paths = summarize.share_metric_paths(summary)
    assert "trip_mode_share.overall" in paths and "auto_ownership_share" in paths
    assert "trip_mode_share.by_purpose.work" in paths
    assert "counts" not in " ".join(paths) and "checks" not in " ".join(paths)
    assert summarize.metric_n(summary, "trip_mode_share.overall") == summary["counts"]["trips"]
    assert summarize.metric_n(summary, "trip_mode_share.by_purpose.work") == summary["counts"]["trips_by_purpose"]["work"]
    assert summarize.metric_n(summary, "cdap_share") == summary["counts"]["persons"]
    with pytest.raises(KeyError):
        summarize.get_metric(summary, "trip_mode_share.by_purpose.nope")


def test_null_and_orphan_detection(tmp_path, fixtures_dir):
    import shutil
    out = tmp_path / "output"
    shutil.copytree(fixtures_dir / "output", out)
    trips = out / "final_trips.csv"
    rows = trips.read_text().splitlines()
    header = rows[0].split(",")
    first = rows[1].split(",")
    first[header.index("trip_mode")] = ""
    first[header.index("tour_id")] = "999999999"
    rows[1] = ",".join(first)
    trips.write_text("\n".join(rows) + "\n")
    s = summarize.summarize_output(out)
    assert s["checks"]["no_null_trip_modes"] is False
    assert s["checks"]["every_trip_has_tour"] is False
    assert s["checks"]["all_passed"] is False
    assert "null" in s["trip_mode_share"]["overall"]


def test_missing_table_and_column_errors(tmp_path):
    with pytest.raises(summarize.SummaryError, match="missing output table"):
        summarize.summarize_output(tmp_path)
    (tmp_path / "final_households.csv").write_text("household_id\n1\n")
    with pytest.raises(summarize.SummaryError, match="lacks expected column"):
        summarize.summarize_output(tmp_path)


def test_compact_text(summary):
    text = summarize.compact_text(summary)
    assert "trip modes:" in text and "checks: all passed" in text
