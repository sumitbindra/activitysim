"""One real run of the example at the smoke sample size (slow: about 1.5 minutes).

Run with ``pytest -m slow``. Uses the real example/ but a temporary runs directory
and a temporary targets file, so nothing in the repo's runs/ or targets/ changes.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from asim_harness import errors, example, ledger, paths, runner, summarize, targets

SMOKE_SAMPLE_SIZE = 500

pytestmark = pytest.mark.slow


@pytest.fixture
def real_example_tmp_runs(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("ASIM_HARNESS_ROOT", str(root))
    monkeypatch.setenv("ASIM_EXAMPLE_DIR", str(root / "example"))
    monkeypatch.setenv("ASIM_RUNS_DIR", str(tmp_path / "runs"))
    try:
        example.check_example()
    except example.ExampleMissing as e:
        pytest.skip(str(e))
    return tmp_path


def test_smoke_run_produces_summary_and_scorecard(real_example_tmp_runs):
    tmp = real_example_tmp_runs
    m = runner.run("integration smoke", sample_size=SMOKE_SAMPLE_SIZE)
    assert m["status"] == "succeeded", runner.failure_explanation(paths.run_dir(m["run_id"]), m)
    run_dir = paths.run_dir(m["run_id"])
    assert (run_dir / "output" / "log" / "activitysim.log").is_file()
    assert m["step_timings"] and "trip_mode_choice" in m["step_timings"]
    assert m["settings_overrides"] == {"inherit_settings": True, "households_sample_size": SMOKE_SAMPLE_SIZE}

    summary = summarize.load_summary(m["run_id"])
    assert summary["counts"]["households"] == SMOKE_SAMPLE_SIZE
    assert summary["checks"]["all_passed"] is True

    # score against a targets file bootstrapped from this very run -> full pass; then against the
    # repo's real targets (if present) -> a scorecard exists either way
    targets_file = tmp / "targets.yaml"
    targets.bootstrap(m["run_id"], path=targets_file)
    card = targets.check_run(m["run_id"], targets_file)
    assert card["passed"] is True and card["n_metrics"] >= 5
    assert (run_dir / "scorecard.json").is_file()
    rows = ledger.list_runs()
    assert rows[0]["run_id"] == m["run_id"] and rows[0]["scorecard_pass"] is True
    assert errors.error_for_run(m["run_id"]) is None


def test_failure_fixture_yields_error_json(real_example_tmp_runs):
    """A broken trip mode choice preprocessor override fails with an expression context.

    In a full run the failure surfaces in trip_destination, which evaluates the trip
    mode choice preprocessor while computing its mode choice logsums; only a resume
    after trip_scheduling fails in trip_mode_choice itself (see NOTES.md, Phase 5).
    """
    fixture = Path(__file__).parent / "fixtures" / "failures" / "bad_preprocessor_expression" / "trip_mode_choice_annotate_trips_preprocessor.csv"
    models = "initialize_landuse,initialize_households,compute_accessibility,school_location,workplace_location," \
             "auto_ownership_simulate,free_parking,cdap_simulate,mandatory_tour_frequency,mandatory_tour_scheduling," \
             "joint_tour_frequency,joint_tour_composition,joint_tour_participation,joint_tour_destination," \
             "joint_tour_scheduling,non_mandatory_tour_frequency,non_mandatory_tour_destination," \
             "non_mandatory_tour_scheduling,tour_mode_choice_simulate,atwork_subtour_frequency," \
             "atwork_subtour_destination,atwork_subtour_scheduling,atwork_subtour_mode_choice,stop_frequency," \
             "trip_purpose,trip_destination,trip_purpose_and_destination,trip_scheduling,trip_mode_choice"
    m = runner.run("integration failure", sample_size=50, models=models, override_files=[fixture])
    assert m["status"] == "failed" and m["exit_code"] != 0
    record = errors.error_for_run(m["run_id"])
    assert record["failed_step"] == "trip_destination"
    assert record["exception_type"] == "NameError"
    assert record["expression_context"]["expression"] == "no_such_column_xyz + 1"
    assert record["override_files"] == [fixture.name]
    assert "failed in step trip_destination" in runner.failure_explanation(paths.run_dir(m["run_id"]), m)
