import shutil
from pathlib import Path

import pytest

from asim_harness import errors, ledger
from asim_harness import manifest as mf


def _run_dir(tmp_root, name, log_fixture=None, stdout_fixture=None, stderr_text=None, log_in_subdir=True):
    d = tmp_root / "runs" / name
    (d / "output" / "log").mkdir(parents=True)
    if log_fixture is not None:
        target = d / "output" / ("log/activitysim.log" if log_in_subdir else "activitysim.log")
        shutil.copy(log_fixture, target)
    if stdout_fixture is not None:
        shutil.copy(stdout_fixture, d / "stdout.log")
    if stderr_text is not None:
        (d / "stderr.log").write_text(stderr_text)
    return d


def test_missing_coefficients_file(tmp_root, fixtures_dir):
    d = _run_dir(tmp_root, "r1", fixtures_dir / "logs" / "missing_coefficients_file.log")
    rec = errors.extract(d, exit_code=99)
    assert rec["failed_step"] == "trip_mode_choice"
    assert rec["exception_type"] == "FileNotFoundError"
    assert "trip_mode_choice_coefficients_DOES_NOT_EXIST.csv" in rec["message"]
    assert rec["expression_context"]["missing_file"] == "trip_mode_choice_coefficients_DOES_NOT_EXIST.csv"
    assert rec["traceback_source"] == "log"
    assert rec["traceback_tail"][-1].startswith("FileNotFoundError:")
    assert any("read_model_coefficients" in ln for ln in rec["traceback_tail"])
    assert not any(set(ln.strip()) <= set("^~ ") for ln in rec["traceback_tail"])  # no caret lines
    assert len(rec["log_tail"]) <= errors.LOG_TAIL_LINES
    assert rec["log_path"].endswith("log/activitysim.log")
    assert rec["exit_code"] == 99


def test_bad_preprocessor_expression(tmp_root, fixtures_dir):
    d = _run_dir(tmp_root, "r2", fixtures_dir / "logs" / "bad_preprocessor_expression.log")
    rec = errors.extract(d)
    assert rec["failed_step"] == "trip_mode_choice"
    assert rec["exception_type"] == "NameError"
    assert rec["message"] == "name 'no_such_column_xyz' is not defined"
    ctx = rec["expression_context"]
    assert ctx["expression"] == "no_such_column_xyz + 1"
    assert ctx["trace_label"] == "assign_variables"
    text = errors.text(rec, log_tail_lines=3)
    assert "expression: no_such_column_xyz + 1" in text and "log tail" in text


def test_stdout_traceback_fallback_and_log_in_output_root(tmp_root, fixtures_dir):
    """A log without a traceback (e.g. truncated) still yields the exception from stdout."""
    log = fixtures_dir / "logs" / "missing_coefficients_file.log"
    lines = [ln for ln in log.read_text().splitlines() if "Traceback" not in ln and not ln.startswith(" ")]
    lines = [ln for ln in lines if not ln.startswith("FileNotFoundError")]
    d = _run_dir(tmp_root, "r3", stdout_fixture=fixtures_dir / "logs" / "missing_coefficients_file.stdout")
    (d / "output" / "activitysim.log").write_text("\n".join(lines) + "\n")
    (d / "output" / "log").rmdir()
    rec = errors.extract(d, exit_code=99)
    assert rec["failed_step"] == "trip_mode_choice"
    assert rec["traceback_source"] == "stdout"
    assert rec["exception_type"] == "FileNotFoundError"
    assert rec["log_path"].endswith("output/activitysim.log")


def test_no_log_at_all_uses_stderr(tmp_root):
    d = _run_dir(tmp_root, "r4", stderr_text="Traceback (most recent call last):\n  File \"x.py\", line 1, in <module>\n    boom()\nRuntimeError: boom\n")
    rec = errors.extract(d, exit_code=1)
    assert rec["failed_step"] is None
    assert rec["exception_type"] == "RuntimeError" and rec["message"] == "boom"
    assert rec["traceback_source"] == "stderr"
    assert rec["log_path"] is None and rec["log_tail"] == []


def test_step_marker_fallback_without_error_block(tmp_root):
    d = _run_dir(tmp_root, "r5")
    (d / "output" / "log" / "activitysim.log").write_text(
        "14/09/2026 18:25:20 - INFO - activitysim.core.workflow.runner - #run_model running step cdap_simulate\n"
        "14/09/2026 18:25:21 - INFO - activitysim.core.workflow.runner - #run_model running step mandatory_tour_frequency\n"
    )
    rec = errors.extract(d)
    assert rec["failed_step"] == "mandatory_tour_frequency"
    assert rec["exception_type"] is None and rec["traceback_tail"] == []


def test_parse_helpers():
    assert errors.parse_exception_line("pandas.errors.ParserError: Error tokenizing data") == ("pandas.errors.ParserError", "Error tokenizing data")
    assert errors.parse_exception_line("KeyboardInterrupt") == ("KeyboardInterrupt", "")
    assert errors.parse_exception_line("14/09/2026 18:25:20 - ERROR - root - ValueError: bad") == ("ValueError", "bad")
    ctx = errors.expression_context([
        "14/09/2026 18:25:30 - ERROR - activitysim.core.assign - Error reading spec file: /x/configs/spec.csv",
        "14/09/2026 18:25:30 - ERROR - activitysim.core.simulate - Variable evaluation failed KeyError ('foo') evaluating: df.foo * 2",
    ])
    assert ctx == {"spec_file": "/x/configs/spec.csv", "expression": "df.foo * 2", "exception_type": "KeyError", "message": "'foo'"}
    assert errors.expression_context(["nothing here"]) is None


def test_finalize_hook_and_error_for_run(tmp_root, fixtures_dir):
    d = _run_dir(tmp_root, "20260101-000000-abcdef", fixtures_dir / "logs" / "bad_preprocessor_expression.log")
    manifest = {"run_id": "20260101-000000-abcdef", "label": "x", "status": "failed", "exit_code": 99,
                "created_at": "t", "settings_overrides": {"inherit_settings": True},
                "override_files": ["trip_mode_choice_annotate_trips_preprocessor.csv"]}
    mf.write_manifest(d, manifest)
    errors.finalize_hook(d, manifest)
    rec = errors.load_error("20260101")
    assert rec["failed_step"] == "trip_mode_choice" and rec["override_files"] == ["trip_mode_choice_annotate_trips_preprocessor.csv"]
    assert ledger.reindex()[0]["failed_step"] == "trip_mode_choice"
    compact = errors.compact(rec)
    assert "log_tail" not in compact and compact["exception_type"] == "NameError"
    assert errors.compact(rec, log_tail_lines=5)["log_tail"] == rec["log_tail"][-5:]
    # succeeded runs have no error record
    manifest["status"] = "succeeded"
    mf.write_manifest(d, manifest)
    (d / "error.json").unlink()
    assert errors.error_for_run("20260101") is None
    errors.finalize_hook(d, manifest)
    assert not (d / "error.json").exists()
