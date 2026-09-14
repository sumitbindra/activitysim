import re
from pathlib import Path

import pytest
import yaml

from asim_harness import paths, runner


def test_new_run_id_format():
    rid = runner.new_run_id()
    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{6}", rid)


def test_parse_models():
    assert runner.parse_models("a, b,,c ") == ["a", "b", "c"]
    assert runner.parse_models(["a", " b "]) == ["a", "b"]
    assert runner.parse_models("") is None
    assert runner.parse_models(None) is None


def test_build_settings_overrides_only_has_given_keys():
    assert runner.build_settings_overrides() == {"inherit_settings": True}
    ov = runner.build_settings_overrides(sample_size=500, resume_after="trip_scheduling",
                                         trace_hh_id=7, models=["a"], extra={"chunk_size": 0})
    assert ov == {
        "inherit_settings": True, "households_sample_size": 500, "resume_after": "trip_scheduling",
        "trace_hh_id": 7, "models": ["a"], "chunk_size": 0,
    }


def test_write_override_dir(tmp_root, tmp_path):
    run_dir = tmp_root / "runs" / "r1"
    shadow = tmp_path / "trip_mode_choice_coefficients.csv"
    shadow.write_text("coefficient_name,value\ncoef_x,2.0\n")
    cfg, names = runner.write_override_dir(run_dir, {"inherit_settings": True, "households_sample_size": 5}, [shadow])
    assert cfg == run_dir / "overrides" / "configs"
    assert names == ["trip_mode_choice_coefficients.csv"]
    assert yaml.safe_load((cfg / "settings.yaml").read_text()) == {"inherit_settings": True, "households_sample_size": 5}
    assert (cfg / "trip_mode_choice_coefficients.csv").read_text() == shadow.read_text()


def test_write_override_dir_rejects_settings_and_missing(tmp_root, tmp_path):
    run_dir = tmp_root / "runs" / "r2"
    with pytest.raises(runner.HarnessError):
        runner.write_override_dir(run_dir, {"inherit_settings": True}, [tmp_path / "absent.csv"])
    bad = tmp_path / "settings.yaml"
    bad.write_text("x: 1\n")
    with pytest.raises(runner.HarnessError):
        runner.write_override_dir(run_dir, {"inherit_settings": True}, [bad])


def test_build_command_puts_override_dir_first(tmp_root):
    cmd = runner.build_command(Path("/o"), Path("/out"))
    assert cmd[1:4] == ["-m", "activitysim", "run"]
    assert cmd[cmd.index("-c") + 1] == "/o"
    assert cmd[cmd.index("-c") + 3] == str(paths.configs_dir())
    assert cmd[cmd.index("-o") + 1] == "/out"
    assert cmd[cmd.index("-d") + 1] == str(paths.data_dir())


def test_run_argument_validation_does_not_launch(tmp_root):
    with pytest.raises(runner.HarnessError, match="label"):
        runner.run("   ")
    with pytest.raises(runner.HarnessError, match="resume-from"):
        runner.run("x", resume_after="trip_scheduling")
    with pytest.raises(runner.HarnessError, match="resume-after"):
        runner.run("x", resume_from="20260101-000000-abc123")
    with pytest.raises(runner.HarnessError, match="no run matches"):
        runner.run("x", resume_from="nope", resume_after="_")
    with pytest.raises(runner.HarnessError, match="sample-size"):
        runner.run("x", sample_size=-1)
    assert list((tmp_root / "runs").iterdir()) == []


def test_run_requires_example(tmp_root, monkeypatch):
    monkeypatch.setenv("ASIM_EXAMPLE_DIR", str(tmp_root / "nowhere"))
    with pytest.raises(runner.HarnessError, match="asim init"):
        runner.run("x")


def test_failure_explanation_without_error_json(tmp_root):
    run_dir = tmp_root / "runs" / "r3"
    run_dir.mkdir(parents=True)
    (run_dir / "stderr.log").write_text("Traceback...\nValueError: boom\n")
    text = runner.failure_explanation(run_dir, {"run_id": "r3", "label": "l", "exit_code": 1})
    assert "ValueError: boom" in text and "exit code 1" in text
