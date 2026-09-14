from pathlib import Path

from asim_harness import manifest as mf


def _mk(d: Path, files: dict[str, str]) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        p = d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return d


def test_config_hash_is_stable_and_content_sensitive(tmp_path):
    base = _mk(tmp_path / "base", {"settings.yaml": "a: 1\n", "sub/spec.csv": "x,y\n"})
    over = _mk(tmp_path / "over", {"settings.yaml": "inherit_settings: true\n"})
    h1 = mf.config_hash([over, base])
    assert h1 == mf.config_hash([over, base])
    assert len(h1) == 64
    (base / "sub" / "spec.csv").write_text("x,z\n")
    assert mf.config_hash([over, base]) != h1


def test_config_hash_depends_on_which_dir_a_file_is_in(tmp_path):
    base = _mk(tmp_path / "base", {"a.csv": "1\n"})
    over = _mk(tmp_path / "over", {"b.csv": "2\n"})
    swapped_base = _mk(tmp_path / "base2", {"b.csv": "2\n"})
    swapped_over = _mk(tmp_path / "over2", {"a.csv": "1\n"})
    assert mf.config_hash([over, base]) != mf.config_hash([swapped_over, swapped_base])


def test_config_hash_skips_missing_dirs(tmp_path):
    base = _mk(tmp_path / "base", {"a.csv": "1\n"})
    assert mf.config_hash([tmp_path / "nope", base]) == mf.config_hash([tmp_path / "nope2", base])


def test_data_fingerprint_fields(tmp_path):
    d = _mk(tmp_path / "data", {"households.csv": "id\n1\n", "nested/skims.omx": "zzz"})
    rows = mf.data_fingerprint(d)
    assert [r["name"] for r in rows] == ["households.csv", "nested/skims.omx"]
    assert rows[0]["size"] == 5 and rows[1]["size"] == 3
    assert rows[0]["mtime"].endswith("+00:00")
    assert mf.data_fingerprint(tmp_path / "missing") == []


def test_parse_step_timings(tmp_path):
    log = tmp_path / "timing_log.csv"
    log.write_text(
        "process_name,model_name,seconds,minutes,notes\n"
        "MainProcess,initialize_landuse,1.1,0.0,\n"
        "MainProcess,trip_mode_choice,6.7,0.1,\n"
        "MainProcess,,3,0,\n"
    )
    assert mf.parse_step_timings(log) == {"initialize_landuse": 1.1, "trip_mode_choice": 6.7}
    assert mf.parse_step_timings(tmp_path / "absent.csv") == {}


def test_manifest_round_trip_and_summary(tmp_path):
    m = {
        "run_id": "20260101-000000-abc123", "label": "x", "status": "succeeded", "exit_code": 0,
        "sample_size": 500, "settings_overrides": {"inherit_settings": True, "resume_after": "trip_scheduling"},
        "step_timings": {"a": 1.0, "b": 2.0}, "created_at": "t0", "finished_at": "t1", "duration_s": 3.0,
    }
    mf.write_manifest(tmp_path, m)
    assert mf.read_manifest(tmp_path) == m
    s = mf.summary_of(m)
    assert s["resume_after"] == "trip_scheduling"
    assert s["steps_run"] == 2
    assert s["sample_size"] == 500


def test_versions_available():
    assert mf.activitysim_version()
    assert mf.python_version().startswith("3.")
