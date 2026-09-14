"""Shared fixtures: every test works in a temp harness root, never in the real example/ or runs/."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

SMALL_SETTINGS = {
    "households_sample_size": 100000,
    "resume_after": None,
    "models": ["initialize_landuse", "initialize_households", "trip_mode_choice", "write_tables"],
    "output_tables": {"h5_store": False, "action": "include", "prefix": "final_", "tables": ["trips"]},
}


@pytest.fixture
def tmp_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway harness root with a tiny fake example and an empty runs dir."""
    example = tmp_path / "example"
    (example / "configs").mkdir(parents=True)
    (example / "data").mkdir()
    (example / "configs" / "settings.yaml").write_text(yaml.safe_dump(SMALL_SETTINGS))
    (example / "configs" / "trip_mode_choice.yaml").write_text("SPEC: trip_mode_choice.csv\n")
    (example / "configs" / "trip_mode_choice_coefficients.csv").write_text("coefficient_name,value\ncoef_x,1.0\n")
    (example / "data" / "households.csv").write_text("household_id\n1\n")
    (tmp_path / "runs").mkdir()
    monkeypatch.setenv("ASIM_HARNESS_ROOT", str(tmp_path))
    monkeypatch.setenv("ASIM_EXAMPLE_DIR", str(example))
    monkeypatch.setenv("ASIM_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"
