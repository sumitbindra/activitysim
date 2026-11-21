"""
Exact-match test against a saved ActivitySim 1.x run. Skipped unless the
baseline exists (create it with harness/run_asim1.py).
"""

from pathlib import Path

import pytest

from asim_core.asim1 import Asim1State
from asim_core.compare import compare_column
from asim_core.components import auto_ownership
from asim_core.runner import Runner

REPO = Path(__file__).resolve().parents[2]
BASELINE = REPO / "asim_core/work/asim1_legacy/pipeline.parquetpipeline"
CONFIGS = [REPO / "activitysim/examples/prototype_mtc/configs"]


@pytest.mark.skipif(not BASELINE.exists(), reason="no ActivitySim 1.x baseline run")
def test_accessibility_matches_1x_exactly(tmp_path):
    pytest.importorskip("tables")
    from asim_core.components import compute_accessibility
    from asim_core.skims import import_omx

    skims = import_omx(REPO / "activitysim/examples/prototype_mtc/data/skims.omx", tmp_path / "skims")
    st = Asim1State(BASELINE)
    comp = compute_accessibility(CONFIGS, origins_per_batch=10)
    store = st.load_store(st.previous_step(comp.name))
    Runner([comp], tmp_path / "run", skims=skims).run([comp.name], store)
    ref = st.read("accessibility", comp.name)
    for col in comp.provides["accessibility"]:
        result = compare_column(comp.name, store["accessibility"], col, ref[col])
        assert result.status == "exact", result


@pytest.mark.skipif(not BASELINE.exists(), reason="no ActivitySim 1.x baseline run")
def test_auto_ownership_matches_1x_exactly(tmp_path):
    st = Asim1State(BASELINE)
    comp = auto_ownership(CONFIGS)
    store = st.load_store(st.previous_step(comp.name))
    Runner([comp], tmp_path).run([comp.name], store)
    ref = st.read("households", comp.name)["auto_ownership"]
    result = compare_column(comp.name, store["households"], "auto_ownership", ref)
    assert result.status == "exact", result


@pytest.mark.skipif(not BASELINE.exists(), reason="no ActivitySim 1.x baseline run")
def test_tour_mode_choice_matches_1x_exactly(tmp_path):
    pytest.importorskip("tables")
    from asim_core.components import tour_mode_choice
    from asim_core.skims import import_omx

    skims = import_omx(REPO / "activitysim/examples/prototype_mtc/data/skims.omx", tmp_path / "skims")
    st = Asim1State(BASELINE)
    comp = tour_mode_choice(CONFIGS)
    store = st.load_store(st.previous_step(comp.name))
    Runner([comp], tmp_path / "run", skims=skims).run([comp.name], store)
    ref = st.read("tours", comp.name)
    for col in ("tour_mode", "mode_choice_logsum"):  # logsums bit for bit
        result = compare_column(comp.name, store["tours"], col, ref[col])
        assert result.status == "exact", result


@pytest.mark.skipif(not BASELINE.exists(), reason="no ActivitySim 1.x baseline run")
def test_non_mandatory_tour_scheduling_matches_1x_exactly(tmp_path):
    from asim_core.components import non_mandatory_tour_scheduling

    st = Asim1State(BASELINE)
    # small batches: the results must not depend on the batch size
    comp = non_mandatory_tour_scheduling(CONFIGS, max_rows=20_000)
    store = st.load_store(st.previous_step(comp.name))
    Runner([comp], tmp_path).run([comp.name], store)
    tours = st.read("tours", comp.name)
    for col in ("tdd", "start", "end", "duration"):
        result = compare_column(comp.name, store["tours"], col, tours[col])
        assert result.status == "exact", result
    windows = st.read("person_windows", comp.name)
    for col in windows.columns:
        assert compare_column(comp.name, store["person_windows"], col, windows[col]).status == "exact"


@pytest.mark.skipif(not BASELINE.exists(), reason="no ActivitySim 1.x baseline run")
@pytest.mark.parametrize("seams", [{}, {"logsum_keys": "distinct_inputs"}])
def test_mandatory_tour_scheduling_matches_1x_exactly(tmp_path, seams):
    pytest.importorskip("tables")
    from asim_core.components import mandatory_tour_scheduling
    from asim_core.seams import Engine
    from asim_core.skims import import_omx

    skims = import_omx(REPO / "activitysim/examples/prototype_mtc/data/skims.omx", tmp_path / "skims")
    st = Asim1State(BASELINE)
    comp = mandatory_tour_scheduling(CONFIGS, max_rows=50_000)  # small batches: same results
    store = st.load_store(st.previous_step(comp.name))
    Runner([comp], tmp_path / "run", skims=skims, engine=Engine(seams)).run([comp.name], store)
    tours = st.read("tours", comp.name)
    for col in ("tdd", "start", "end", "duration"):
        result = compare_column(comp.name, store["tours"], col, tours[col])
        assert result.status == "exact", result
    windows = st.read("person_windows", comp.name)
    for col in windows.columns:
        assert compare_column(comp.name, store["person_windows"], col, windows[col]).status == "exact"


@pytest.mark.skipif(not BASELINE.exists(), reason="no ActivitySim 1.x baseline run")
def test_non_mandatory_tour_destination_matches_1x_exactly(tmp_path):
    pytest.importorskip("tables")
    from asim_core.components import non_mandatory_tour_destination
    from asim_core.skims import import_omx

    skims = import_omx(REPO / "activitysim/examples/prototype_mtc/data/skims.omx", tmp_path / "skims")
    st = Asim1State(BASELINE)
    comp = non_mandatory_tour_destination(CONFIGS, max_rows=5_000)  # small batches: same results
    store = st.load_store(st.previous_step(comp.name))
    Runner([comp], tmp_path / "run", skims=skims).run([comp.name], store)
    tours = st.read("tours", comp.name)
    for col in ("destination", "destination_logsum"):  # logsums bit for bit
        result = compare_column(comp.name, store["tours"], col, tours[col])
        assert result.status == "exact", result


@pytest.mark.skipif(not BASELINE.exists(), reason="no ActivitySim 1.x baseline run")
@pytest.mark.parametrize("step", ["school_location", "workplace_location"])
def test_location_choice_matches_1x_exactly(step, tmp_path):
    """Choices, logsums, annotations (persons and households) and the modeled size table."""
    pytest.importorskip("tables")
    from asim_core.components import school_location, workplace_location
    from asim_core.skims import import_omx

    skims = import_omx(REPO / "activitysim/examples/prototype_mtc/data/skims.omx", tmp_path / "skims")
    st = Asim1State(BASELINE)
    make = school_location if step == "school_location" else workplace_location
    comp = make(CONFIGS, max_rows=3_000)  # small batches: same results
    store = st.load_store(st.previous_step(step))
    Runner([comp], tmp_path / "run", skims=skims).run([step], store)
    n = 0
    for table, cols in comp.provides.items():
        ref = st.read(table, step)
        for col in cols:
            result = compare_column(step, store[table], col, ref[col])
            assert result.status == "exact", result
            n += 1
    assert n >= 7


@pytest.mark.skipif(not BASELINE.exists(), reason="no ActivitySim 1.x baseline run")
@pytest.mark.parametrize("step", ["school_location", "non_mandatory_tour_destination"])
def test_choosers_without_a_segment_are_reported(step, tmp_path):
    """1.x leaves choosers whose segment value has no segment without a choice, silently;
    asim_core does the same and reports it as a warning of the step."""
    pytest.importorskip("tables")
    import json

    import numpy as np

    from asim_core.components import non_mandatory_tour_destination, school_location
    from asim_core.skims import import_omx

    skims = import_omx(REPO / "activitysim/examples/prototype_mtc/data/skims.omx", tmp_path / "skims")
    st = Asim1State(BASELINE)
    store = st.load_store(st.previous_step(step))
    if step == "school_location":
        comp = school_location(CONFIGS)
        persons = store["persons"]
        seg = np.asarray(persons["school_segment"]).copy()
        row = np.flatnonzero(np.asarray(persons["is_student"]).astype(bool))[0]
        seg[row] = 9
        persons.set_column("school_segment", seg)
        expect, table, column = "school_segment", "persons", "school_zone_id"
    else:
        comp = non_mandatory_tour_destination(CONFIGS)
        tours = store["tours"]
        kind = np.asarray(tours["tour_type"]).astype(object).copy()
        row = np.flatnonzero(np.asarray(tours["tour_category"]) == "non_mandatory")[0]
        kind[row] = "unknown"
        tours.set_column("tour_type", kind)
        expect, table, column = "'unknown': 1", "tours", "destination"
    report = Runner([comp], tmp_path / "run", skims=skims).run([step], store)
    assert len(report.warnings) == 1 and expect in report.warnings[0]["message"], report.warnings
    assert report.warnings[0]["step"] == step
    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    assert manifest["warnings"] == report.warnings
    value = np.asarray(store[table][column])[row]
    assert value == -1 or np.isnan(value)  # no choice, as in 1.x


@pytest.mark.skipif(not BASELINE.exists(), reason="no ActivitySim 1.x baseline run")
@pytest.mark.parametrize("step", ["initialize_landuse", "initialize_households", "free_parking", "cdap_simulate",
                                  "mandatory_tour_frequency"])
def test_block1_steps_match_1x_exactly(step, tmp_path):
    from asim_core import components

    st = Asim1State(BASELINE)
    data = [REPO / "activitysim/examples/prototype_mtc/data"]
    make = getattr(components, step)
    comp = make(CONFIGS, data) if step.startswith("initialize") else make(CONFIGS)
    store = st.load_store(st.previous_step(comp.name))
    Runner([comp], tmp_path / "run").run([comp.name], store)
    for table, cols in comp.provides.items():
        ref = st.read(table, comp.name)
        assert list(store[table].index) == list(ref.index), table
        for col in cols:
            result = compare_column(comp.name, store[table], col, ref[col])
            assert result.status == "exact", result


BLOCK2 = ["joint_tour_frequency", "joint_tour_composition", "joint_tour_participation", "joint_tour_destination",
          "joint_tour_scheduling", "non_mandatory_tour_frequency", "atwork_subtour_frequency",
          "atwork_subtour_destination", "atwork_subtour_scheduling", "atwork_subtour_mode_choice"]


@pytest.fixture(scope="module")
def mtc_skims(tmp_path_factory):
    pytest.importorskip("tables")
    from asim_core.skims import import_omx

    return import_omx(REPO / "activitysim/examples/prototype_mtc/data/skims.omx",
                      tmp_path_factory.mktemp("skims") / "skims")


@pytest.mark.skipif(not BASELINE.exists(), reason="no ActivitySim 1.x baseline run")
@pytest.mark.parametrize("step", BLOCK2)
def test_block2_steps_match_1x_exactly(step, tmp_path, mtc_skims):
    from asim_core import components

    st = Asim1State(BASELINE)
    comp = getattr(components, step)(CONFIGS)
    store = st.load_store(st.previous_step(comp.name))
    Runner([comp], tmp_path / "run", skims=mtc_skims).run([comp.name], store)
    for table, cols in comp.provides.items():
        ref = st.read(table, comp.name)
        assert list(store[table].index) == list(ref.index), table
        for col in cols:
            result = compare_column(comp.name, store[table], col, ref[col])
            assert result.status == "exact", result
