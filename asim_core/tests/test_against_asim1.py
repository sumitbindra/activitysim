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


BLOCK3 = ["stop_frequency", "trip_purpose", "trip_destination", "trip_purpose_and_destination", "trip_scheduling",
          "trip_mode_choice"]


@pytest.mark.skipif(not BASELINE.exists(), reason="no ActivitySim 1.x baseline run")
@pytest.mark.parametrize("step", BLOCK3)
def test_block3_steps_match_1x_exactly(step, tmp_path, mtc_skims):
    from asim_core import components

    st = Asim1State(BASELINE)
    comp = getattr(components, step)(CONFIGS)
    store = st.load_store(st.previous_step(comp.name))
    Runner([comp], tmp_path / "run", skims=mtc_skims).run([comp.name], store)
    for table, cols in comp.provides.items():
        ref = st.read(table, comp.name)
        assert list(store[table].index) == list(ref.index), table
        assert sorted(store[table].column_names) == sorted(ref.columns), table  # e.g. trips' failed dropped
        for col in cols:
            result = compare_column(comp.name, store[table], col, ref[col])
            assert result.status == "exact", result


@pytest.mark.skipif(not BASELINE.exists(), reason="no ActivitySim 1.x baseline run")
def test_trip_destination_original_sampling_matches_1x_exactly(tmp_path, mtc_skims):
    """The way sampling worked before (every column copied, sizes matched row by row), behind --seams."""
    from asim_core import components
    from asim_core.seams import Engine

    st = Asim1State(BASELINE)
    comp = components.trip_destination(CONFIGS)
    store = st.load_store(st.previous_step(comp.name))
    engine = Engine({"sample_columns": "all_columns", "size_lookup": "match"})
    Runner([comp], tmp_path / "run", skims=mtc_skims, engine=engine).run([comp.name], store)
    ref = st.read("trips", comp.name)
    assert list(store["trips"].index) == list(ref.index)
    for col in comp.provides["trips"]:
        result = compare_column(comp.name, store["trips"], col, ref[col])
        assert result.status == "exact", result


def _spec_with_rows(configs, name, rows):
    """A copy of an MTC spec file with rows added, (expression, coefficients) each, for a configs dir put first."""
    text = (REPO / "activitysim/examples/prototype_mtc/configs" / name).read_text().rstrip("\n")
    before = "," * text.split("\n")[0].split(",").index("Expression")  # Label, Description: empty
    (configs / name).write_text(text + "\n" + "\n".join(f"{before}{e},{c}" for e, c in rows) + "\n")


@pytest.mark.skipif(not BASELINE.exists(), reason="no ActivitySim 1.x baseline run")
def test_trip_destination_spec_columns_are_checked_before_the_run(tmp_path, mtc_skims):
    """A column the sample or choice spec reads, which the trips don't have, stops the run before it starts."""
    from asim_core import components
    from asim_core.runner import StartupCheckError

    st = Asim1State(BASELINE)
    store = st.load_store(st.previous_step("trip_destination"))
    configs = tmp_path / "configs"
    configs.mkdir()
    ones, shopping = ",".join(["1"] * 10), "0,0,0,0,1,0,0,0,0,0"
    # a made-up column (every purpose); a tour column (1.x gives the specs the trips, not the tours)
    _spec_with_rows(configs, "trip_destination_sample.csv", [("df.no_such_column", ones), ("df.hhsize", shopping)])
    _spec_with_rows(configs, "trip_destination.csv", [("df.pick_cnt", ones)])
    comp = components.trip_destination([configs] + CONFIGS)
    with pytest.raises(StartupCheckError) as err:
        Runner([comp], tmp_path / "run", skims=mtc_skims).run([comp.name], store)
    message = str(err.value)
    assert "trip_destination_sample.csv reads 'no_such_column'" in message
    assert "trip_destination_sample.csv reads 'hhsize' (purposes shopping)" in message
    assert "trip_destination.csv reads 'pick_cnt'" in message

    # a zone's size column read directly: 1.x's sample joins them, asim_core does not (yet)
    _spec_with_rows(configs, "trip_destination_sample.csv", [("df.shopping", ones)])
    (configs / "trip_destination.csv").unlink()
    comp = components.trip_destination([configs] + CONFIGS)
    with pytest.raises(NotImplementedError, match=r"size columns \['shopping'\]"):
        Runner([comp], tmp_path / "run", skims=mtc_skims).run([comp.name], store)


def test_trip_ids_follow_1x_rule():
    """1.x's trip IDs: tour_id x 2 x (most trips per leg) + trip number, inbound after outbound."""
    import pandas as pd
    from asim_core.components.stop_frequency import initialize_from_tours

    alts = pd.DataFrame({"out": [0, 1, 3], "in": [0, 2, 1]}, index=pd.Index(["0out_0in", "1out_2in", "3out_1in"],
                                                                            name="alt"))
    tours = pd.DataFrame({"stop_frequency": ["1out_2in", "0out_0in", "3out_1in"], "person_id": [1, 2, 3],
                          "household_id": [10, 20, 30], "primary_purpose": ["work", "shopping", "school"],
                          "origin": [5, 6, 7], "destination": [8, 9, 10]},
                         index=pd.Index([100, 7, 55], name="tour_id"))
    trips = initialize_from_tours(tours, alts, 4)
    assert list(trips.index) == [801, 802, 805, 806, 807, 57, 61, 441, 442, 443, 444, 445, 446]
    t = trips.loc[[801, 802, 805, 806, 807]]
    assert list(t.outbound) == [True, True, False, False, False]
    assert list(t.trip_count) == [2, 2, 3, 3, 3]
    assert list(t.origin) == [5, 5, 8, 8, 8] and list(t.destination) == [8, 8, 5, 5, 5]  # stops are chosen later


MWCOG = REPO / "asim_core/work/mwcog/asim1_legacy/pipeline.parquetpipeline"
MWCOG_CONFIGS = [REPO / "activitysim/examples/prototype_mwcog/configs"]
MWCOG_STEPS = ["work_from_home", "transit_pass_subsidy", "transit_pass_ownership", "telecommute_frequency"]


@pytest.mark.skipif(not MWCOG.exists(), reason="no ActivitySim 1.x MWCOG run")
@pytest.mark.parametrize("step", MWCOG_STEPS)
def test_mwcog_person_steps_match_1x_exactly(step, tmp_path):
    from asim_core import components

    st = Asim1State(MWCOG)
    comp = getattr(components, step)(MWCOG_CONFIGS)
    store = st.load_store(st.previous_step(comp.name))
    Runner([comp], tmp_path / "run").run([comp.name], store)
    if step == "work_from_home":  # also sets is_out_of_home_worker, and -1 workplaces for those at home
        assert sorted(comp.provides["persons"]) == ["is_out_of_home_worker", "work_from_home", "workplace_zone_id"]
    for table, cols in comp.provides.items():
        ref = st.read(table, comp.name)
        assert list(store[table].index) == list(ref.index), table
        for col in cols:
            result = compare_column(comp.name, store[table], col, ref[col])
            assert result.status == "exact", result


@pytest.mark.skipif(not MWCOG.exists(), reason="no ActivitySim 1.x MWCOG run")
def test_transit_pass_subsidy_preprocessor_runs_before_the_filter(tmp_path):
    """
    1.x's transit_pass_subsidy runs its preprocessor on every person, then
    picks the choosers by CHOOSER_FILTER_COLUMN_NAME, which may be a column
    the preprocessor makes. Here: whether someone else in the household
    works (worked out over every person). A person's draw depends only on
    their ID, so the choosers choose as in 1.x's run, where everyone chose.
    """
    import numpy as np

    from asim_core import components

    configs = tmp_path / "configs"
    configs.mkdir()
    settings = (MWCOG_CONFIGS[0] / "transit_pass_subsidy.yaml").read_text()
    (configs / "transit_pass_subsidy.yaml").write_text(
        settings + "\nCHOOSER_FILTER_COLUMN_NAME: other_worker\npreprocessor:\n  SPEC: tps_pre\n  DF: persons\n")
    (configs / "tps_pre.csv").write_text(
        'Description,Target,Expression\n,other_worker,"other_than(df.household_id, df.is_worker)"\n')
    st = Asim1State(MWCOG)
    before = st.previous_step("transit_pass_subsidy")
    store = st.load_store(before)
    comp = components.transit_pass_subsidy([configs] + MWCOG_CONFIGS)
    Runner([comp], tmp_path / "run").run([comp.name], store)

    persons = st.read("persons", before)
    workers = persons.groupby("household_id").is_worker.transform("sum")
    other_worker = (workers - persons.is_worker.astype(int)).to_numpy() > 0
    assert other_worker.any() and not other_worker.all()
    everyone = st.read("persons", "transit_pass_subsidy").transit_pass_subsidy.to_numpy()
    got = np.asarray(store["persons"]["transit_pass_subsidy"])
    assert got.dtype == np.int64
    assert (got == np.where(other_worker, everyone, 0)).all()


@pytest.mark.skipif(not BASELINE.exists(), reason="no ActivitySim 1.x baseline run")
@pytest.mark.parametrize("step", ["write_trip_matrices", "write_tables", "summarize"])
def test_output_files_match_1x_exactly(step, tmp_path, mtc_skims):
    """The output steps from 1.x's saved state: every OMX cell, every CSV byte (checkpoints aside: they
    need the run's steps), every summary."""
    import sys

    sys.path.insert(0, str(REPO / "asim_core/harness"))
    from compare_outputs import HISTORY_FILES, compare_output_files

    from asim_core.components.output import write_tables
    from asim_core.components.summarize import summarize
    from asim_core.components.trip_matrices import write_trip_matrices

    make = {"write_trip_matrices": write_trip_matrices, "write_tables": write_tables, "summarize": summarize}[step]
    st = Asim1State(BASELINE)
    comp = make(CONFIGS)
    store = st.load_store(st.previous_step(comp.name))
    Runner([comp], tmp_path / "run", skims=mtc_skims, save_outputs=False).run([comp.name], store)
    rows = compare_output_files(tmp_path / "run" / "output", BASELINE.parent, steps=[step], skip=HISTORY_FILES)
    assert len(rows) >= 5
    assert all(status == "exact" for _, status, _ in rows), [r for r in rows if r[1] != "exact"]


SANDAG = REPO / "asim_core/work/sandag/asim1_1zone/pipeline.parquetpipeline"
SANDAG_EXAMPLE = REPO / "activitysim/examples/placeholder_sandag"
SANDAG_CONFIGS = [SANDAG_EXAMPLE / "configs_1_zone", REPO / "activitysim/examples/prototype_mtc/configs"]


@pytest.mark.skipif(not SANDAG.exists(), reason="no ActivitySim 1.x SANDAG one-zone run")
@pytest.mark.parametrize("step", ["initialize_households", "cdap_simulate", "stop_frequency", "trip_mode_choice"])
def test_sandag_one_zone_steps_match_1x_exactly(step, tmp_path):
    """SANDAG's data with MTC's model files layered under its own (constants.yaml inherits MTC's)."""
    from asim_core import components
    from asim_core.skims import prepare_skims_from_settings

    st = Asim1State(SANDAG)
    data = [SANDAG_EXAMPLE / "data_1"]
    make = getattr(components, step)
    comp = make(SANDAG_CONFIGS, data) if step == "initialize_households" else make(SANDAG_CONFIGS)
    store = st.load_store(st.previous_step(comp.name))
    skims = prepare_skims_from_settings(SANDAG_CONFIGS, data, REPO / "asim_core/work/placeholder_sandag")
    Runner([comp], tmp_path / "run", skims=skims, save_outputs=False).run([comp.name], store)
    for table, cols in comp.provides.items():
        ref = st.read(table, comp.name)
        assert list(store[table].index) == list(ref.index), table
        for col in cols:
            result = compare_column(comp.name, store[table], col, ref[col])
            assert result.status == "exact", result
