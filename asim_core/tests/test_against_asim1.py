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
