import numpy as np
import pandas as pd
import pytest

from asim_core.component import Component, ContractError
from asim_core.runner import Runner, StartupCheckError
from asim_core.store import Store, Table


def make_store():
    hh = Table("households", np.array([10, 20, 30]), "household_id",
               {"income": np.array([10.0, 50.0, 90.0]), "home_zone_id": np.array([1, 2, 1])})
    lu = Table("land_use", np.array([1, 2]), "zone_id", {"density": np.array([5.0, 7.0])})
    return Store([hh, lu])


def rich(ctx):
    return {"households": {"is_rich": ctx.column("households", "income") > 40}}


def rich_and_dense(ctx):
    ids = ctx.ids("households")
    density = ctx.column_for("land_use", "density", ctx.column("households", "home_zone_id"))
    return {"households": {"rich_and_dense": ctx.column("households", "is_rich") & (density > 6)},
            } if len(ids) else {}


STEP_A = Component("rich", rich, needs={"households": ["income"]},
                   provides={"households": ["is_rich"]})
STEP_B = Component("rich_and_dense", rich_and_dense,
                   needs={"households": ["is_rich", "home_zone_id"], "land_use": ["density"]},
                   provides={"households": ["rich_and_dense"]})


def test_runs_steps_in_order_and_saves_outputs(tmp_path):
    store = make_store()
    report = Runner([STEP_A, STEP_B], tmp_path).run(["rich", "rich_and_dense"], store)
    assert store["households"]["rich_and_dense"].tolist() == [False, True, False]
    assert [m for _, m in report.plan] == ["computed", "computed"]
    saved = pd.read_parquet(tmp_path / "state" / "rich" / "households.parquet")
    assert saved["is_rich"].tolist() == [False, True, True]
    assert len(report.profiler.records) == 2


def test_dropping_a_needed_step_fails_before_anything_runs(tmp_path):
    store = make_store()
    with pytest.raises(StartupCheckError, match=r"needs households.\['is_rich'\]"):
        Runner([STEP_A, STEP_B], tmp_path).run(["rich_and_dense"], store)
    assert "is_rich" not in store["households"]  # nothing ran


def test_unknown_step_fails_at_startup(tmp_path):
    with pytest.raises(StartupCheckError, match="no component"):
        Runner([STEP_A], tmp_path).run(["rich", "park_and_ride"], make_store())


def test_reading_an_undeclared_column_is_an_error(tmp_path):
    sneaky = Component("sneaky", lambda ctx: {"households": {"x": ctx.column("households", "income")}},
                       needs={"households": ["home_zone_id"]}, provides={"households": ["x"]})
    with pytest.raises(ContractError, match="did not declare"):
        Runner([sneaky], tmp_path).run(["sneaky"], make_store())


def test_writing_an_undeclared_column_is_an_error(tmp_path):
    sloppy = Component("sloppy", lambda ctx: {"households": {"y": np.zeros(3)}},
                       needs={"households": ["income"]}, provides={"households": ["x"]})
    with pytest.raises(ContractError, match="declared"):
        Runner([sloppy], tmp_path).run(["sloppy"], make_store())
