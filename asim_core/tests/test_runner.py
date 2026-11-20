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


class FakeRun:
    """A saved 1.x run: tables as they were after each step."""

    def __init__(self, states):
        self.states = states  # step -> {table: DataFrame}, in step order
        self.steps = list(states)

    def previous_step(self, step):
        i = self.steps.index(step)
        return self.steps[i - 1] if i else None

    def tables_at(self, step):
        out = {}
        for s in self.steps[:self.steps.index(step) + 1]:
            out.update({t: s for t in self.states[s]})
        return out

    def written_at(self, step):
        return list(self.states[step])

    def read(self, table, step):
        saved = self.tables_at(step).get(table)
        if saved is None:
            raise KeyError(table)
        return self.states[saved][table]

    def columns_at(self, step):
        return {t: list(self.read(t, step).columns) for t in self.tables_at(step)}


def test_replayed_step_changes_win_and_untouched_rows_keep_ours(tmp_path):
    idx = pd.Index([10, 20, 30], name="household_id")
    income = [10.0, 50.0, 90.0]
    run = FakeRun({
        "load": {"households": pd.DataFrame({"income": income}, index=idx)},
        "count": {"households": pd.DataFrame({"income": income, "n": [20.0, 100.0, 180.0]}, index=idx)},
        # a later 1.x step changes one household's value
        "adjust": {"households": pd.DataFrame({"income": income, "n": [20.0, 999.0, 180.0]}, index=idx)},
    })
    count = Component("count", lambda ctx: {"households": {"n": ctx.column("households", "income") * 2}},
                      needs={"households": ["income"]}, provides={"households": ["n"]})
    store = Store()
    report = Runner([count], tmp_path, replay=run).run(["load", "count", "adjust"], store)
    assert store["households"]["n"].tolist() == [20.0, 999.0, 180.0]
    (cmp,) = report.carried_over
    assert cmp.status == "exact" and cmp.rows == 2  # the two rows 1.x's step left alone
    assert "1 rows changed" in cmp.note
