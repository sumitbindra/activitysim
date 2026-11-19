import numpy as np
import pytest

from asim_core.component import Component, ContractError
from asim_core.runner import Runner, StartupCheckError
from asim_core.skims import SkimStore, import_omx
from asim_core.store import Store, Table

tables = pytest.importorskip("tables")  # only needed for importing OMX


@pytest.fixture
def omx(tmp_path):
    """A tiny OMX file: 3 zones, one plain skim and one per time period."""
    path = tmp_path / "skims.omx"
    rng = np.random.default_rng(1)
    mats = {"DIST": rng.random((3, 3)) * 10, "TIME__AM": rng.random((3, 3)) * 30,
            "TIME__PM": rng.random((3, 3)) * 30}
    with tables.open_file(str(path), "w") as f:
        f.root._v_attrs["OMX_VERSION"] = b"0.2"
        f.root._v_attrs["SHAPE"] = np.array([3, 3], dtype=np.int32)
        f.create_group("/", "data")
        for name, m in mats.items():
            f.create_carray("/data", name, obj=m)
    return path, mats


def test_import_and_read(omx, tmp_path):
    path, mats = omx
    store = import_omx(path, tmp_path / "store")
    assert store.names == ["DIST", "TIME__AM", "TIME__PM"]
    assert store.dtype == np.float32
    for name, m in mats.items():
        np.testing.assert_array_equal(store[name], m.astype(np.float32))
    reopened = SkimStore(tmp_path / "store")
    np.testing.assert_array_equal(reopened["DIST"], store["DIST"])
    with pytest.raises(ValueError):
        reopened["DIST"][0, 0] = 1.0  # read-only


def test_zone_positions(omx, tmp_path):
    path, _ = omx
    default = import_omx(path, tmp_path / "a")
    assert default.positions([2, 0]).tolist() == [2, 0]
    with pytest.raises(KeyError):
        default.positions([3])
    numbered = import_omx(path, tmp_path / "b", zone_ids=[101, 102, 103])
    assert numbered.positions([103, 101]).tolist() == [2, 0]
    with pytest.raises(KeyError):
        numbered.positions([5])
    with pytest.raises(KeyError, match="no skim named"):
        numbered["NOPE"]


def _store():
    return Store([Table("land_use", np.array([0, 1, 2]), "zone_id", {"x": np.ones(3)})])


def test_components_must_declare_skims(omx, tmp_path):
    path, mats = omx
    skims = import_omx(path, tmp_path / "s")

    def run(ctx):
        pos = ctx.skim_positions(ctx.ids("land_use"))
        return {"land_use": {"d": ctx.skim("DIST", pos, pos)}}

    ok = Component("ok", run, needs={"land_use": [], "skims": ["DIST"]}, provides={"land_use": ["d"]})
    store = _store()
    Runner([ok], tmp_path / "r1", skims=skims).run(["ok"], store)
    np.testing.assert_array_equal(store["land_use"]["d"], np.diag(mats["DIST"]).astype(np.float32))

    sneaky = Component("sneaky", run, needs={"land_use": [], "skims": ["TIME__AM"]},
                       provides={"land_use": ["d"]})
    with pytest.raises(ContractError, match="skims.DIST"):
        Runner([sneaky], tmp_path / "r2", skims=skims).run(["sneaky"], _store())

    missing = Component("missing", run, needs={"land_use": [], "skims": ["TOLL"]},
                        provides={"land_use": ["d"]})
    with pytest.raises(StartupCheckError, match="TOLL"):
        Runner([missing], tmp_path / "r3", skims=skims).run(["missing"], _store())
    with pytest.raises(StartupCheckError, match="no skim store"):
        Runner([ok], tmp_path / "r4").run(["ok"], _store())
