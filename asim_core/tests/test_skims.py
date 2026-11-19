import os
import time

import numpy as np
import pytest

from asim_core.component import Component, ContractError
from asim_core.runner import Runner, StartupCheckError
from asim_core.skims import (
    SkimStore,
    fingerprint,
    import_omx,
    omx_files_from_settings,
    prepare_skims,
    prepare_skims_from_settings,
)
from asim_core.store import Store, Table

tables = pytest.importorskip("tables")  # only needed for importing OMX


def write_omx(path, mats):
    with tables.open_file(str(path), "w") as f:
        f.root._v_attrs["OMX_VERSION"] = b"0.2"
        shape = next(iter(mats.values())).shape
        f.root._v_attrs["SHAPE"] = np.array(shape, dtype=np.int32)
        f.create_group("/", "data")
        for name, m in mats.items():
            f.create_carray("/data", name, obj=np.asarray(m, dtype=np.float64))


def bump_mtime(path):
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


@pytest.fixture
def omx(tmp_path):
    """A tiny OMX file: 3 zones, one plain skim and one per time period."""
    path = tmp_path / "skims.omx"
    rng = np.random.default_rng(1)
    mats = {"DIST": rng.random((3, 3)) * 10, "TIME__AM": rng.random((3, 3)) * 30,
            "TIME__PM": rng.random((3, 3)) * 30}
    write_omx(path, mats)
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
    with pytest.raises(FileExistsError):
        import_omx(path, tmp_path / "store")  # never overwritten


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


def test_fingerprint_changes_when_the_file_changes(omx):
    path, _ = omx
    before = fingerprint([path])
    assert fingerprint([path]) == before
    bump_mtime(path)
    assert fingerprint([path]) != before
    assert fingerprint([path], dtype="float64") != fingerprint([path])


def test_read_skim_cache_off_imports_every_run(omx, tmp_path):
    path, _ = omx
    cache = tmp_path / "cache"
    first = prepare_skims([path], cache, reuse=False)
    second = prepare_skims([path], cache, reuse=False)
    assert second.path != first.path  # a new folder: never overwritten
    np.testing.assert_array_equal(first["DIST"], second["DIST"])  # the first is still readable
    remaining = [p for p in cache.iterdir() if p.name.startswith("asim_core_taz_")]
    assert remaining == [second.path]  # old folder removed


def test_read_skim_cache_on_reuses_only_unchanged_skims(omx, tmp_path):
    path, mats = omx
    cache = tmp_path / "cache"
    first = prepare_skims([path], cache, reuse=True)
    again = prepare_skims([path], cache, reuse=True)
    assert again.path == first.path  # reused

    # assignment writes new skims: the cache must not be reused
    new = {k: v + 1.0 for k, v in mats.items()}
    write_omx(path, new)
    bump_mtime(path)
    updated = prepare_skims([path], cache, reuse=True)
    assert updated.path != first.path
    np.testing.assert_array_equal(updated["DIST"], new["DIST"].astype(np.float32))


def test_later_files_win_and_unknown_periods_are_skipped(tmp_path):
    a, b = tmp_path / "a.omx", tmp_path / "b.omx"
    write_omx(a, {"DIST": np.ones((2, 2)), "TIME__AM": np.ones((2, 2)), "TIME__XX": np.ones((2, 2))})
    write_omx(b, {"DIST": np.full((2, 2), 2.0)})
    with pytest.warns(UserWarning) as record:
        store = import_omx([a, b], tmp_path / "s", periods=["AM", "PM"])
    messages = " ".join(str(w.message) for w in record)
    assert "duplicate skim 'DIST'" in messages and "unknown time period 'XX'" in messages
    assert store.names == ["DIST", "TIME__AM"]
    assert store["DIST"][0, 0] == 2.0


def test_a_failed_import_leaves_nothing_behind(tmp_path):
    bad = tmp_path / "bad.omx"
    write_omx(bad, {"DIST": np.ones((2, 3))})  # not square
    with pytest.raises(ValueError, match="square"):
        prepare_skims([bad], tmp_path / "cache")
    assert not any((tmp_path / "cache").iterdir())


def test_files_found_as_1x_does(tmp_path):
    configs, data1, data2 = (tmp_path / d for d in ("configs", "data1", "data2"))
    for d in (configs, data1, data2):
        d.mkdir()
    write_omx(data2 / "skims1.omx", {"DIST": np.ones((2, 2))})
    write_omx(data2 / "skims2.omx", {"TIME__AM": np.ones((2, 2))})
    for value in ["taz_skims: skims*.omx",
                  "taz_skims: [skims1.omx, skims2.omx]",
                  "taz_skims:\n  omx: skims*.omx"]:
        (configs / "network_los.yaml").write_text(value + "\nskim_time_periods:\n  labels: [AM, PM]\n")
        found = omx_files_from_settings([configs], [data1, data2])
        assert [p.name for p in found] == ["skims1.omx", "skims2.omx"]
    store = prepare_skims_from_settings([configs], [data1, data2], tmp_path / "out")
    assert store.names == ["DIST", "TIME__AM"]
    assert store.path.parent == tmp_path / "out" / "cache"
    (configs / "network_los.yaml").write_text("taz_skims: missing.omx\n")
    with pytest.raises(FileNotFoundError):
        omx_files_from_settings([configs], [data1, data2])


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
