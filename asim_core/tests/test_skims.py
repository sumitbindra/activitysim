import os
import time
from pathlib import Path

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


def write_omx(path, mats, lookup=None):
    with tables.open_file(str(path), "w") as f:
        f.root._v_attrs["OMX_VERSION"] = b"0.2"
        shape = next(iter(mats.values())).shape
        f.root._v_attrs["SHAPE"] = np.array(shape, dtype=np.int32)
        f.create_group("/", "data")
        for name, m in mats.items():
            f.create_carray("/data", name, obj=np.asarray(m, dtype=np.float64))
        if lookup is not None:
            f.create_group("/", "lookup")
            f.create_carray("/lookup", "zone_number", obj=np.asarray(lookup))


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


def test_cleanup_leaves_other_models_caches_alone(omx, tmp_path):
    path, mats = omx
    other = tmp_path / "other.omx"
    write_omx(other, {"DIST": np.ones((3, 3))})
    cache = tmp_path / "cache"
    mine = prepare_skims([path], cache)
    theirs = prepare_skims([other], cache)
    assert mine.path.exists() and theirs.path.exists()
    newer = prepare_skims([path], cache)  # replaces only the older cache of the same files
    assert not mine.path.exists() and newer.path.exists() and theirs.path.exists()


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


def _write_settings(configs, recode: bool | None):
    lines = ["input_table_list:", "  - tablename: land_use", "    index_col: zone_id"]
    if recode is not None:
        lines += ["    recode_columns:", "      zone_id: zero-based"]
        lines += [f"recode_pipeline_columns: {recode}"]
    (configs / "settings.yaml").write_text("\n".join(lines) + "\n")
    (configs / "network_los.yaml").write_text("taz_skims: skims.omx\n")


def test_zone_numbers_follow_1x_recoding_rules(tmp_path):
    from asim_core.skims import land_use_zones_recoded

    configs, data = tmp_path / "configs", tmp_path / "data"
    configs.mkdir(), data.mkdir()
    write_omx(data / "skims.omx", {"DIST": np.arange(9.0).reshape(3, 3)}, lookup=[10, 20, 30])

    # recoding off (1.x's default, as in MWCOG): original zone numbers, via the OMX lookup
    _write_settings(configs, recode=False)
    assert not land_use_zones_recoded([configs])
    store = prepare_skims_from_settings([configs], [data], tmp_path / "out1")
    assert store.positions([30, 10]).tolist() == [2, 0]

    # recode_columns alone does nothing without recode_pipeline_columns
    _write_settings(configs, recode=None)
    assert not land_use_zones_recoded([configs])

    # both set (as in MTC): zones renumbered 0, 1, 2 = matrix positions
    _write_settings(configs, recode=True)
    assert land_use_zones_recoded([configs])
    store = prepare_skims_from_settings([configs], [data], tmp_path / "out2")
    assert store.positions([2, 0]).tolist() == [2, 0]
    assert store.omx_zone_numbers.tolist() == [10, 20, 30]


def test_zone_order_is_checked_against_land_use(tmp_path):
    from asim_core.skims import check_zone_order

    write_omx(tmp_path / "skims.omx", {"DIST": np.ones((3, 3))}, lookup=[10, 20, 30])
    store = import_omx(tmp_path / "skims.omx", tmp_path / "s")
    good = Table("land_use", np.array([0, 1, 2]), "zone_id", {"_original_zone_id": np.array([10, 20, 30])})
    bad = Table("land_use", np.array([0, 1, 2]), "zone_id", {"_original_zone_id": np.array([10, 30, 20])})
    assert check_zone_order(store, good) is None
    assert "doesn't match land_use" in check_zone_order(store, bad)


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


def test_skim_access_and_lookup_options_give_the_same_values(tmp_path):
    """Every way of holding skims (memory-mapped, pages mapped up front, in memory) and of
    looking them up (2-D indexing, 1-D gather) gives the reference's values, bit for bit."""
    from asim_core.seams import Engine

    rng = np.random.default_rng(1)
    mats = {f"S{k}__{p}": rng.random((40, 40)).astype(np.float64) for k in range(5) for p in ("AM", "MD")}
    path = tmp_path / "s.omx"
    write_omx(path, mats)
    store = import_omx(path, tmp_path / "cache")
    o, d = rng.integers(0, 40, 5000), rng.integers(0, 40, 5000)
    names = ["S1__AM", "S3__MD", "S4__AM"]
    ref = {n: store[n][o, d] for n in names}
    for access in ("memory_map", "prefault", "in_memory"):
        for lookup in ("index_2d", "flat_take"):
            e = Engine({"skim_access": access, "skim_lookup": lookup})
            held = e.skim_access(store, names)
            for n in names:
                assert np.array_equal(e.skim_lookup(held[n], o, d), ref[n]), (access, lookup, n)
            assert np.array_equal(held.positions(np.arange(5)), store.positions(np.arange(5)))


def test_zones_without_a_lookup_are_numbered_from_1_as_in_1x(tmp_path):
    """Original zone numbers and no zone lookup in the OMX file: 1.x reads zone z from
    row z - 1 ("assume this is a one-based skim map"). Renumbered zones: zone i is row i."""
    configs, data = tmp_path / "configs", tmp_path / "data"
    configs.mkdir(), data.mkdir()
    mat = np.arange(9.0).reshape(3, 3)
    write_omx(data / "skims.omx", {"DIST": mat})

    _write_settings(configs, recode=False)
    store = prepare_skims_from_settings([configs], [data], tmp_path / "out1")
    assert store.positions([1, 3]).tolist() == [0, 2]
    with pytest.raises(KeyError):
        store.positions([0])  # no zone 0
    with pytest.raises(KeyError):
        store.positions([4])

    _write_settings(configs, recode=True)
    store = prepare_skims_from_settings([configs], [data], tmp_path / "out2")
    assert store.positions([0, 2]).tolist() == [0, 2]


def test_a_zone_lookup_in_any_order_finds_the_right_values(tmp_path):
    """Zones listed out of order in the OMX lookup are matched by number, with both lookups."""
    from asim_core.seams import Engine

    order = [30, 10, 20]  # matrix rows/columns, not ascending
    mat = np.array([[0.0, 1, 2], [3, 4, 5], [6, 7, 8]])  # mat[i, j] = 3 * i + j
    write_omx(tmp_path / "skims.omx", {"DIST": mat}, lookup=order)
    store = import_omx(tmp_path / "skims.omx", tmp_path / "s", zone_ids="omx")
    orig, dest = np.array([10, 10, 30, 20]), np.array([30, 20, 10, 20])
    row = {z: i for i, z in enumerate(order)}
    expected = np.array([mat[row[o], row[d]] for o, d in zip(orig, dest)], dtype=np.float32)
    for lookup in ("index_2d", "flat_take"):
        got = Engine({"skim_lookup": lookup}).skim_lookup(
            store["DIST"], store.positions(orig), store.positions(dest))
        assert np.array_equal(got, expected), lookup


class _Problems:
    def __init__(self):
        self.found = []

    def add(self, level, table, column, message, *args, **kwargs):
        self.found.append((level, message))


@pytest.mark.parametrize("lookup, zones, recoded, expect", [
    (None, [1, 2, 3], False, None),                       # zone z is row z
    (None, [0, 1, 2], False, "outside 1 to 3"),           # zone 0 has no row
    (None, [1, 2, 7], False, "outside 1 to 3"),
    (None, [1, 2], False, "have no land use"),            # fewer zones than rows: warning
    (None, [1, 2, 3], True, None),                        # renumbered: i-th smallest zone is row i
    (None, [1, 2, 7], True, "gaps"),                      # can't be checked: warning
    ([30, 10, 20], [10, 20, 30], False, None),            # any order, matched by number
    ([30, 10, 20], [10, 20, 30], True, "ascending zone order"),  # renumbered: lookup ignored by 1.x
    ([10, 20, 30], [10, 20, 30], True, None),
    ([10, 20, 30], [10, 40], False, "not in the skims"),
])
def test_input_checks_find_zones_where_1x_would_look(lookup, zones, recoded, expect):
    from asim_core.inputs.checks import _Checker

    p = _Problems()
    _Checker._check_skim_zones(p, Path("skims.omx"), (3, 3),
                               None if lookup is None else np.array(lookup), np.sort(np.array(zones)), recoded)
    if expect is None:
        assert p.found == []
    else:
        assert len(p.found) == 1 and expect in p.found[0][1], p.found


def test_zone_pairs_share_positions_and_give_the_same_values():
    from asim_core.seams import Engine
    from asim_core.skims import ZonePairs

    rng = np.random.default_rng(3)
    m = rng.random((30, 30)).astype(np.float32)
    o, d = rng.integers(0, 30, 1000), rng.integers(0, 30, 1000)
    pairs = ZonePairs(o, d, 30)
    rows = np.flatnonzero(o % 2 == 0)
    for lookup in ("index_2d", "flat_take"):
        e = Engine({"skim_lookup": lookup})
        assert np.array_equal(e.skim_lookup(m, o, d, pairs=pairs), m[o, d])
        assert np.array_equal(e.skim_lookup(m, d, o, pairs=pairs.reversed()), m[d, o])
        sub = pairs.subset("even", rows)
        assert np.array_equal(e.skim_lookup(m, sub.orig, sub.dest, pairs=sub), m[o[rows], d[rows]])
    assert pairs.reversed().reversed() is pairs
    assert pairs.subset("even", rows) is pairs.subset("even", rows)
