import numpy as np
import pytest

pytest.importorskip("tables")

from asim_core.omx import read_omx, write_omx  # noqa: E402


def test_round_trip_and_openmatrix_layout(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "trips.omx"
    a, b = rng.random((4, 4)), np.arange(16.0).reshape(4, 4)
    write_omx(path, {"SOV_AM": a, "WALK_AM": b}, {"zone_id": np.array([1, 2, 5, 9])})
    matrices, mappings, attrs = read_omx(path)
    assert np.array_equal(matrices["SOV_AM"], a) and np.array_equal(matrices["WALK_AM"], b)
    assert mappings["zone_id"].dtype == np.uint32 and mappings["zone_id"].tolist() == [1, 2, 5, 9]
    assert attrs["OMX_VERSION"] == b"0.2" and attrs["SHAPE"].dtype == np.int32 and attrs["SHAPE"].tolist() == [4, 4]
    omx = pytest.importorskip("openmatrix")  # the package 1.x writes with: it reads ours
    with omx.open_file(str(path)) as f:
        assert sorted(f.list_matrices()) == ["SOV_AM", "WALK_AM"] and f.shape() == (4, 4)
        assert np.array_equal(np.array(f["SOV_AM"]), a) and f.mapping("zone_id") == {1: 0, 2: 1, 5: 2, 9: 3}


def test_shapes_must_agree(tmp_path):
    with pytest.raises(ValueError, match="matrices are"):
        write_omx(tmp_path / "x.omx", {"A": np.zeros((2, 2)), "B": np.zeros((3, 3))}, {})
