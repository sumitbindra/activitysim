import pytest

from asim_harness import example


def test_list_and_read_configs(tmp_root):
    names = [c["path"] for c in example.list_configs()]
    assert "settings.yaml" in names and "trip_mode_choice.yaml" in names
    assert example.read_config("trip_mode_choice.yaml").startswith("SPEC:")
    assert example.base_models()[0] == "initialize_landuse"
    assert example.read_config("trip_mode_choice_coefficients.csv", max_bytes=5).startswith("coeff")
    assert "truncated" in example.read_config("trip_mode_choice_coefficients.csv", max_bytes=5)


@pytest.mark.parametrize("bad", ["../settings.yaml", "/etc/passwd", "", "sub/../../x", "..", "C:/x"])
def test_read_config_rejects_escapes(tmp_root, bad):
    with pytest.raises((ValueError, FileNotFoundError)):
        example.read_config(bad)


def test_read_config_missing(tmp_root):
    with pytest.raises(FileNotFoundError):
        example.read_config("nope.csv")


def test_check_example(tmp_root, monkeypatch):
    example.check_example()
    monkeypatch.setenv("ASIM_EXAMPLE_DIR", str(tmp_root / "missing"))
    with pytest.raises(example.ExampleMissing, match="asim init"):
        example.check_example()
