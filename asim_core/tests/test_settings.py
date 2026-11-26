"""Settings files are read as 1.x's read_settings_file reads them, whichever file it is."""

import pytest

from asim_core.spec.legacy import load_global_constants, read_settings


def write(folder, name, text):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(text)


def test_inherit_settings_fills_in_from_the_next_folder(tmp_path):
    region, base = tmp_path / "region", tmp_path / "base"
    write(region, "constants.yaml", "inherit_settings: True\nTRANSIT_SCALE_FACTOR: 1\n")
    write(base, "constants.yaml", "TRANSIT_SCALE_FACTOR: 100\nPTYPE_FULL: 1\n")
    assert load_global_constants([region, base]) == {"TRANSIT_SCALE_FACTOR": 1, "PTYPE_FULL": 1}
    # without the directive, the first file is the whole of it (as before)
    write(region, "constants.yaml", "TRANSIT_SCALE_FACTOR: 1\n")
    assert load_global_constants([region, base]) == {"TRANSIT_SCALE_FACTOR": 1}


def test_a_chain_of_three_and_inheriting_a_named_file(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    write(a, "m.yaml", "inherit_settings: True\nX: a\n")
    write(b, "m.yaml", "inherit_settings: True\nX: b\nY: b\n")
    write(c, "m.yaml", "X: c\nY: c\nZ: c\n")
    assert read_settings([a, b, c], "m.yaml") == {"X": "a", "Y": "b", "Z": "c"}
    write(a, "settings_mp.yaml", "inherit_settings: m.yaml\nmultiprocess: True\n")  # 1.x's settings_mp.yaml
    assert read_settings([a, b, c], "settings_mp") == {"multiprocess": True, "X": "a", "Y": "b", "Z": "c"}


def test_include_settings_and_missing_files(tmp_path):
    a = tmp_path / "a"
    write(a, "joint_tour_destination.yaml", "include_settings: non_mandatory_tour_destination.yaml\n")
    write(a, "non_mandatory_tour_destination.yaml", "SAMPLE_SIZE: 30\n")
    assert read_settings([a], "joint_tour_destination.yaml") == {"SAMPLE_SIZE": 30}
    write(a, "bad.yaml", "include_settings: x.yaml\nSAMPLE_SIZE: 3\n")
    with pytest.raises(ValueError, match="alone"):
        read_settings([a], "bad.yaml")
    write(a, "loop.yaml", "include_settings: loop.yaml\n")
    with pytest.raises(ValueError, match="each other"):
        read_settings([a], "loop.yaml")
    with pytest.raises(FileNotFoundError):
        read_settings([a], "nothing.yaml")
    assert read_settings([a], "nothing.yaml", mandatory=False) == {}


def test_settings_files_read_as_utf8_or_the_system_encoding(tmp_path, monkeypatch):
    """UTF-8 (YAML's encoding) whatever the system; a file that isn't UTF-8 (Windows Notepad's "ANSI")
    falls back to the system's encoding, as 1.x opens settings files (here Windows' usual cp1252)."""
    import locale

    from asim_core.spec.legacy import read_text, read_yaml

    (tmp_path / "utf8.yaml").write_bytes("label: Zürich – downtown  # a comment with → and µ\n".encode("utf-8"))
    assert read_yaml(tmp_path / "utf8.yaml") == {"label": "Zürich – downtown"}
    (tmp_path / "ansi.yaml").write_bytes("label: café\n".encode("cp1252"))  # not valid UTF-8
    monkeypatch.setattr(locale, "getpreferredencoding", lambda do_setlocale=True: "cp1252")
    assert read_text(tmp_path / "ansi.yaml") == "label: café\n"
    assert read_yaml(tmp_path / "ansi.yaml") == {"label": "café"}
