"""The harness's thread setting: one thread by default, as 1.x's command line; --threads N; --fast for no limit."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import _threads  # noqa: E402


@pytest.fixture
def clean_env(monkeypatch):
    for v in _threads.VARIABLES:  # restored after the test, so other tests' worker processes don't inherit them
        monkeypatch.setenv(v, "unset-by-test")
        monkeypatch.delenv(v)
    return monkeypatch


@pytest.mark.parametrize("argv,expect", [([], 1), (["--threads", "4"], 4), (["--threads=6"], 6),
                                          (["--fast"], None), (["--threads", "0"], None)])
def test_thread_setting(clean_env, argv, expect):
    import os

    assert _threads.limit_from_argv(["--out", "x", *argv]) == expect
    for v in _threads.VARIABLES:
        assert os.environ.get(v) == (None if expect is None else str(expect))
