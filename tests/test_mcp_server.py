"""The MCP server, driven through a real stdio client against a temp runs directory."""

from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from asim_harness import manifest as mf
from asim_harness.jsonio import write_json

EXPECTED_TOOLS = {
    "list_runs", "run_model", "get_run", "summarize_run", "check_targets", "compare_runs",
    "get_error", "get_log_tail", "read_config", "list_configs",
}


def _seed_run(tmp_root, run_id, status="succeeded", summary=None, error=None):
    d = tmp_root / "runs" / run_id
    (d / "output" / "log").mkdir(parents=True)
    mf.write_manifest(d, {"run_id": run_id, "label": f"label {run_id}", "status": status, "exit_code": 0 if status == "succeeded" else 99,
                          "created_at": run_id, "settings_overrides": {"inherit_settings": True}, "sample_size": 500,
                          "step_timings": {"trip_mode_choice": 1.0}})
    if summary is not None:
        write_json(d / "summary.json", summary)
    if error is not None:
        write_json(d / "error.json", error)
    (d / "output" / "log" / "activitysim.log").write_text("line1\nline2\nline3\n")
    return d


def _summary(run_id, walk=0.6):
    return {"run_id": run_id, "counts": {"households": 10, "persons": 20, "tours": 30, "trips": 100,
                                         "tours_by_purpose": {"work": 30}, "trips_by_purpose": {"work": 100}},
            "auto_ownership_share": {"0": 0.5, "1": 0.5}, "cdap_share": {"M": 1.0},
            "tour_mode_share": {"overall": {"WALK": walk, "BIKE": round(1 - walk, 6)}, "by_purpose": {"work": {"WALK": walk, "BIKE": round(1 - walk, 6)}}},
            "trip_mode_share": {"overall": {"WALK": walk, "BIKE": round(1 - walk, 6)}, "by_purpose": {"work": {"WALK": walk, "BIKE": round(1 - walk, 6)}}},
            "tour_frequency": {"mandatory": {"0": 1.0}, "non_mandatory": {"0": 1.0}}, "checks": {"all_passed": True}}


async def _with_session(env, fn):
    params = StdioServerParameters(command=sys.executable, args=["-m", "asim_harness.cli", "mcp"], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await fn(session)


def _call(session, name, args):
    return session.call_tool(name, args)


@pytest.fixture
def server_env(tmp_root):
    env = dict(os.environ)
    env.update({k: os.environ[k] for k in ("ASIM_HARNESS_ROOT", "ASIM_EXAMPLE_DIR", "ASIM_RUNS_DIR")})
    return env


def test_tools_and_read_only_calls(tmp_root, server_env):
    a, b = "20260101-000000-aaaaaa", "20260102-000000-bbbbbb"
    _seed_run(tmp_root, a, summary=_summary(a))
    _seed_run(tmp_root, b, summary=_summary(b, walk=0.7))
    _seed_run(tmp_root, "20260103-000000-cccccc", status="failed",
              error={"run_id": "20260103-000000-cccccc", "failed_step": "trip_mode_choice", "exception_type": "NameError",
                     "message": "boom", "traceback_tail": [], "expression_context": None, "log_tail": ["x"] * 50,
                     "log_path": "p", "stderr_tail": [], "exit_code": 99})
    from asim_harness import ledger, targets
    ledger.reindex()
    targets.bootstrap(a)

    async def scenario(s):
        out = {}
        tools = await s.list_tools()
        out["tools"] = {t.name for t in tools.tools}
        out["run_model_desc"] = next(t.description for t in tools.tools if t.name == "run_model")
        r = await _call(s, "list_runs", {"limit": 2})
        out["list_runs"] = r.structured_content
        r = await _call(s, "get_run", {"run_id": "20260103"})
        out["get_run_failed"] = r.structured_content
        r = await _call(s, "summarize_run", {"run_id": "20260101", "detail": "overall"})
        out["summarize"] = r.structured_content
        r = await _call(s, "check_targets", {"run_id": "20260102"})
        out["check"] = r.structured_content
        r = await _call(s, "get_run", {"run_id": "20260102"})
        out["get_run_scored"] = r.structured_content
        r = await _call(s, "check_targets", {"run_id": "20260102", "metric": "trip_mode_share.overall"})
        out["check_one"] = r.structured_content
        r = await _call(s, "check_targets", {"run_id": "20260102", "metric": "nope"})
        out["check_bad_metric"] = (r.is_error, r.content[0].text)
        r = await _call(s, "compare_runs", {"run_a": "20260101", "run_b": "20260102"})
        out["compare"] = r.structured_content
        r = await _call(s, "get_error", {"run_id": "20260103", "log_tail_lines": 5})
        out["error"] = r.structured_content
        r = await _call(s, "get_error", {"run_id": "20260101"})
        out["no_error"] = r.structured_content
        r = await _call(s, "get_log_tail", {"run_id": "20260103", "lines": 2})
        out["log_tail"] = r.structured_content
        r = await _call(s, "list_configs", {})
        out["configs"] = r.structured_content
        r = await _call(s, "read_config", {"relative_path": "trip_mode_choice.yaml"})
        out["read"] = r.structured_content
        r = await _call(s, "read_config", {"relative_path": "../settings.yaml"})
        out["escape"] = (r.is_error, r.content[0].text)
        r = await _call(s, "read_config", {"relative_path": "/etc/passwd"})
        out["absolute"] = (r.is_error, r.content[0].text)
        r = await _call(s, "get_run", {"run_id": "nope"})
        out["unknown_run"] = (r.is_error, r.content[0].text)
        r = await _call(s, "run_model", {"label": "   ", "wait": False})
        out["bad_label"] = (r.is_error, r.content[0].text)
        r = await _call(s, "run_model", {"label": "x", "resume_from": "20260101", "wait": False})
        out["bad_resume"] = (r.is_error, r.content[0].text)
        return out

    out = asyncio.run(_with_session(server_env, scenario))
    assert out["tools"] == EXPECTED_TOOLS
    assert "wait=false" in out["run_model_desc"]
    assert [r["run_id"] for r in out["list_runs"]["runs"]] == ["20260103-000000-cccccc", b]
    assert out["list_runs"]["runs"][1]["scorecard_pass"] is None
    assert out["get_run_failed"]["run"]["status"] == "failed"
    assert out["get_run_failed"]["error"]["failed_step"] == "trip_mode_choice"
    assert len(out["get_run_failed"]["error"]["log_tail"]) == 20
    assert "failed in step trip_mode_choice" in out["get_run_failed"]["explanation"]
    assert "slowest_steps" in out["get_run_failed"]["run"] and "step_timings" not in out["get_run_failed"]["run"]
    assert "by_purpose" not in out["summarize"]["trip_mode_share"] and "trips_by_purpose" not in out["summarize"]["counts"]
    assert out["check"]["passed"] is False and out["check"]["failed"][0]["metric"].endswith("mode_share.overall") or out["check"]["n_failed"] == 4
    assert out["check_one"]["detail"] == "all" and len(out["check_one"]["failed"]) == 1
    assert out["check_one"]["failed"][0]["categories"]["WALK"] == {"delta": pytest.approx(0.1), "share": 0.7, "target": 0.6}
    assert out["check_bad_metric"][0] is True and "no metric" in out["check_bad_metric"][1]
    assert out["get_run_scored"]["scorecard"]["detail"] == "summary"
    assert all("categories" not in r for r in out["get_run_scored"]["scorecard"]["failed"])
    assert out["compare"]["n_drifted"] == 4 and out["compare"]["drifted"][0]["deltas"]["WALK"] == pytest.approx(0.1)
    assert out["error"]["exception_type"] == "NameError" and len(out["error"]["log_tail"]) == 5
    assert out["no_error"]["error"] is None
    assert out["log_tail"]["lines"] == ["line2", "line3"] and out["log_tail"]["total_lines"] == 3
    assert {f["path"] for f in out["configs"]["files"]} >= {"settings.yaml", "trip_mode_choice.yaml"}
    assert out["read"]["content"].startswith("SPEC:") and out["read"]["truncated"] is False
    assert out["escape"][0] is True and "escapes" in out["escape"][1]
    assert out["absolute"][0] is True and "relative" in out["absolute"][1]
    assert out["unknown_run"][0] is True and "no run matches" in out["unknown_run"][1]
    assert out["bad_label"][0] is True and "label" in out["bad_label"][1]
    assert out["bad_resume"][0] is True and "resume-after" in out["bad_resume"][1]
    # every result above is small
    for key in ("list_runs", "get_run_failed", "summarize", "check", "compare", "error", "log_tail", "configs", "read"):
        assert len(json.dumps(out[key])) < 20000, key
