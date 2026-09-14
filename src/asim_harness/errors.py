"""Structured failure extraction: turn a failed run's logs into runs/<id>/error.json.

What ActivitySim 1.4 leaves behind when a step fails (verified on manufactured
failures, see NOTES.md Phase 3):

- ``output/log/activitysim.log`` has ``#run_model running step <name>`` markers,
  then ``===== ERROR IN <step> =====`` / the message / a full traceback /
  ``===== / =====``, then ``activitysim run encountered an unrecoverable error``
  with the traceback again.
- Expression failures are logged before that as
  ``<trace_label> - <ExcType> (<msg>) evaluating: <expression>``,
  ``Variable evaluation failed <ExcType> (<msg>) evaluating: <expression>`` or
  ``assign_variables expression: <target> = <expression>``; spec-file problems
  as ``Error reading spec file: <path>``, ``read_model_spec error reading <path>``
  or ``Coefficient File Invalid: <path>``.
- The CLI prints the traceback to **stdout** (not stderr) and exits 99.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import ledger, paths
from .jsonio import read_json, write_json

ERROR_NAME = "error.json"
LOG_TAIL_LINES = 100
TRACEBACK_FRAMES = 15

STEP_MARKER = re.compile(r"#run_model running step (\S+)")
ERROR_IN = re.compile(r"===== ERROR IN (\S+) =====")
TRACEBACK_START = "Traceback (most recent call last):"
FRAME_LINE = re.compile(r'^\s+File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>.+)$')
EXCEPTION_LINE = re.compile(r"^(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|Warning|Exit|Interrupt|Fault|Stop|Denied|Found))(?:: (?P<msg>.*))?$")
EVAL_PATTERNS = (
    re.compile(r"(?P<label>[\w.\-]+) - (?P<exc>[\w.]+) \((?P<msg>.*)\) evaluating: (?P<expr>.+)$"),
    re.compile(r"Variable evaluation failed (?P<exc>[\w.]+) \((?P<msg>.*)\) evaluating: (?P<expr>.+)$"),
    re.compile(r"assign_variables expression: (?P<target>\S+) = (?P<expr>.+)$"),
)
SPEC_FILE_PATTERNS = (
    re.compile(r"Error reading spec file: (?P<path>.+)$"),
    re.compile(r"read_model_spec error reading (?P<path>.+)$"),
    re.compile(r"Coefficient (?:Template )?File Invalid: (?P<path>.+)$"),
)
MISSING_FILE = re.compile(r"file '(?P<name>[^']+)' not in \[")
SITE_PACKAGES = re.compile(r"^.*?/site-packages/")
CARET_LINE = re.compile(r"^[\s^~]+$")


def _strip_log_prefix(line: str) -> str:
    """'14/09/2026 18:25:30 - ERROR - root - message' -> 'message'."""
    parts = line.split(" - ", 3)
    if len(parts) == 4 and re.match(r"\d\d/\d\d/\d{4} ", parts[0]):
        return parts[3]
    return line


def _read_lines(path: Path | None) -> list[str]:
    if path is None or not Path(path).is_file():
        return []
    return Path(path).read_text(errors="replace").splitlines()


def last_traceback(lines: list[str]) -> tuple[list[str], str | None]:
    """The last traceback block in ``lines``: (block lines, final exception line)."""
    starts = [i for i, ln in enumerate(lines) if ln.strip() == TRACEBACK_START]
    if not starts:
        return [], None
    i = starts[-1]
    block = [lines[i]]
    exc_line = None
    for ln in lines[i + 1:]:
        if ln.startswith((" ", "\t")):
            block.append(ln)
            continue
        if ln.strip() == "":
            continue
        # first non-indented line after the frames is the exception line
        block.append(ln)
        exc_line = ln
        break
    return block, exc_line


def parse_exception_line(line: str | None) -> tuple[str | None, str | None]:
    if not line:
        return None, None
    line = _strip_log_prefix(line).strip()
    m = EXCEPTION_LINE.match(line)
    if m:
        return m.group("type"), (m.group("msg") or "").strip()
    if ": " in line:
        head, _, rest = line.partition(": ")
        if re.fullmatch(r"[A-Za-z_][\w.]*", head):
            return head, rest.strip()
    return None, line


def traceback_tail(block: list[str], frames: int = TRACEBACK_FRAMES) -> list[str]:
    """Last ``frames`` frames of a traceback block, paths shortened, plus the exception line."""
    if not block:
        return []
    frame_groups: list[list[str]] = []
    current: list[str] | None = None
    trailer: list[str] = []
    for ln in block[1:]:
        m = FRAME_LINE.match(ln)
        if m:
            path = SITE_PACKAGES.sub("site-packages/", m.group("file"))
            current = [f'File "{path}", line {m.group("line")}, in {m.group("func")}']
            frame_groups.append(current)
        elif CARET_LINE.match(ln):
            continue  # Python 3.11 error-location carets
        elif ln.startswith((" ", "\t")) and current is not None:
            current.append("    " + ln.strip())
        else:
            trailer.append(ln.strip())
    kept = frame_groups[-frames:]
    out: list[str] = []
    if len(frame_groups) > frames:
        out.append(f"... {len(frame_groups) - frames} earlier frame(s) omitted")
    for group in kept:
        out.extend(group)
    out.extend(trailer)
    return out


def expression_context(lines: list[str]) -> dict | None:
    """Spec file / expression details ActivitySim logged before the traceback, if any."""
    ctx: dict = {}
    for raw in lines:
        ln = _strip_log_prefix(raw)
        for pat in SPEC_FILE_PATTERNS:
            m = pat.search(ln)
            if m:
                ctx["spec_file"] = m.group("path").strip()
        for pat in EVAL_PATTERNS:
            m = pat.search(ln)
            if m:
                g = m.groupdict()
                ctx["expression"] = g["expr"].strip()
                if g.get("label"):
                    ctx["trace_label"] = g["label"]
                if g.get("target"):
                    ctx["target"] = g["target"]
                if g.get("exc"):
                    ctx["exception_type"] = g["exc"]
                    ctx["message"] = (g.get("msg") or "").strip()
        m = MISSING_FILE.search(ln)
        if m:
            ctx["missing_file"] = m.group("name")
    return ctx or None


def extract(run_dir: Path, exit_code: int | None = None) -> dict:
    """Build the error record for a run directory from its log, stdout and stderr."""
    run_dir = Path(run_dir)
    output_dir = run_dir / "output"
    log_path = paths.find_log_file(output_dir)
    log_lines = _read_lines(log_path)
    stdout_lines = _read_lines(run_dir / "stdout.log")
    stderr_lines = [ln for ln in _read_lines(run_dir / "stderr.log") if ln.strip() and "pkg_resources" not in ln]

    failed_step = None
    m = [ERROR_IN.search(ln) for ln in log_lines]
    m = [x for x in m if x]
    if m:
        failed_step = m[-1].group(1)
    else:
        steps = [STEP_MARKER.search(ln) for ln in log_lines]
        steps = [x for x in steps if x]
        if steps:
            failed_step = steps[-1].group(1)

    block, exc_line = last_traceback(log_lines)
    source = "log"
    if not block:
        block, exc_line = last_traceback(stdout_lines)
        source = "stdout"
    if not block:
        block, exc_line = last_traceback(stderr_lines)
        source = "stderr"
    exc_type, message = parse_exception_line(exc_line)

    ctx = expression_context(log_lines) or expression_context(stdout_lines)
    if exc_type is None and ctx and ctx.get("exception_type"):
        exc_type, message = ctx["exception_type"], ctx.get("message")
    if message is None:
        # message logged right after the ERROR IN marker, when there is no traceback at all
        for i, ln in enumerate(log_lines):
            if ERROR_IN.search(ln) and i + 1 < len(log_lines):
                message = _strip_log_prefix(log_lines[i + 1]).strip()
        if message is None and stderr_lines:
            message = stderr_lines[-1].strip()

    return {
        "failed_step": failed_step,
        "exception_type": exc_type,
        "message": message,
        "traceback_tail": traceback_tail(block),
        "traceback_source": source if block else None,
        "expression_context": ctx,
        "exit_code": exit_code,
        "log_path": str(log_path) if log_path else None,
        "log_tail": log_lines[-LOG_TAIL_LINES:],
        "stderr_tail": stderr_lines[-10:],
    }


def error_path(run_id: str) -> Path:
    return paths.run_dir(run_id) / ERROR_NAME


def finalize_hook(run_dir: Path, manifest: dict) -> None:
    """Called by the runner after every run: write error.json for failed runs."""
    if manifest.get("status") != "failed":
        return
    record = extract(run_dir, manifest.get("exit_code"))
    record["run_id"] = manifest.get("run_id")
    record["override_files"] = manifest.get("override_files") or []
    write_json(Path(run_dir) / ERROR_NAME, record)


def load_error(run_id: str) -> dict | None:
    run_id = ledger.resolve_run_id(run_id)
    path = error_path(run_id)
    return read_json(path) if path.exists() else None


def error_for_run(run_id: str, force: bool = False) -> dict | None:
    """error.json for a run, extracting it on demand for failed runs that lack one."""
    run_id = ledger.resolve_run_id(run_id)
    path = error_path(run_id)
    if path.exists() and not force:
        return read_json(path)
    manifest = read_json(paths.run_dir(run_id) / "manifest.json")
    if manifest.get("status") != "failed":
        return None
    record = extract(paths.run_dir(run_id), manifest.get("exit_code"))
    record["run_id"] = run_id
    record["override_files"] = manifest.get("override_files") or []
    write_json(path, record)
    return record


def compact(record: dict, log_tail_lines: int = 0) -> dict:
    """Sized for a tool result: everything but the 100-line log tail (opt in with log_tail_lines)."""
    out = {k: v for k, v in record.items() if k not in ("log_tail",)}
    if log_tail_lines:
        out["log_tail"] = record.get("log_tail", [])[-log_tail_lines:]
    return out


def text(record: dict, log_tail_lines: int = 15) -> str:
    lines = [
        f"run {record.get('run_id')} failed in step: {record.get('failed_step') or 'unknown'}",
        f"exception: {record.get('exception_type') or 'unknown'}: {record.get('message') or ''}",
    ]
    if record.get("override_files"):
        lines.append(f"override files in this run: {', '.join(record['override_files'])}")
    ctx = record.get("expression_context") or {}
    if ctx:
        for key in ("spec_file", "missing_file", "trace_label", "target", "expression"):
            if ctx.get(key):
                lines.append(f"  {key}: {ctx[key]}")
    tb = record.get("traceback_tail") or []
    if tb:
        lines.append(f"traceback (last {TRACEBACK_FRAMES} frames, from {record.get('traceback_source')}):")
        lines.extend("  " + ln for ln in tb)
    if record.get("stderr_tail"):
        lines.append("stderr tail:")
        lines.extend("  " + ln for ln in record["stderr_tail"][-5:])
    lines.append(f"log: {record.get('log_path')}")
    tail = record.get("log_tail") or []
    if log_tail_lines and tail:
        lines.append(f"log tail (last {min(log_tail_lines, len(tail))} of {len(tail)} kept lines):")
        lines.extend("  " + ln for ln in tail[-log_tail_lines:])
    return "\n".join(lines)
