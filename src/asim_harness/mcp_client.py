"""A tiny stdio client for the harness's own MCP server: ``asim tool <name> [json-args]``.

It exists so the tools can be exercised from a shell (debugging, CI, and agent
sessions that have no MCP client of their own) with exactly the results an MCP
client would see. Every call is appended to ``runs/.tool-calls.log``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from . import paths

CALL_LOG_NAME = ".tool-calls.log"


def _server_params() -> StdioServerParameters:
    env = dict(os.environ)
    env.setdefault("ASIM_HARNESS_ROOT", str(paths.root()))
    return StdioServerParameters(command=sys.executable, args=["-m", "asim_harness.cli", "mcp"], env=env,
                                 cwd=str(paths.root()))


def _log_call(name: str, arguments: dict, is_error: bool, size: int) -> None:
    try:
        log = paths.runs_dir() / CALL_LOG_NAME
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as f:
            f.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": name,
                                "arguments": arguments, "is_error": is_error, "result_bytes": size}) + "\n")
    except OSError:
        pass


async def _list_tools() -> list[dict[str, Any]]:
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [{"name": t.name, "description": (t.description or "").strip()} for t in result.tools]


async def _call(name: str, arguments: dict, timeout: float) -> tuple[bool, Any]:
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments, read_timeout_seconds=timeout)
            is_error = bool(getattr(result, "is_error", False))
            structured = getattr(result, "structured_content", None)
            if is_error or structured is None:
                text = "\n".join(getattr(c, "text", "") for c in (result.content or []))
                return is_error, text
            return False, structured


def list_tools() -> list[dict[str, Any]]:
    return asyncio.run(_list_tools())


def call_tool(name: str, arguments: dict | None = None, timeout: float = 3600.0) -> tuple[bool, Any]:
    """Return (is_error, payload). payload is the structured result, or the error/text content."""
    arguments = arguments or {}
    is_error, payload = asyncio.run(_call(name, arguments, timeout))
    _log_call(name, arguments, is_error, len(json.dumps(payload, default=str)))
    return is_error, payload
