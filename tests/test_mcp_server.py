from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from sparkdocs.mcp_server import create_server  # noqa: E402
from sparkdocs.mcp_workspace import DocumentWorkspace  # noqa: E402

MCP_DISCOVERY_TIMEOUT_SECONDS = 5.0


async def _list_tools_with_timeout(server: object) -> object:
    return await asyncio.wait_for(
        server.list_tools(),  # type: ignore[attr-defined]
        timeout=MCP_DISCOVERY_TIMEOUT_SECONDS,
    )


def test_mcp_surface_exposes_only_bounded_document_tools(tmp_path: Path) -> None:
    server = create_server(DocumentWorkspace(tmp_path))

    tools = asyncio.run(_list_tools_with_timeout(server))

    assert {tool.name for tool in tools} == {
        "ingest_document",
        "search_document",
        "document_status",
        "clear_document",
    }
