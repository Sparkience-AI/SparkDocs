from __future__ import annotations

import asyncio
from pathlib import Path
from threading import get_ident

import pytest

pytest.importorskip("mcp")

from sparkdocs.mcp_server import create_server  # noqa: E402
from sparkdocs.mcp_workspace import DocumentWorkspace, WorkspaceStatus  # noqa: E402

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


def test_mcp_ingest_runs_outside_the_transport_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = DocumentWorkspace(tmp_path)
    transport_thread = get_ident()

    def ingest(relative_path: str) -> WorkspaceStatus:
        assert relative_path == "sample.pdf"
        assert get_ident() != transport_thread
        return WorkspaceStatus(loaded=True)

    monkeypatch.setattr(workspace, "ingest", ingest)
    server = create_server(workspace)
    result = asyncio.run(server.call_tool("ingest_document", {"relative_path": "sample.pdf"}))
    assert result[1]["loaded"] is True
