from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from sparkdocs.mcp_server import create_server  # noqa: E402
from sparkdocs.mcp_workspace import DocumentWorkspace  # noqa: E402


def test_mcp_surface_exposes_only_bounded_document_tools(tmp_path: Path) -> None:
    server = create_server(DocumentWorkspace(tmp_path))

    tools = asyncio.run(server.list_tools())

    assert {tool.name for tool in tools} == {
        "ingest_document",
        "search_document",
        "document_status",
        "clear_document",
    }
