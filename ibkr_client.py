"""MCP client for IBKR HTTP server on fragserv."""

import asyncio
import logging

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from config import IBKR_MCP_URL

log = logging.getLogger(__name__)


async def _call_ibkr_tool(tool_name: str, arguments: dict | None = None) -> str:
    """Open a streamable-http session, call one tool, return text result."""
    transport_kwargs = {
        "url": IBKR_MCP_URL,
        "httpx_client_factory": lambda: httpx.AsyncClient(timeout=10),
    }
    async with streamablehttp_client(**transport_kwargs) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments or {})
            # Extract text from content blocks
            parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
            return "\n".join(parts)


def call_ibkr_tool(tool_name: str, arguments: dict | None = None) -> str | None:
    """Sync wrapper — returns None on any error."""
    try:
        return asyncio.run(_call_ibkr_tool(tool_name, arguments))
    except Exception as e:
        log.warning("IBKR call failed (%s): %s", tool_name, e)
        return None
