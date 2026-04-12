# coding: utf-8
"""
Entry point for the OpenManus Hybrid MCP Server.
Launches the FastMCP server using Streamable HTTP transport (MCP 2025-03-26).
Disables DNS rebinding protection so Railway's proxy Host header is accepted.
Registers BearerAuthMiddleware for Fix 5 (auth enforcement).
"""
import logging
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", os.environ.get("FASTMCP_PORT", "8080")))
HOST = "0.0.0.0"

logger.info(f"[run_mcp_server] Starting Streamable HTTP on {HOST}:{PORT}")

from app.mcp.server import BearerAuthMiddleware, mcp


# Override host/port settings
mcp.settings.host = HOST
mcp.settings.port = PORT

# Disable DNS rebinding protection — Railway's edge proxy handles security.
try:
    from mcp.server.transport_security import TransportSecuritySettings

    mcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=False
    )
    logger.info("DNS rebinding protection disabled (Railway edge handles security)")
except (ImportError, AttributeError):
    logger.info("TransportSecuritySettings not available — skipping")

# Fix 5 — Register auth middleware on the underlying Starlette app
# Must be done after the app is built, so we build it first.
# Prefer Streamable HTTP (/mcp endpoint) — fall back to SSE for older mcp lib.
try:
    app = mcp.streamable_http_app()
    logger.info("Using Streamable HTTP transport (POST /mcp)")
except AttributeError:
    app = mcp.sse_app()
    logger.info("Falling back to SSE transport (mcp lib lacks streamable_http_app)")
app.add_middleware(BearerAuthMiddleware)

logger.info(f"Starting MCP server on {HOST}:{PORT} with Bearer auth enforcement")

import uvicorn


uvicorn.run(app, host=HOST, port=PORT, h11_max_incomplete_event_size=16384)
