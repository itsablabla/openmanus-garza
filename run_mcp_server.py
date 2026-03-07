# coding: utf-8
"""
Entry point for the OpenManus Hybrid MCP Server.
Launches the FastMCP server using SSE transport via uvicorn directly,
with proxy headers support for Railway's reverse proxy.
"""
import asyncio
import os

if __name__ == "__main__":
    import uvicorn
    from app.mcp.server import mcp

    port = int(os.environ.get("PORT", os.environ.get("FASTMCP_PORT", "8080")))
    host = "0.0.0.0"

    print(f"[run_mcp_server] Starting SSE on {host}:{port}", flush=True)

    # Get the Starlette app from FastMCP (includes custom_route endpoints)
    starlette_app = mcp.sse_app()

    uvicorn.run(
        starlette_app,
        host=host,
        port=port,
        proxy_headers=True,          # Trust X-Forwarded-* headers from Railway
        forwarded_allow_ips="*",     # Allow all IPs to set forwarded headers
        log_level="info",
    )
