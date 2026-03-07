import logging
import sys

# Configure logging FIRST before any other imports that might trigger config loading
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stderr)])

import argparse
import asyncio
import atexit
import json
import os
from inspect import Parameter, Signature
from typing import Any, Dict, Optional

import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

# ---------------------------------------------------------------------------
# NOTE: Heavy imports (app.logger, agents, tools) are deferred to avoid
# slowing down the Starlette/uvicorn startup. Railway's healthcheck fires
# within seconds of the process starting, so the /health route must be
# reachable before any LLM config or agent initialization occurs.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Auth Middleware
# ---------------------------------------------------------------------------

class BearerAuthMiddleware(BaseHTTPMiddleware):
    """
    Validates Bearer token on the /sse endpoint.
    If MCP_SERVER_AUTH_TOKEN is not set, auth is disabled (dev mode).
    """

    async def dispatch(self, request: Request, call_next):
        auth_token = os.getenv("MCP_SERVER_AUTH_TOKEN")

        # Only enforce auth if the token is configured
        if auth_token and request.url.path == "/sse":
            auth_header = request.headers.get("Authorization", "")
            if not auth_header:
                return JSONResponse(
                    {"error": "Missing Authorization header"}, status_code=401
                )
            parts = auth_header.split()
            if len(parts) != 2 or parts[0].lower() != "bearer" or parts[1] != auth_token:
                return JSONResponse(
                    {"error": "Invalid authentication credentials"}, status_code=401
                )

        return await call_next(request)


# ---------------------------------------------------------------------------
# Health Check — must be reachable IMMEDIATELY on startup
# ---------------------------------------------------------------------------

_server_ready = False

async def health_check(request: Request) -> JSONResponse:
    """Simple health endpoint for Railway's deployment checks."""
    return JSONResponse({
        "status": "ok",
        "service": "openmanus-mcp",
        "ready": _server_ready,
    })


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

class MCPServer:
    """
    OpenManus MCP Server.

    Exposes two layers of tools:
      1. High-level Agent Tools — run_manus, run_data_analysis, run_swe, run_browser
         Each spawns a fresh, isolated agent instance per call for full autonomy.
      2. Low-level Primitive Tools — bash, str_replace_editor, terminate
         Direct tool access for callers that need fine-grained control.

    Agent and tool imports are deferred until register_all_tools() is called
    so that the HTTP server (and /health endpoint) can start immediately.
    """

    def __init__(self, name: str = "openmanus"):
        port = int(os.getenv("PORT", os.getenv("FASTMCP_PORT", "8000")))
        self.server = FastMCP(name, port=port)
        self.tools: Dict[str, Any] = {}

    def _load_tools(self) -> None:
        """Deferred import and instantiation of all tools and agents."""
        # These imports are slow (LLM config, Playwright, etc.) — keep them here
        from app.logger import logger as _logger
        self._logger = _logger

        from app.tool.base import BaseTool
        from app.tool.bash import Bash
        from app.tool.str_replace_editor import StrReplaceEditor
        from app.tool.terminate import Terminate
        from app.agent.manus import Manus
        from app.agent.data_analysis import DataAnalysis
        from app.agent.swe import SWEAgent
        from app.agent.browser import BrowserAgent
        from app.mcp.agent_tool import AgentTool

        # --- High-level: Agent-as-a-Tool (stateless, isolated per call) ---
        self.tools["run_manus"] = AgentTool(agent_class=Manus)
        self.tools["run_data_analysis"] = AgentTool(agent_class=DataAnalysis)
        self.tools["run_swe"] = AgentTool(agent_class=SWEAgent)
        self.tools["run_browser"] = AgentTool(agent_class=BrowserAgent)

        # --- Low-level: Primitive tools for direct control ---
        self.tools["bash"] = Bash()
        self.tools["editor"] = StrReplaceEditor()
        self.tools["terminate"] = Terminate()

        _logger.info(f"Loaded {len(self.tools)} tools: {list(self.tools.keys())}")

    def register_tool(self, tool: Any, method_name: Optional[str] = None) -> None:
        """Register a tool with parameter validation and documentation."""
        logger = self._logger
        tool_name = method_name or tool.name
        tool_param = tool.to_param()
        tool_function = tool_param["function"]

        async def tool_method(**kwargs):
            logger.info(f"Executing {tool_name}: {kwargs}")
            result = await tool.execute(**kwargs)
            logger.info(f"Result of {tool_name}: {result}")
            if hasattr(result, "model_dump"):
                return json.dumps(result.model_dump())
            elif isinstance(result, dict):
                return json.dumps(result)
            return result

        tool_method.__name__ = tool_name
        tool_method.__doc__ = self._build_docstring(tool_function)
        tool_method.__signature__ = self._build_signature(tool_function)

        param_props = tool_function.get("parameters", {}).get("properties", {})
        required_params = tool_function.get("parameters", {}).get("required", [])
        tool_method._parameter_schema = {
            param_name: {
                "description": param_details.get("description", ""),
                "type": param_details.get("type", "any"),
                "required": param_name in required_params,
            }
            for param_name, param_details in param_props.items()
        }

        self.server.tool()(tool_method)
        logger.info(f"Registered tool: {tool_name}")

    def _build_docstring(self, tool_function: dict) -> str:
        description = tool_function.get("description", "")
        param_props = tool_function.get("parameters", {}).get("properties", {})
        required_params = tool_function.get("parameters", {}).get("required", [])
        docstring = description
        if param_props:
            docstring += "\n\nParameters:\n"
            for param_name, param_details in param_props.items():
                required_str = (
                    "(required)" if param_name in required_params else "(optional)"
                )
                param_type = param_details.get("type", "any")
                param_desc = param_details.get("description", "")
                docstring += (
                    f"    {param_name} ({param_type}) {required_str}: {param_desc}\n"
                )
        return docstring

    def _build_signature(self, tool_function: dict) -> Signature:
        param_props = tool_function.get("parameters", {}).get("properties", {})
        required_params = tool_function.get("parameters", {}).get("required", [])
        parameters = []
        for param_name, param_details in param_props.items():
            param_type = param_details.get("type", "")
            default = Parameter.empty if param_name in required_params else None
            annotation = Any
            if param_type == "string":
                annotation = str
            elif param_type == "integer":
                annotation = int
            elif param_type == "number":
                annotation = float
            elif param_type == "boolean":
                annotation = bool
            elif param_type == "object":
                annotation = dict
            elif param_type == "array":
                annotation = list
            param = Parameter(
                name=param_name,
                kind=Parameter.KEYWORD_ONLY,
                default=default,
                annotation=annotation,
            )
            parameters.append(param)
        return Signature(parameters=parameters)

    async def cleanup(self) -> None:
        if hasattr(self, "_logger"):
            self._logger.info("Cleaning up server resources")

    def register_all_tools(self) -> None:
        self._load_tools()
        for tool in self.tools.values():
            self.register_tool(tool)

    def _build_sse_app(self) -> Starlette:
        """
        Build a Starlette app wrapping FastMCP's SSE transport,
        adding the health check route and Bearer auth middleware.
        The /health route is registered FIRST so it responds immediately.
        """
        # Get the raw Starlette app from FastMCP (has /sse and /messages/ routes)
        fastmcp_app = self.server.sse_app()

        # Wrap it with our health check and auth middleware
        app = Starlette(
            debug=False,
            routes=[
                Route("/health", endpoint=health_check),
                Mount("/", app=fastmcp_app),
            ],
            middleware=[
                Middleware(BearerAuthMiddleware),
            ],
        )
        return app

    def run(self, transport: str = "stdio") -> None:
        """Run the MCP server in the specified transport mode."""
        global _server_ready
        atexit.register(lambda: asyncio.run(self.cleanup()))

        if transport == "sse":
            port = int(os.getenv("PORT", os.getenv("FASTMCP_PORT", "8000")))
            # Always bind to 0.0.0.0 in SSE mode so Railway's load balancer can reach us.
            # FastMCP defaults to 127.0.0.1 which is unreachable from outside the container.
            host = "0.0.0.0"

            # Build the Starlette app FIRST (fast — no heavy imports yet)
            # so uvicorn can start serving /health immediately.
            app = self._build_sse_app()

            # Register tools in a background task after the server is up
            # This ensures /health responds before the slow agent imports complete.
            async def _startup():
                global _server_ready
                import asyncio as _asyncio
                # Small delay to let uvicorn fully bind the port
                await _asyncio.sleep(0.5)
                logging.info("Loading tools and agents in background...")
                self.register_all_tools()
                _server_ready = True
                logging.info(
                    f"OpenManus MCP server ready on {host}:{port}\n"
                    f"  /health  — health check\n"
                    f"  /sse     — MCP endpoint (auth: {'enabled' if os.getenv('MCP_SERVER_AUTH_TOKEN') else 'disabled'})\n"
                    f"  Tools: {list(self.tools.keys())}"
                )

            # Add startup event to Starlette app
            app.add_event_handler("startup", _startup)

            logging.info(f"Starting uvicorn on {host}:{port} ...")
            uvicorn.run(app, host=host, port=port, log_level="info")
        else:
            # stdio mode — load everything synchronously (no healthcheck needed)
            self.register_all_tools()
            self.server.run(transport=transport)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenManus MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport mode: stdio (local) or sse (web/Railway)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    server = MCPServer()
    server.run(transport=args.transport)
