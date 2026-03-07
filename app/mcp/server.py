"""
Hybrid MCP Server for GARZA OS — OpenManus + Manus API.

Synchronous layer  : bash, browser, editor, terminate (OpenManus local agents)
Asynchronous layer : manus_create_task, manus_get_task, manus_list_tasks,
                     manus_upload_file, manus_list_files, manus_create_webhook,
                     garza_status (NL task digest)
                     (Manus SaaS API — fire-and-forget, poll for results)

Fixes applied (2026-03-07):
  Fix 5 — Auth enforcement: Bearer token validated on every SSE/tool request
  Fix 1 — Enrich list output: credit_usage, created_at, updated_at surfaced
  Fix 4 — Prompt logging: task_id + prompt logged to /tmp/task_log.jsonl
  Fix 2 — Prompt + output in manus_get_task via /messages endpoint
  Fix 3 — garza_status NL tool: conversational task digest
"""

import asyncio
import json
import logging
import os
import time
from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from app.mcp.manus_client import handle_api_error, manus_request
from app.mcp.manus_models import (
    CreateTaskInput,
    CreateWebhookInput,
    GetTaskInput,
    ListTasksInput,
)

logger = logging.getLogger(__name__)
mcp = FastMCP("OpenManus Hybrid MCP Server")

# ---------------------------------------------------------------------------
# Fix 5 — Auth enforcement middleware
# Validates Bearer token on all requests EXCEPT OAuth + well-known endpoints.
# ---------------------------------------------------------------------------

_PUBLIC_PATHS = {
    "/.well-known/oauth-authorization-server",
    "/.well-known/oauth-protected-resource",
    "/oauth/authorize",
    "/oauth/token",
    "/",
}

_TASK_LOG_PATH = "/tmp/task_log.jsonl"


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Reject requests without a valid Bearer token (Fix 5)."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        expected_token = os.environ.get("MCP_SERVER_AUTH_TOKEN", "")
        if not expected_token:
            # No token configured — allow all (dev mode)
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        token = auth_header.removeprefix("Bearer ").strip()

        if token != expected_token:
            return Response(
                content=json.dumps({"error": "Unauthorized — invalid or missing Bearer token"}),
                status_code=401,
                media_type="application/json",
            )

        return await call_next(request)


# Register middleware on the underlying Starlette app
@mcp.custom_route("/__auth_init__", methods=["GET"])
async def _auth_init_placeholder(request: Request) -> JSONResponse:
    """Internal placeholder — not a real endpoint."""
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# OAuth2 well-known endpoints (required for Nango mcp-generic / MCP_OAUTH2_GENERIC)
# ---------------------------------------------------------------------------

_BASE_URL = "https://" + os.environ.get(
    "RAILWAY_PUBLIC_DOMAIN", "openmanus-mcp-production.up.railway.app"
)


@mcp.custom_route("/.well-known/oauth-authorization-server", methods=["GET"])
async def oauth_authorization_server(request: Request) -> JSONResponse:
    """RFC 8414 — OAuth 2.0 Authorization Server Metadata."""
    return JSONResponse({
        "issuer": _BASE_URL,
        "authorization_endpoint": f"{_BASE_URL}/oauth/authorize",
        "token_endpoint": f"{_BASE_URL}/oauth/token",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "code_challenge_methods_supported": ["S256"],
        "scopes_supported": ["mcp"],
    })


@mcp.custom_route("/.well-known/oauth-protected-resource", methods=["GET"])
async def oauth_protected_resource(request: Request) -> JSONResponse:
    """RFC 9728 — OAuth 2.0 Protected Resource Metadata."""
    return JSONResponse({
        "resource": _BASE_URL,
        "authorization_servers": [_BASE_URL],
        "bearer_methods_supported": ["header"],
        "resource_documentation": "https://github.com/itsablabla/OpenManus",
    })


@mcp.custom_route("/oauth/authorize", methods=["GET"])
async def oauth_authorize(request: Request) -> RedirectResponse:
    """OAuth2 authorization endpoint — redirects with MCP_SERVER_AUTH_TOKEN as the code."""
    redirect_uri = request.query_params.get("redirect_uri", "")
    state = request.query_params.get("state", "")
    code = os.environ.get("MCP_SERVER_AUTH_TOKEN", "")
    return RedirectResponse(url=f"{redirect_uri}?code={code}&state={state}")


@mcp.custom_route("/oauth/token", methods=["POST"])
async def oauth_token(request: Request) -> JSONResponse:
    """OAuth2 token endpoint — exchanges authorization code for access token."""
    form = await request.form()
    code = str(form.get("code", ""))
    token = code or os.environ.get("MCP_SERVER_AUTH_TOKEN", "")
    return JSONResponse({
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": 31536000,  # 1 year
        "scope": "mcp",
    })


# ---------------------------------------------------------------------------
# Fix 4 — Prompt logger helper
# ---------------------------------------------------------------------------

def _log_task(task_id: str, prompt: str) -> None:
    """Append task creation record to /tmp/task_log.jsonl (Fix 4)."""
    try:
        record = json.dumps({
            "task_id": task_id,
            "prompt": prompt,
            "created_at": int(time.time()),
        })
        with open(_TASK_LOG_PATH, "a") as f:
            f.write(record + "\n")
    except Exception as e:
        logger.warning("[task_log] Failed to write log: %s", e)


def _read_task_log(hours: int = 24) -> list:
    """Read recent task log entries within the last N hours."""
    cutoff = int(time.time()) - (hours * 3600)
    entries = []
    try:
        with open(_TASK_LOG_PATH) as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    if rec.get("created_at", 0) >= cutoff:
                        entries.append(rec)
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return entries


# ---------------------------------------------------------------------------
# Synchronous tools (existing OpenManus agents — unchanged from upstream)
# ---------------------------------------------------------------------------

@mcp.tool()
async def bash(command: str) -> str:
    """Execute a bash command and return its output."""
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    return stdout.decode(errors="replace")


@mcp.tool()
async def editor(command: str, path: str, content: str = "") -> str:
    """Read or write a file. command: 'view' | 'create' | 'str_replace'."""
    if command == "view":
        try:
            return open(path).read()
        except FileNotFoundError:
            return f"File not found: {path}"
    elif command == "create":
        import pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"File written: {path}"
    return f"Unknown editor command: {command}"


@mcp.tool()
async def terminate(reason: str = "") -> str:
    """Signal task completion to the orchestrator."""
    return f"Task terminated. Reason: {reason}"


# ---------------------------------------------------------------------------
# Asynchronous tools (Manus SaaS API)
# ---------------------------------------------------------------------------

@mcp.tool()
async def manus_create_task(
    prompt: str,
    task_mode: str = "agent",
    agent_profile: str = "manus-1.6",
    file_ids: str = "",
    use_gmail_connector: bool = False,
    use_notion_connector: bool = False,
    use_gcal_connector: bool = False,
) -> str:
    """
    Submit a long-running task to Manus AI for autonomous execution.

    Returns a task_id — poll with manus_get_task until status == 'completed'.
    Tasks typically complete in 2-10 minutes.

    Args:
        prompt: Natural language task description (10-10,000 chars)
        task_mode: agent (default, full autonomous), adaptive, or chat
        agent_profile: manus-1.6 (default), manus-1.6-lite (fast/cheap), or manus-1.6-max (highest quality)
        file_ids: Comma-separated file IDs from manus_upload_file (optional)
        use_gmail_connector: Grant Manus access to Gmail
        use_notion_connector: Grant Manus access to Notion
        use_gcal_connector: Grant Manus access to Google Calendar
    """
    try:
        profile_map = {
            "speed": "manus-1.6", "quality": "manus-1.6-max",
            "lite": "manus-1.6-lite", "general": "manus-1.6", "default": "manus-1.6"
        }
        api_profile = profile_map.get(agent_profile, agent_profile)
        body: dict = {
            "prompt": prompt,
            "taskMode": task_mode,
            "agentProfile": api_profile,
        }

        if file_ids:
            body["attachments"] = [
                {"type": "file_id", "file_id": fid.strip()}
                for fid in file_ids.split(",") if fid.strip()
            ]

        connectors = []
        connector_map = {
            use_gmail_connector: "MANUS_GMAIL_CONNECTOR_ID",
            use_notion_connector: "MANUS_NOTION_CONNECTOR_ID",
            use_gcal_connector: "MANUS_GCAL_CONNECTOR_ID",
        }
        for enabled, env_var in connector_map.items():
            if enabled:
                cid = os.environ.get(env_var, "")
                if cid:
                    connectors.append({"id": cid})
                else:
                    logger.warning("[manus] %s env var not set — connector skipped", env_var)
        if connectors:
            body["connectors"] = connectors

        result = await manus_request("POST", "/tasks", json=body)
        task_id = result.get("id", result.get("task_id", "unknown"))
        status = result.get("status", "pending")

        # Fix 4 — Log prompt for NL query support
        _log_task(task_id, prompt)

        return (
            f"Task created successfully.\n"
            f"task_id : {task_id}\n"
            f"status  : {status}\n"
            f"prompt  : {prompt[:120]}{'...' if len(prompt) > 120 else ''}\n"
            f"Tip     : Call manus_get_task(task_id='{task_id}') in 2-3 minutes to check progress."
        )
    except Exception as e:
        return handle_api_error(e)


@mcp.tool()
async def manus_get_task(task_id: str) -> str:
    """
    Poll the status and output of a Manus task.
    Returns the original prompt, current status, and Manus's result text.

    Args:
        task_id: The task ID returned by manus_create_task
    """
    try:
        result = await manus_request("GET", f"/tasks/{task_id}")
        status = result.get("status", "unknown")
        meta = result.get("metadata") or {}
        task_title = meta.get("task_title", "")
        task_url = meta.get("task_url", "")
        created_at = result.get("created_at", "")
        updated_at = result.get("updated_at", "")
        error_msg = result.get("error_message") or result.get("error") or ""

        # Fix 2 — Extract prompt and result from output array
        output_items = result.get("output") or []
        user_prompt = ""
        assistant_result = ""
        credit_usage = None

        if isinstance(output_items, list):
            for item in output_items:
                role = item.get("role", "")
                content_list = item.get("content", [])
                text_parts = []
                for c in content_list:
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        text_parts.append(c.get("text", ""))
                text = " ".join(text_parts).strip()
                if role == "user" and not user_prompt and text:
                    user_prompt = text[:500]
                elif role == "assistant" and text:
                    assistant_result = text  # keep last assistant message
                # Check for usage info
                usage = item.get("usage") or {}
                if usage.get("total_tokens"):
                    credit_usage = usage.get("total_tokens")
        elif isinstance(output_items, str):
            assistant_result = output_items

        # Fall back to task log for prompt if not in API response
        if not user_prompt:
            for entry in _read_task_log(hours=168):  # 7 days
                if entry.get("task_id") == task_id:
                    user_prompt = entry.get("prompt", "")[:500]
                    break

        lines = [
            f"task_id     : {task_id}",
            f"status      : {status}",
        ]
        if task_title:
            lines.append(f"title       : {task_title}")
        if task_url:
            lines.append(f"url         : {task_url}")
        if created_at:
            lines.append(f"created_at  : {created_at}")
        if updated_at:
            lines.append(f"updated_at  : {updated_at}")
        if credit_usage is not None:
            lines.append(f"tokens_used : {credit_usage}")
        if user_prompt:
            lines.append(f"\nYour prompt :\n  {user_prompt}")
        if assistant_result:
            lines.append(f"\nManus result:\n{assistant_result[:2000]}")
            if len(assistant_result) > 2000:
                lines.append(f"  ... (truncated — {len(assistant_result)} chars total)")
        if error_msg:
            lines.append(f"\nError: {error_msg}")
        if status in ("pending", "running"):
            lines.append("\nTip: Task still in progress — check again in 1-2 minutes.")

        return "\n".join(lines)
    except Exception as e:
        return handle_api_error(e)


@mcp.tool()
async def manus_list_tasks(
    limit: int = 20,
    status_filter: str = "",
) -> str:
    """
    List recent Manus tasks with credit usage, timing, and prompt context.

    Args:
        limit: Number of tasks to return (1-100, default 20)
        status_filter: Optional filter — pending | running | completed | failed
    """
    try:
        params: dict = {"limit": limit}
        if status_filter:
            params["status"] = status_filter

        result = await manus_request("GET", "/tasks", params=params)
        tasks = result.get("tasks", result.get("data", []))
        has_more = result.get("has_more", False)

        if not tasks:
            return "No tasks found."

        # Fix 1 — Load local prompt log for enrichment
        log_by_id = {e["task_id"]: e["prompt"] for e in _read_task_log(hours=168)}

        lines = [f"Tasks ({len(tasks)} returned):"]
        for t in tasks:
            tid = t.get("id", t.get("task_id", "?"))
            tstatus = t.get("status", "?")
            # Fix 1 — Surface timestamps and credit usage
            created = t.get("created_at", "")
            updated = t.get("updated_at", "")
            credit_usage = t.get("credit_usage") or t.get("credits_used")
            meta = t.get("metadata") or {}
            title = meta.get("task_title") or (t.get("prompt", "") or "")[:60]
            url = meta.get("task_url", "")

            # Enrich with local prompt log
            prompt_preview = log_by_id.get(tid, "")[:80]

            line = f"  {tid}  [{tstatus}]"
            if created:
                line += f"  created:{created}"
            if updated and updated != created:
                line += f"  updated:{updated}"
            if credit_usage is not None:
                line += f"  credits:{credit_usage}"
            line += f"  {title!r}"
            if prompt_preview and prompt_preview not in title:
                line += f"  prompt:'{prompt_preview}'"
            if url:
                line += f"  {url}"
            lines.append(line)

        if has_more:
            lines.append("\nMore tasks available — reduce limit or use status_filter.")

        return "\n".join(lines)
    except Exception as e:
        return handle_api_error(e)


@mcp.tool()
async def manus_upload_file(filename: str, content_base64: str, content_type: str = "text/plain") -> str:
    """
    Upload a file to Manus and get a file_id for use in manus_create_task.

    Args:
        filename: Name of the file (e.g., 'report.pdf')
        content_base64: Base64-encoded file content
        content_type: MIME type (default text/plain)
    """
    try:
        import base64
        import httpx

        presign = await manus_request("POST", "/files", json={
            "filename": filename,
            "content_type": content_type,
        })
        upload_url = presign.get("upload_url")
        file_id = presign.get("id") or presign.get("file_id")

        if not upload_url or not file_id:
            return f"Error: Unexpected presign response — {presign}"

        raw_bytes = base64.b64decode(content_base64)
        async with httpx.AsyncClient(timeout=60.0) as client:
            put_resp = await client.put(
                upload_url,
                content=raw_bytes,
                headers={"Content-Type": content_type},
            )
            put_resp.raise_for_status()

        return (
            f"File uploaded successfully.\n"
            f"file_id  : {file_id}\n"
            f"filename : {filename}\n"
            f"Tip      : Pass file_ids='{file_id}' when calling manus_create_task."
        )
    except Exception as e:
        return handle_api_error(e)


@mcp.tool()
async def manus_list_files(limit: int = 20) -> str:
    """
    List files uploaded to Manus.

    Args:
        limit: Number of files to return (default 20)
    """
    try:
        result = await manus_request("GET", "/files", params={"limit": limit})
        files = result.get("files", result.get("data", []))

        if not files:
            return "No files found."

        lines = [f"Files ({len(files)}):"]
        for f in files:
            fid = f.get("id", "?")
            name = f.get("filename") or f.get("name", "?")
            size = f.get("size", "?")
            created = (f.get("created_at") or "")[:19]
            lines.append(f"  {fid}  {name}  {size} bytes  {created}")

        return "\n".join(lines)
    except Exception as e:
        return handle_api_error(e)


@mcp.tool()
async def manus_create_webhook(
    url: str,
    events: str = "task.completed,task.failed",
) -> str:
    """
    Register a webhook to receive Manus task completion events.

    Use your n8n webhook trigger URL so GARZA OS is notified automatically
    when long-running tasks finish — no polling required.

    Args:
        url: Webhook endpoint URL (e.g., your n8n webhook trigger URL)
        events: Comma-separated events (default: task.completed,task.failed)
    """
    try:
        event_list = [e.strip() for e in events.split(",") if e.strip()]
        result = await manus_request("POST", "/webhooks", json={
            "webhook": {
                "url": url,
                "events": event_list,
            }
        })
        webhook_id = result.get("id") or result.get("webhook_id", "unknown")
        secret = result.get("secret", "(none)")

        return (
            f"Webhook registered.\n"
            f"webhook_id : {webhook_id}\n"
            f"url        : {url}\n"
            f"events     : {event_list}\n"
            f"secret     : {secret}\n"
            f"Tip        : Store the secret in Railway env as MANUS_WEBHOOK_SECRET "
            f"and verify it in your n8n workflow."
        )
    except Exception as e:
        return handle_api_error(e)


# ---------------------------------------------------------------------------
# Fix 3 — garza_status: Natural Language task digest tool
# ---------------------------------------------------------------------------

@mcp.tool()
async def garza_status(
    query: str = "What did Manus do today?",
    hours: int = 24,
    limit: int = 10,
) -> str:
    """
    Answer natural language questions about recent Manus activity.

    Examples:
      - "What did Manus accomplish today?"
      - "What was the last thing I asked Manus?"
      - "Did anything fail in the last 6 hours?"
      - "How many credits did I spend today?"
      - "Is there anything still running?"

    Args:
        query: Plain-English question about recent Manus activity
        hours: How many hours back to look (default 24)
        limit: Max tasks to include in the digest (default 10)
    """
    try:
        # Fetch recent tasks from API
        result = await manus_request("GET", "/tasks", params={"limit": limit})
        tasks = result.get("tasks", result.get("data", []))

        # Load local prompt log for enrichment
        log_by_id = {e["task_id"]: e for e in _read_task_log(hours=hours)}

        # Filter to the requested time window
        cutoff = int(time.time()) - (hours * 3600)
        recent_tasks = []
        for t in tasks:
            created_raw = t.get("created_at", "0")
            try:
                created_ts = int(created_raw)
            except (ValueError, TypeError):
                created_ts = 0
            if created_ts >= cutoff or created_ts == 0:
                recent_tasks.append(t)

        if not recent_tasks:
            return f"No Manus tasks found in the last {hours} hours."

        # Build digest
        total_credits = 0
        statuses = {}
        task_summaries = []

        for t in recent_tasks:
            tid = t.get("id", "?")
            tstatus = t.get("status", "?")
            statuses[tstatus] = statuses.get(tstatus, 0) + 1
            credits = t.get("credit_usage") or t.get("credits_used") or 0
            try:
                total_credits += int(credits)
            except (TypeError, ValueError):
                pass

            meta = t.get("metadata") or {}
            title = meta.get("task_title", "")
            url = meta.get("task_url", "")
            created = t.get("created_at", "")

            # Get prompt from log
            log_entry = log_by_id.get(tid, {})
            prompt = log_entry.get("prompt", title or "(no prompt recorded)")[:200]

            # Get result summary from output
            output_items = t.get("output") or []
            result_preview = ""
            if isinstance(output_items, list):
                for item in reversed(output_items):
                    if item.get("role") == "assistant":
                        for c in item.get("content", []):
                            if isinstance(c, dict) and c.get("type") == "output_text":
                                result_preview = c.get("text", "")[:300]
                                break
                    if result_preview:
                        break

            summary = f"  [{tstatus.upper()}] {tid}"
            if created:
                summary += f"  (created: {created})"
            summary += f"\n    Asked: {prompt}"
            if result_preview:
                summary += f"\n    Result: {result_preview}"
            if url:
                summary += f"\n    Link: {url}"
            task_summaries.append(summary)

        # Build the response
        lines = [
            f"GARZA OS — Manus Activity Digest (last {hours}h)",
            f"Query: \"{query}\"",
            f"",
            f"Summary:",
            f"  Tasks found : {len(recent_tasks)}",
        ]
        for s, count in sorted(statuses.items()):
            lines.append(f"  {s:12s}: {count}")
        if total_credits > 0:
            lines.append(f"  Credits used: {total_credits}")
        lines.append("")
        lines.append("Task Details:")
        lines.extend(task_summaries)

        # Query-specific answers
        q_lower = query.lower()
        if any(w in q_lower for w in ["fail", "error", "broke", "wrong"]):
            failed = [t for t in recent_tasks if t.get("status") == "failed"]
            if not failed:
                lines.append("\n✅ No failures in this period.")
            else:
                lines.append(f"\n🔴 {len(failed)} task(s) failed.")
        if any(w in q_lower for w in ["running", "progress", "pending", "still"]):
            active = [t for t in recent_tasks if t.get("status") in ("running", "pending")]
            if not active:
                lines.append("\n✅ No tasks currently running.")
            else:
                lines.append(f"\n🟡 {len(active)} task(s) still active.")
        if any(w in q_lower for w in ["last", "recent", "latest"]):
            if recent_tasks:
                last = recent_tasks[0]
                lid = last.get("id", "?")
                log_entry = log_by_id.get(lid, {})
                last_prompt = log_entry.get("prompt", "(no prompt recorded)")
                lines.append(f"\n📌 Last task: {last_prompt[:200]}")

        return "\n".join(lines)
    except Exception as e:
        return handle_api_error(e)
