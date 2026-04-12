# Hybrid MCP Server — OpenManus + Manus API

GARZA OS integration that combines fast local agents with Manus AI's autonomous cloud execution.

## Architecture

```
Claude / Devin (GARZA OS)
    │
    └── Hybrid MCP Server (Railway)  — 25 tools
            │
            ├── Sync Layer (3 tools) — OpenManus local agents
            │     bash, editor, terminate
            │     ↳ Direct execution, results inline
            │
            ├── Manus API Layer (9 tools) — Async cloud execution
            │     manus_create_task    → fire-and-forget
            │     manus_get_task       → poll for results
            │     manus_list_tasks     → browsing history
            │     manus_upload_file    → attach context
            │     manus_list_files     → file management
            │     manus_create_webhook → push notifications via n8n
            │     manus_get_steps      → step-by-step task trace
            │     manus_get_parent     → resolve subtask lineage
            │     manus_watch_task     → long-poll until completion
            │
            ├── Observability Layer (4 tools)
            │     manus_cost_summary   → daily/weekly spend breakdown
            │     manus_diagnose_task  → LLM root-cause analysis
            │     manus_resume_task    → retry failed tasks
            │     manus_triage_task    → priority + action recommendation
            │
            ├── Fleet Management (5 tools)
            │     agent_registry_list   → list registered agents
            │     agent_registry_update → update agent metadata
            │     garza_fleet_status    → fleet-wide health overview
            │     manus_kill_task       → cancel a running task
            │     manus_kill_zombies    → kill all stale tasks
            │
            ├── Intelligence Layer (3 tools)
            │     garza_status         → NL task digest (LLM summarization)
            │     garza_flow_status    → cross-agent workflow view
            │     garza_daily_brief    → morning briefing for Jaden
            │
            └── Memory Layer (1 tool)
                  garza_recall         → search long-term memory (Fabric AI)
```

## When to Use Each Layer

| Use case | Tool |
|----------|------|
| Run shell commands | `bash` |
| Read / write files | `editor` |
| Signal task done | `terminate` |
| Deep research, multi-step web tasks | `manus_create_task` (agent, quality) |
| Quick research or data lookup | `manus_create_task` (agent, speed) |
| Check if a Manus task finished | `manus_get_task` |
| List past Manus tasks | `manus_list_tasks` |
| Give Manus a document to work with | `manus_upload_file` → `manus_create_task` |
| Auto-notify n8n on completion | `manus_create_webhook` |
| Watch a task until it finishes | `manus_watch_task` |
| See what Manus did step-by-step | `manus_get_steps` |
| Find parent of a subtask | `manus_get_parent` |
| Check weekly spend | `manus_cost_summary` |
| Debug why a task failed | `manus_diagnose_task` |
| Retry a failed task | `manus_resume_task` |
| Get priority recommendation | `manus_triage_task` |
| Cancel a running task | `manus_kill_task` |
| Kill all zombie tasks | `manus_kill_zombies` |
| Ask "What did Manus do today?" | `garza_status` |
| View cross-agent workflows | `garza_flow_status` |
| Morning briefing | `garza_daily_brief` |
| List all registered agents | `agent_registry_list` |
| Update agent metadata | `agent_registry_update` |
| Fleet-wide health check | `garza_fleet_status` |
| Search past session memory | `garza_recall` |

## Railway Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_MODEL` | ✅ | e.g. `gpt-4o` or `claude-sonnet-4-5` |
| `LLM_BASE_URL` | ✅ | e.g. `https://api.openai.com/v1` |
| `LLM_API_KEY` | ✅ | Your LLM provider API key |
| `MANUS_API_KEY` | ✅ | From manus.im → Settings → API |
| `GEMINI_API_KEY` | Optional | Google Gemini key for garza_status LLM summarization |
| `FABRIC_SO_MCP_URL` | Optional | Fabric AI MCP URL for memory layer (garza_recall) |
| `FABRIC_SO_API_KEY` | Optional | Fabric AI API key |
| `MCP_SERVER_AUTH_TOKEN` | Optional | Bearer token for auth enforcement (Fix 5) |
| `MANUS_GMAIL_CONNECTOR_ID` | Optional | Manus connector ID for Gmail access |
| `MANUS_NOTION_CONNECTOR_ID` | Optional | Manus connector ID for Notion access |
| `MANUS_GCAL_CONNECTOR_ID` | Optional | Manus connector ID for Google Calendar access |
| `GARZA_RUNAWAY_MINUTES` | Optional | Threshold (min) before flagging runaway tasks (default 30) |
| `GARZA_RUNAWAY_CREDITS` | Optional | Credit threshold for runaway warnings (default 10) |

## Deployment

### 1. Fork & connect to Railway

Fork `FoundationAgents/OpenManus` (already done if you're reading this).
In Railway → New Project → Deploy from GitHub → select `itsablabla/openmanus-garza`.

### 2. Set environment variables

Add all required variables from the table above in Railway → Variables.

### 3. Deploy

Railway auto-deploys on push. `entrypoint.sh` runs at startup, writes `config/config.toml`
from env vars, then starts the MCP server.

### 4. Verify startup logs

```
[entrypoint] Writing config/config.toml from environment variables...
[entrypoint] config.toml written. Starting MCP server (Streamable HTTP, 0.0.0.0:8080)...
[run_mcp_server] Starting Streamable HTTP on 0.0.0.0:8080
Using Streamable HTTP transport (POST /mcp)
```

## Connect to Claude / MCP Clients

### Streamable HTTP (recommended)

Use any MCP client that supports the Streamable HTTP transport:

```
URL: https://openmanus-mcp-production.up.railway.app/mcp
Transport: Streamable HTTP
Headers:
  Accept: application/json, text/event-stream
  Authorization: Bearer <MCP_SERVER_AUTH_TOKEN>  (if auth enabled)
```

### Claude Desktop (claude_desktop_config.json)

```json
{
  "mcpServers": {
    "openmanus-hybrid": {
      "url": "https://openmanus-mcp-production.up.railway.app/mcp"
    }
  }
}
```

## Diagnostic Test

```bash
MANUS_API_KEY=your_key python tests/test_manus_api_tools.py
```

Runs create → poll → list → webhook test directly against Manus API.

## Cost Reference

| Profile | Credits | Best for |
|---------|---------|----------|
| `speed` | ~50-100 credits | Quick research, summaries |
| `quality` | ~150-300 credits | Deep research, complex multi-step tasks |

Credits are consumed per task. See manus.im/pricing for current rates.

## File Map

```
run_mcp_server.py              ← server entry point (Streamable HTTP + auth middleware)
entrypoint.sh                  ← writes config.toml at container startup
railway.json                   ← Railway deployment config
Dockerfile                     ← CMD fixed to ./entrypoint.sh
app/
  config.py                    ← env var substitution + debug logging
  mcp/
    server.py                  ← hybrid server (25 tools across 6 layers)
    manus_client.py            ← Manus API client
    manus_models.py            ← Pydantic input models
tests/
  test_manus_api_tools.py      ← diagnostic script
```
