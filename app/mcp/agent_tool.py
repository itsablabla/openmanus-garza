"""
AgentTool: Wraps any OpenManus agent as a stateless, MCP-compatible tool.

Each call creates a fresh agent instance, runs it to completion, and cleans up.
This ensures thread safety and predictable concurrent operation.

Key design decision: `parameters` is a plain dict field (not a @property) so
that Pydantic's to_param() correctly serializes it instead of returning the
property descriptor object. Using @property causes Pydantic to return the
descriptor itself rather than calling it, resulting in:
  AttributeError: 'property' object has no attribute 'get'
"""
import asyncio
from typing import Any, Optional, Type

from pydantic import Field

from app.logger import logger
from app.tool.base import BaseTool, ToolResult

# The parameters schema for all AgentTool instances — a single string prompt.
_AGENT_TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "description": (
                "The natural language task description for the agent to execute. "
                "Be as specific and detailed as possible."
            ),
        }
    },
    "required": ["prompt"],
}


class AgentTool(BaseTool):
    """
    A generic wrapper that exposes an OpenManus agent as a single-call MCP tool.

    The agent is instantiated fresh on every execute() call, ensuring complete
    isolation between requests. This is the "Agent-as-a-Tool" pattern.
    """

    # Store agent_class as a plain field (not Pydantic-managed) to avoid
    # issues with non-serializable types.
    agent_class: Any = Field(default=None, exclude=True)
    agent_config: dict = Field(default_factory=dict, exclude=True)

    # parameters MUST be a plain dict field (NOT a @property) so that
    # BaseTool.to_param() serializes it correctly via self.parameters.
    # Pydantic returns the property descriptor object instead of calling it,
    # which breaks the MCP server's docstring/signature building.
    parameters: dict = Field(default_factory=lambda: _AGENT_TOOL_PARAMETERS.copy())

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, agent_class: Type, **agent_config):
        agent_name = getattr(agent_class, "name", agent_class.__name__).lower()
        # Replace spaces/dashes for a clean tool name
        tool_name = f"run_{agent_name.replace('-', '_').replace(' ', '_')}"
        agent_description = getattr(
            agent_class, "description", "An OpenManus agent."
        )

        super().__init__(
            name=tool_name,
            description=(
                f"{agent_description}\n\n"
                f"Runs the {agent_class.__name__} agent autonomously to completion. "
                f"Provide a detailed natural language prompt describing the task."
            ),
            parameters=_AGENT_TOOL_PARAMETERS.copy(),
        )
        # Store via object.__setattr__ to bypass pydantic immutability
        object.__setattr__(self, "agent_class", agent_class)
        object.__setattr__(self, "agent_config", agent_config)

    async def execute(self, prompt: str, **kwargs) -> ToolResult:
        """
        Creates a fresh agent instance, runs it with the given prompt,
        and ensures cleanup regardless of success or failure.
        """
        agent_class = object.__getattribute__(self, "agent_class")
        agent_config = object.__getattribute__(self, "agent_config")

        logger.info(
            f"[AgentTool] Spawning {agent_class.__name__} for prompt: {prompt[:80]}..."
        )
        agent = None
        try:
            # Use async factory .create() if available (e.g., Manus), else direct init
            if hasattr(agent_class, "create") and asyncio.iscoroutinefunction(
                agent_class.create
            ):
                agent = await agent_class.create(**agent_config)
            else:
                agent = agent_class(**agent_config)

            result = await agent.run(prompt)
            logger.info(f"[AgentTool] {agent_class.__name__} completed successfully.")
            return ToolResult(output=result)

        except Exception as e:
            logger.error(
                f"[AgentTool] {agent_class.__name__} failed: {e}", exc_info=True
            )
            return ToolResult(output=f"Agent execution failed: {str(e)}", error=str(e))

        finally:
            if agent is not None and hasattr(agent, "cleanup"):
                try:
                    await agent.cleanup()
                except Exception as cleanup_err:
                    logger.warning(
                        f"[AgentTool] Cleanup failed for {agent_class.__name__}: {cleanup_err}"
                    )
