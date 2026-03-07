"""
Daytona SDK stub for environments where the real Daytona SDK is not available.

This stub satisfies all import statements in app/daytona/ so that agents
(Manus, DataAnalysis, SWEAgent, BrowserAgent) can be imported and registered
as MCP tools. Actual Daytona sandbox operations will raise NotImplementedError
at runtime if called without a real Daytona API key and SDK.
"""
from enum import Enum
from typing import Any, Optional


class SandboxState(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    ARCHIVED = "archived"
    UNKNOWN = "unknown"


class _StubBase:
    """Base for all stub classes — raises NotImplementedError on any method call."""

    def __getattr__(self, name: str):
        def _not_implemented(*args, **kwargs):
            raise NotImplementedError(
                f"Daytona SDK is not installed. "
                f"Install the real 'daytona' package or provide a DAYTONA_API_KEY "
                f"to use sandbox features. (Called: {self.__class__.__name__}.{name})"
            )
        return _not_implemented


class DaytonaConfig(_StubBase):
    def __init__(self, api_key: str = "", server_url: str = "", target: str = ""):
        self.api_key = api_key
        self.server_url = server_url
        self.target = target


class Daytona(_StubBase):
    def __init__(self, config: Optional[DaytonaConfig] = None):
        self.config = config


class Sandbox(_StubBase):
    pass


class Resources(_StubBase):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class CreateSandboxFromImageParams(_StubBase):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class SessionExecuteRequest(_StubBase):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


__all__ = [
    "Daytona",
    "DaytonaConfig",
    "Sandbox",
    "SandboxState",
    "Resources",
    "CreateSandboxFromImageParams",
    "SessionExecuteRequest",
]
