"""Runtime backends, and the name-to-backend lookup the CLI uses."""

from __future__ import annotations

from .ollama import DEFAULT_HOST as OLLAMA_HOST
from .ollama import OllamaRuntime
from .openai_compat import DEFAULT_BASE_URL as OPENAI_BASE_URL
from .openai_compat import OpenAICompatRuntime
from .replay import ReplayRuntime, VirtualClock

#: Runtimes that can be selected with ``--runtime``. Replay is absent on
#: purpose: it is reached through the ``replay`` subcommand, so a
#: recording can never be mistaken for a live measurement.
LIVE_RUNTIMES = ("ollama", "openai")

DEFAULT_ENDPOINTS = {"ollama": OLLAMA_HOST, "openai": OPENAI_BASE_URL}

__all__ = [
    "DEFAULT_ENDPOINTS",
    "LIVE_RUNTIMES",
    "OLLAMA_HOST",
    "OPENAI_BASE_URL",
    "OllamaRuntime",
    "OpenAICompatRuntime",
    "ReplayRuntime",
    "VirtualClock",
]
