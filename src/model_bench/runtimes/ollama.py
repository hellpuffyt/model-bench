"""The Ollama runtime, spoken over its native NDJSON streaming API.

Ollama is the one backend here that can answer all three questions
honestly. ``/api/generate`` streams one JSON object per token-ish chunk
and ends with an object carrying ``eval_count``, so token counts are
*reported* rather than guessed; and ``/api/ps`` lists the models
currently resident with their size and VRAM share, which is the only
memory figure an HTTP client can obtain that means anything.

Generation is pinned to ``temperature: 0`` with a fixed seed. A
benchmark wants the same work done every repetition, and sampling
temperature changes how many tokens come out, which would move the very
denominator being measured.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any

from ..errors import ProtocolError, RuntimeUnavailable
from ..http import DEFAULT_TIMEOUT, Opener, request_json, stream_lines
from ..models import Chunk, MemoryReading, PromptSpec
from ..runtime import Runtime

DEFAULT_HOST = "http://127.0.0.1:11434"


class OllamaRuntime(Runtime):
    """Talks to an Ollama server over ``/api/generate`` and ``/api/ps``."""

    name = "ollama"

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        seed: int = 0,
        opener: Opener | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.seed = seed
        self._opener = opener

    @property
    def endpoint(self) -> str:
        return f"{self.host}/api/generate"

    def _kwargs(self) -> dict[str, Any]:
        extra: dict[str, Any] = {"timeout": self.timeout}
        if self._opener is not None:
            extra["opener"] = self._opener
        return extra

    def probe(self) -> None:
        """Ask for the model list; anything but a mapping is a failure."""
        url = f"{self.host}/api/tags"
        payload = request_json(self.name, url, **self._kwargs())
        if not isinstance(payload, Mapping) or "models" not in payload:
            raise RuntimeUnavailable(
                self.name, url, "response did not look like an Ollama server"
            )

    def available_models(self) -> list[str]:
        """Model names the server has pulled, for a friendly error."""
        payload = request_json(self.name, f"{self.host}/api/tags", **self._kwargs())
        if not isinstance(payload, Mapping):
            return []
        models = payload.get("models")
        if not isinstance(models, list):
            return []
        return [
            str(entry["name"])
            for entry in models
            if isinstance(entry, Mapping) and "name" in entry
        ]

    def stream(self, model: str, prompt: PromptSpec) -> Iterator[Chunk]:
        body = {
            "model": model,
            "prompt": prompt.text,
            "stream": True,
            "options": {
                "temperature": 0,
                "seed": self.seed,
                "num_predict": prompt.max_tokens,
            },
        }
        for line in stream_lines(
            self.name, self.endpoint, payload=body, **self._kwargs()
        ):
            yield _parse_line(line, self.endpoint)

    def memory(self, model: str) -> MemoryReading | None:
        return read_memory(
            self.host, model, timeout=self.timeout, opener=self._opener
        )


def _parse_line(line: str, endpoint: str) -> Chunk:
    """One NDJSON line into a :class:`Chunk`.

    An ``error`` field is Ollama's way of reporting a missing model
    mid-stream; it becomes an exception rather than an empty chunk, so a
    run against a model that is not installed fails loudly instead of
    reporting a very fast zero-token generation.
    """
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as error:
        raise ProtocolError("ollama", f"not JSON: {line[:80]!r} ({error})") from error
    if not isinstance(payload, Mapping):
        raise ProtocolError("ollama", f"expected a JSON object, got {line[:80]!r}")
    if "error" in payload:
        raise RuntimeUnavailable("ollama", endpoint, str(payload["error"]))
    return Chunk(
        text=str(payload.get("response", "")),
        done=bool(payload.get("done", False)),
        output_tokens=_optional_int(payload.get("eval_count")),
        prompt_tokens=_optional_int(payload.get("prompt_eval_count")),
    )


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def read_memory(
    host: str,
    model: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    opener: Opener | None = None,
) -> MemoryReading | None:
    """The resident size of ``model`` from ``/api/ps``, if it is loaded.

    Returns ``None`` when the model is not in the list — after a run it
    normally is, but a server that unloaded it in between should produce
    "n/a" rather than a stale or invented number.
    """
    url = f"{host.rstrip('/')}/api/ps"
    extra: dict[str, Any] = {"timeout": timeout}
    if opener is not None:
        extra["opener"] = opener
    payload = request_json("ollama", url, **extra)
    if not isinstance(payload, Mapping):
        return None
    entries = payload.get("models")
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("name", "")) != model:
            continue
        size = _optional_int(entry.get("size"))
        if size is None:
            return None
        return MemoryReading(
            total_bytes=size,
            vram_bytes=_optional_int(entry.get("size_vram")),
            source=url,
        )
    return None
