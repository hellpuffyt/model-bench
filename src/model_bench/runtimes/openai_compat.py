"""An OpenAI-compatible runtime: llama.cpp, vLLM, LM Studio, TGI, and so on.

These servers all speak ``POST /v1/chat/completions`` with
``stream: true`` and reply in server-sent events — ``data: {json}``
lines terminated by ``data: [DONE]``. That is enough to measure time to
first token and wall time, which is why one adapter covers so many
engines.

Token counts are the awkward part. The spec makes the trailing ``usage``
object optional, and several implementations omit it even when asked. So
this runtime requests it (``stream_options.include_usage``) and uses it
when it arrives, but when it does not, the harness falls back to
counting chunks and the report says ``tokens: chunks`` instead of
``tokens: reported``. Memory is not exposed by the API at all, so
:meth:`memory` stays ``None`` and the report prints ``n/a``: guessing
would be worse than admitting it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any

from ..errors import ProtocolError, RuntimeUnavailable
from ..http import DEFAULT_TIMEOUT, Opener, request_json, stream_lines
from ..models import Chunk, PromptSpec
from ..runtime import Runtime

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DATA_PREFIX = "data:"
DONE_SENTINEL = "[DONE]"


class OpenAICompatRuntime(Runtime):
    """Streams completions from any OpenAI-compatible chat endpoint."""

    name = "openai"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        opener: Opener | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._opener = opener

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def _kwargs(self) -> dict[str, Any]:
        extra: dict[str, Any] = {"timeout": self.timeout, "headers": self._headers()}
        if self._opener is not None:
            extra["opener"] = self._opener
        return extra

    def probe(self) -> None:
        url = f"{self.base_url}/models"
        payload = request_json(self.name, url, **self._kwargs())
        if not isinstance(payload, Mapping) or "data" not in payload:
            raise RuntimeUnavailable(
                self.name, url, "response did not look like an OpenAI-compatible server"
            )

    def stream(self, model: str, prompt: PromptSpec) -> Iterator[Chunk]:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt.text}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0,
            "max_tokens": prompt.max_tokens,
        }
        for line in stream_lines(
            self.name, self.endpoint, payload=body, **self._kwargs()
        ):
            chunk = _parse_event(line, self.endpoint)
            if chunk is not None:
                yield chunk


def _parse_event(line: str, endpoint: str) -> Chunk | None:
    """One SSE line into a chunk, or ``None`` if it carries no data.

    Comment lines (``: keep-alive``) and the ``event:`` / ``id:`` fields
    are ignored rather than rejected: they are legal SSE and several
    servers emit them, and a benchmark that crashed on a heartbeat would
    be useless against exactly the long generations it exists to measure.
    """
    if line.startswith(":"):
        return None
    if not line.startswith(DATA_PREFIX):
        return None
    data = line[len(DATA_PREFIX) :].strip()
    if data == DONE_SENTINEL:
        return Chunk(text="", done=True)
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as error:
        raise ProtocolError("openai", f"not JSON: {data[:80]!r} ({error})") from error
    if not isinstance(payload, Mapping):
        raise ProtocolError("openai", f"expected a JSON object, got {data[:80]!r}")
    if "error" in payload:
        raise RuntimeUnavailable("openai", endpoint, _error_text(payload["error"]))
    text, finished = _delta(payload)
    output_tokens, prompt_tokens = _usage(payload)
    return Chunk(
        text=text,
        done=finished or output_tokens is not None,
        output_tokens=output_tokens,
        prompt_tokens=prompt_tokens,
    )


def _error_text(error: Any) -> str:
    if isinstance(error, Mapping):
        return str(error.get("message", error))
    return str(error)


def _delta(payload: Mapping[str, Any]) -> tuple[str, bool]:
    """The text and finish flag of the first choice, forgivingly."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", False
    first = choices[0]
    if not isinstance(first, Mapping):
        return "", False
    finished = first.get("finish_reason") is not None
    delta = first.get("delta")
    if isinstance(delta, Mapping):
        content = delta.get("content")
        if isinstance(content, str):
            return content, finished
    return "", finished


def _usage(payload: Mapping[str, Any]) -> tuple[int | None, int | None]:
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return None, None
    return _as_int(usage.get("completion_tokens")), _as_int(usage.get("prompt_tokens"))


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
