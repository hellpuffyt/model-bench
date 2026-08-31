"""Shared fakes.

These stand in for the two things a benchmark cannot have in CI: a model
server, and the passage of time. Both are replaced at the interface the
production code already has — :class:`Runtime` and the HTTP opener — so
the tests exercise the real harness, the real statistics and the real
CLI rather than a parallel implementation of them.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from model_bench.models import Chunk, MemoryReading, PromptSpec, Sample, TokenSource
from model_bench.runtime import Clock, Runtime
from model_bench.runtimes import VirtualClock

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
LLAMA_RECORDING = FIXTURES / "ollama-llama32-3b.json"

#: One scripted generation: (chunk offset seconds, text) pairs.
Script = Sequence[tuple[float, str]]


class ScriptedRuntime(Runtime):
    """A runtime that plays fixed scripts against a clock it controls.

    ``scripts`` are consumed in order, one per call to :meth:`stream`,
    and the last one repeats forever so a test does not have to spell out
    warm-up runs it does not care about.
    """

    name = "scripted"

    def __init__(
        self,
        scripts: Sequence[Script],
        *,
        end_gap: float = 0.01,
        reported_tokens: int | None = None,
        prompt_tokens: int | None = None,
        memory: MemoryReading | None = None,
        probe_error: Exception | None = None,
        memory_error: Exception | None = None,
        endpoint: str = "http://scripted.invalid/generate",
    ) -> None:
        self.scripts = list(scripts)
        self.end_gap = end_gap
        self.reported_tokens = reported_tokens
        self.prompt_tokens = prompt_tokens
        self._memory = memory
        self.probe_error = probe_error
        self.memory_error = memory_error
        self._endpoint = endpoint
        self._clock = VirtualClock()
        self.calls: list[tuple[str, str]] = []

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def clock(self) -> Clock:
        return self._clock

    def probe(self) -> None:
        if self.probe_error is not None:
            raise self.probe_error

    def stream(self, model: str, prompt: PromptSpec) -> Iterator[Chunk]:
        self.calls.append((model, prompt.name))
        script = self.scripts[min(len(self.calls) - 1, len(self.scripts) - 1)]
        start = self._clock.now
        for offset, text in script:
            self._clock.advance_to(start + offset)
            yield Chunk(text=text)
        last = script[-1][0] if script else 0.0
        self._clock.advance_to(start + last + self.end_gap)
        yield Chunk(
            text="",
            done=True,
            output_tokens=self.reported_tokens,
            prompt_tokens=self.prompt_tokens,
        )

    def memory(self, model: str) -> MemoryReading | None:
        if self.memory_error is not None:
            raise self.memory_error
        return self._memory


class FakeResponse:
    """Enough of an HTTP response for :mod:`model_bench.http`."""

    def __init__(self, body: str) -> None:
        self._buffer = io.BytesIO(body.encode("utf-8"))
        self.closed = False

    def read(self) -> bytes:
        return self._buffer.read()

    def __iter__(self) -> Iterator[bytes]:
        return iter(self._buffer)

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.closed = True


class FakeOpener:
    """An opener that answers from a script, recording what it was asked."""

    def __init__(self, *bodies: str, error: Exception | None = None) -> None:
        self.bodies = list(bodies)
        self.error = error
        self.requests: list[Any] = []
        self.responses: list[FakeResponse] = []

    def __call__(self, request: Any, timeout: float) -> FakeResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        body = self.bodies.pop(0) if len(self.bodies) > 1 else self.bodies[0]
        response = FakeResponse(body)
        self.responses.append(response)
        return response

    @property
    def payload(self) -> dict[str, Any]:
        """The JSON body of the most recent request."""
        data = self.requests[-1].data
        assert data is not None
        parsed: dict[str, Any] = json.loads(data.decode("utf-8"))
        return parsed


def make_sample(
    prompt: str = "p",
    repetition: int = 0,
    ttft_s: float = 0.1,
    total_s: float = 1.1,
    output_tokens: int = 100,
    token_source: TokenSource = TokenSource.REPORTED,
    output_chars: int = 400,
    prompt_tokens: int | None = 12,
) -> Sample:
    """A valid sample, with only the interesting field overridden."""
    return Sample(
        prompt=prompt,
        repetition=repetition,
        ttft_s=ttft_s,
        total_s=total_s,
        output_tokens=output_tokens,
        token_source=token_source,
        output_chars=output_chars,
        prompt_tokens=prompt_tokens,
    )


def ndjson(*objects: dict[str, Any]) -> str:
    return "".join(json.dumps(obj) + "\n" for obj in objects)


def sse(*chunks: str) -> str:
    return "".join(f"data: {chunk}\n\n" for chunk in chunks)


@pytest.fixture
def recording_path() -> Path:
    """The committed recording of a real llama3.2:3b run."""
    return LLAMA_RECORDING


@pytest.fixture
def three_prompts() -> tuple[PromptSpec, ...]:
    return (
        PromptSpec("a", "first", 8),
        PromptSpec("b", "second", 8),
        PromptSpec("c", "third", 8),
    )
