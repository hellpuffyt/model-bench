"""Recordings: a benchmark run written down so it can be replayed.

A recording is the timestamp of every streamed chunk of every
repetition, plus what the runtime reported about tokens and memory. It
exists for three reasons.

**Reviewability.** A pull request that claims a speed-up can attach the
recording it measured, and anyone can replay it and get the identical
table rather than taking the table on trust.

**Testability.** The statistics, the aggregation and every report format
can be exercised end to end against a recording, with no model, no
server, and no sleeping in a test suite.

**Honesty.** Replay is a distinct mode with its own subcommand, and a
report built from one carries ``replayed_from``, which every format
prints. A recording is never quietly substituted for a live run.

Times are stored as offsets in seconds from the start of that
repetition's request, not as absolute timestamps: offsets are what the
metrics are computed from, and they stay meaningful when a file is moved
between machines.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import RecordingError
from .models import MemoryReading

SCHEMA = "model-bench/recording/1"


@dataclass(frozen=True)
class RecordedChunk:
    """One streamed chunk: when it arrived, and what it carried."""

    at_s: float
    text: str


@dataclass(frozen=True)
class RecordedStream:
    """One repetition of one prompt, start to finish."""

    prompt: str
    repetition: int
    chunks: tuple[RecordedChunk, ...]
    end_s: float
    reported_output_tokens: int | None = None
    reported_prompt_tokens: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "repetition": self.repetition,
            "end_s": self.end_s,
            "reported_output_tokens": self.reported_output_tokens,
            "reported_prompt_tokens": self.reported_prompt_tokens,
            "chunks": [
                {"at_s": chunk.at_s, "text": chunk.text} for chunk in self.chunks
            ],
        }


@dataclass(frozen=True)
class Recording:
    """Everything needed to replay a run exactly."""

    runtime: str
    model: str
    endpoint: str
    streams: tuple[RecordedStream, ...]
    memory: MemoryReading | None = None
    warmup: int = 0

    @property
    def prompts(self) -> tuple[str, ...]:
        """Prompt names in the order they were first recorded."""
        seen: list[str] = []
        for stream in self.streams:
            if stream.prompt not in seen:
                seen.append(stream.prompt)
        return tuple(seen)

    def streams_for(self, prompt: str) -> tuple[RecordedStream, ...]:
        return tuple(s for s in self.streams if s.prompt == prompt)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "runtime": self.runtime,
            "model": self.model,
            "endpoint": self.endpoint,
            "warmup": self.warmup,
            "memory": None if self.memory is None else self.memory.as_dict(),
            "streams": [stream.as_dict() for stream in self.streams],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=False) + "\n"

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")


def _require_mapping(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecordingError(f"{what} must be a JSON object, got {type(value).__name__}")
    return value


def _require_number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordingError(f"{what} must be a number, got {value!r}")
    return float(value)


def _optional_int(value: Any, what: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecordingError(f"{what} must be an integer or null, got {value!r}")
    return value


def _parse_memory(value: Any) -> MemoryReading | None:
    if value is None:
        return None
    payload = _require_mapping(value, "memory")
    total = _optional_int(payload.get("total_bytes"), "memory.total_bytes")
    if total is None:
        raise RecordingError("memory.total_bytes is required when memory is present")
    return MemoryReading(
        total_bytes=total,
        vram_bytes=_optional_int(payload.get("vram_bytes"), "memory.vram_bytes"),
        source=str(payload.get("source", "unknown")),
    )


def _parse_stream(value: Any, index: int) -> RecordedStream:
    payload = _require_mapping(value, f"streams[{index}]")
    for key in ("prompt", "chunks", "end_s"):
        if key not in payload:
            raise RecordingError(f"streams[{index}] is missing {key!r}")
    raw_chunks = payload["chunks"]
    if not isinstance(raw_chunks, Sequence) or isinstance(raw_chunks, (str, bytes)):
        raise RecordingError(f"streams[{index}].chunks must be a list")
    chunks: list[RecordedChunk] = []
    previous = 0.0
    for position, raw in enumerate(raw_chunks):
        chunk = _require_mapping(raw, f"streams[{index}].chunks[{position}]")
        at_s = _require_number(
            chunk.get("at_s"), f"streams[{index}].chunks[{position}].at_s"
        )
        if at_s < previous:
            raise RecordingError(
                f"streams[{index}].chunks[{position}].at_s ({at_s}) goes backwards; "
                "chunk times must not decrease"
            )
        previous = at_s
        chunks.append(RecordedChunk(at_s=at_s, text=str(chunk.get("text", ""))))
    end_s = _require_number(payload["end_s"], f"streams[{index}].end_s")
    if end_s < previous:
        raise RecordingError(
            f"streams[{index}].end_s ({end_s}) is before its last chunk ({previous})"
        )
    return RecordedStream(
        prompt=str(payload["prompt"]),
        repetition=_optional_int(payload.get("repetition"), "repetition") or 0,
        chunks=tuple(chunks),
        end_s=end_s,
        reported_output_tokens=_optional_int(
            payload.get("reported_output_tokens"), "reported_output_tokens"
        ),
        reported_prompt_tokens=_optional_int(
            payload.get("reported_prompt_tokens"), "reported_prompt_tokens"
        ),
    )


def loads(text: str) -> Recording:
    """Parse a recording, rejecting anything it cannot replay faithfully."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise RecordingError(f"not valid JSON: {error}") from error
    document = _require_mapping(payload, "recording")
    schema = document.get("schema")
    if schema != SCHEMA:
        raise RecordingError(f"unknown recording schema {schema!r}; expected {SCHEMA!r}")
    raw_streams = document.get("streams")
    if not isinstance(raw_streams, Sequence) or isinstance(raw_streams, (str, bytes)):
        raise RecordingError("recording.streams must be a list")
    if not raw_streams:
        raise RecordingError("recording contains no streams")
    streams = tuple(
        _parse_stream(raw, index) for index, raw in enumerate(raw_streams)
    )
    return Recording(
        runtime=str(document.get("runtime", "unknown")),
        model=str(document.get("model", "unknown")),
        endpoint=str(document.get("endpoint", "unknown")),
        streams=streams,
        memory=_parse_memory(document.get("memory")),
        warmup=_optional_int(document.get("warmup"), "warmup") or 0,
    )


def load(path: str | Path) -> Recording:
    """Read and parse a recording file."""
    location = Path(path)
    try:
        text = location.read_text(encoding="utf-8")
    except OSError as error:
        raise RecordingError(f"cannot read recording {location}: {error}") from error
    try:
        return loads(text)
    except RecordingError as error:
        raise RecordingError(f"{location}: {error}") from error
