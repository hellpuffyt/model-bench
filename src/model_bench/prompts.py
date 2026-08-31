"""The fixed prompt suite.

A benchmark is only a comparison if the work is identical, so the
prompts are committed constants rather than something generated per run.
Three sizes, because the shape of the answer changes with each:

``tiny``
    A handful of output tokens. Dominated by time to first token, which
    is what makes an assistant feel responsive.

``short``
    A paragraph. The everyday case, long enough for decode throughput to
    mean something.

``long``
    A few hundred tokens. Where a runtime's KV-cache behaviour and any
    per-token overhead start to show.

Each prompt caps its own output (``max_tokens``). Without a cap, a
chattier model would be "slower" only because it was asked, in effect, a
larger question — and the whole comparison would be about verbosity
rather than speed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from .errors import UsageError
from .models import PromptSpec

BUILTIN_PROMPTS: tuple[PromptSpec, ...] = (
    PromptSpec(
        name="tiny",
        text="Reply with exactly the word: pong",
        max_tokens=16,
    ),
    PromptSpec(
        name="short",
        text=(
            "In one paragraph, explain what a B-tree index is and why a "
            "database uses one instead of scanning every row."
        ),
        max_tokens=128,
    ),
    PromptSpec(
        name="long",
        text=(
            "Write a thorough technical explanation of how HTTP/2 multiplexing "
            "works. Cover streams, frames, flow control, header compression "
            "with HPACK, and how head-of-line blocking at the TCP layer still "
            "affects it. Aim for several hundred words."
        ),
        max_tokens=512,
    ),
)

SETS: dict[str, tuple[str, ...]] = {
    "all": ("tiny", "short", "long"),
    "quick": ("tiny", "short"),
    "tiny": ("tiny",),
    "short": ("short",),
    "long": ("long",),
}


def builtin(name: str) -> PromptSpec:
    """One built-in prompt by name."""
    for prompt in BUILTIN_PROMPTS:
        if prompt.name == name:
            return prompt
    known = ", ".join(prompt.name for prompt in BUILTIN_PROMPTS)
    raise UsageError(f"no built-in prompt {name!r}; there is: {known}")


def prompt_set(name: str) -> tuple[PromptSpec, ...]:
    """A named group of built-in prompts."""
    try:
        names = SETS[name]
    except KeyError:
        known = ", ".join(sorted(SETS))
        raise UsageError(f"no prompt set {name!r}; there is: {known}") from None
    return tuple(builtin(entry) for entry in names)


def _spec_from(entry: object, where: str) -> PromptSpec:
    if not isinstance(entry, Mapping):
        raise UsageError(f"{where}: each prompt must be a JSON object")
    missing = [key for key in ("name", "text") if key not in entry]
    if missing:
        raise UsageError(f"{where}: prompt is missing {', '.join(missing)}")
    raw_max = entry.get("max_tokens", 128)
    if isinstance(raw_max, bool) or not isinstance(raw_max, int):
        raise UsageError(f"{where}: max_tokens must be an integer, got {raw_max!r}")
    return PromptSpec(
        name=str(entry["name"]), text=str(entry["text"]), max_tokens=raw_max
    )


def load_prompts(path: str | Path) -> tuple[PromptSpec, ...]:
    """Read a custom prompt file: a JSON list of ``{name, text, max_tokens}``.

    Duplicate names are rejected. Prompt names key every table row and
    every recorded stream, so two prompts sharing one would silently
    merge two different measurements into one column.
    """
    location = Path(path)
    try:
        text = location.read_text(encoding="utf-8")
    except OSError as error:
        raise UsageError(f"cannot read prompt file {location}: {error}") from error
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise UsageError(f"{location} is not valid JSON: {error}") from error
    if isinstance(payload, Mapping):
        payload = payload.get("prompts")
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise UsageError(
            f"{location} must contain a list of prompts, or an object with a "
            "'prompts' list"
        )
    if not payload:
        raise UsageError(f"{location} contains no prompts")
    specs = [_spec_from(entry, str(location)) for entry in payload]
    return tuple(deduplicated(specs, str(location)))


def deduplicated(specs: Iterable[PromptSpec], where: str) -> list[PromptSpec]:
    """The prompts, verifying that no name appears twice."""
    seen: set[str] = set()
    result: list[PromptSpec] = []
    for spec in specs:
        if spec.name in seen:
            raise UsageError(f"{where}: duplicate prompt name {spec.name!r}")
        seen.add(spec.name)
        result.append(spec)
    return result
