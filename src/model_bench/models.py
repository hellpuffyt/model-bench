"""The data the benchmark moves around.

Everything here is a frozen dataclass. A benchmark result is evidence,
and evidence that a later stage can quietly edit is not evidence.

The one idea worth reading before the rest of the package is
:class:`TokenSource`. Tokens per second is a ratio, and its denominator
(elapsed seconds) is easy to measure honestly while its numerator (how
many tokens the model produced) is not: an HTTP client sees chunks of
text, not tokens. So every sample records *where its token count came
from*, the reports print that, and a run whose counts were inferred from
chunk boundaries is never presented as though the server had counted
them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .errors import UsageError
from .stats import Summary, summarize


class TokenSource(str, Enum):
    """Where a sample's output token count came from."""

    REPORTED = "reported"
    """The runtime told us, e.g. Ollama's ``eval_count``. Trustworthy."""

    CHUNKS = "chunks"
    """Counted streamed chunks. An approximation, and labelled as one."""

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Chunk:
    """One streamed piece of a completion.

    ``output_tokens`` is set only on the final chunk, and only by
    runtimes whose protocol actually reports a count.
    """

    text: str
    done: bool = False
    output_tokens: int | None = None
    prompt_tokens: int | None = None


@dataclass(frozen=True)
class MemoryReading:
    """How much memory a runtime says the loaded model occupies.

    This is what the *runtime* reports about the model it has resident,
    not the RSS of the benchmark process, which would be meaningless.
    ``source`` names the endpoint the figure came from so a reader can
    check it. When a runtime cannot report memory the reading is absent
    entirely — the reports print "n/a" rather than a zero.
    """

    total_bytes: int
    source: str
    vram_bytes: int | None = None

    @property
    def resident_bytes(self) -> int:
        """Bytes held outside VRAM, i.e. in ordinary system memory."""
        if self.vram_bytes is None:
            return self.total_bytes
        return max(0, self.total_bytes - self.vram_bytes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_bytes": self.total_bytes,
            "vram_bytes": self.vram_bytes,
            "source": self.source,
        }


@dataclass(frozen=True)
class PromptSpec:
    """A fixed prompt, held constant across runtimes so runs compare."""

    name: str
    text: str
    max_tokens: int = 128

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise UsageError("a prompt needs a non-empty name")
        if not self.text:
            raise UsageError(f"prompt {self.name!r} has no text")
        if self.max_tokens < 1:
            raise UsageError(
                f"prompt {self.name!r} asks for {self.max_tokens} tokens; "
                "it must be at least 1"
            )

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "text": self.text, "max_tokens": self.max_tokens}


@dataclass(frozen=True)
class Sample:
    """One completed generation: the atom every statistic is built from.

    ``ttft_s`` is the wall time from sending the request to the first
    chunk that carried text. ``total_s`` runs to the end of the stream.
    """

    prompt: str
    repetition: int
    ttft_s: float
    total_s: float
    output_tokens: int
    token_source: TokenSource
    output_chars: int
    prompt_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.ttft_s < 0 or self.total_s < 0:
            raise UsageError("a sample cannot have negative elapsed time")
        if self.total_s < self.ttft_s:
            raise UsageError(
                "a sample cannot finish before its first token "
                f"(ttft {self.ttft_s}s > total {self.total_s}s)"
            )
        if self.output_tokens < 0:
            raise UsageError("a sample cannot have a negative token count")

    @property
    def decode_s(self) -> float:
        """Time spent generating after the first token appeared."""
        return self.total_s - self.ttft_s

    @property
    def tokens_per_second(self) -> float | None:
        """Decode throughput, excluding the wait for the first token.

        ``None`` when it cannot be computed honestly: a stream that
        produced one token or none has no decode phase to measure, and
        neither does one that finished within the clock's resolution.
        Dividing anyway would print an infinity or a number derived from
        timer noise.
        """
        if self.output_tokens < 2 or self.decode_s <= 0.0:
            return None
        return (self.output_tokens - 1) / self.decode_s

    @property
    def overall_tokens_per_second(self) -> float | None:
        """Throughput including the first-token wait, as a user feels it."""
        if self.output_tokens < 1 or self.total_s <= 0.0:
            return None
        return self.output_tokens / self.total_s

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "repetition": self.repetition,
            "ttft_s": self.ttft_s,
            "total_s": self.total_s,
            "output_tokens": self.output_tokens,
            "prompt_tokens": self.prompt_tokens,
            "token_source": self.token_source.value,
            "output_chars": self.output_chars,
            "tokens_per_second": self.tokens_per_second,
        }


@dataclass(frozen=True)
class PromptResult:
    """Every repetition of one prompt, plus the statistics over them."""

    prompt: str
    samples: tuple[Sample, ...]

    def __post_init__(self) -> None:
        if not self.samples:
            raise UsageError(f"prompt {self.prompt!r} produced no samples")

    @property
    def ttft(self) -> Summary:
        return summarize([sample.ttft_s for sample in self.samples])

    @property
    def total(self) -> Summary:
        return summarize([sample.total_s for sample in self.samples])

    @property
    def throughput(self) -> Summary | None:
        """Tokens per second across repetitions, or ``None`` if unmeasurable."""
        rates = [
            rate
            for rate in (sample.tokens_per_second for sample in self.samples)
            if rate is not None
        ]
        if not rates:
            return None
        return summarize(rates)

    @property
    def output_tokens(self) -> Summary:
        return summarize([float(sample.output_tokens) for sample in self.samples])

    @property
    def token_source(self) -> TokenSource:
        """``REPORTED`` only if every repetition was reported."""
        if all(s.token_source is TokenSource.REPORTED for s in self.samples):
            return TokenSource.REPORTED
        return TokenSource.CHUNKS

    def as_dict(self) -> dict[str, Any]:
        throughput = self.throughput
        return {
            "prompt": self.prompt,
            "repetitions": len(self.samples),
            "token_source": self.token_source.value,
            "ttft_s": self.ttft.as_dict(),
            "total_s": self.total.as_dict(),
            "tokens_per_second": None if throughput is None else throughput.as_dict(),
            "output_tokens": self.output_tokens.as_dict(),
            "samples": [sample.as_dict() for sample in self.samples],
        }


@dataclass(frozen=True)
class BenchmarkReport:
    """The whole outcome of one ``model-bench`` invocation.

    ``replayed_from`` is not decoration. When a report was produced by
    replaying a recording rather than by talking to a live runtime, every
    output format says so, because the numbers are then a re-run of a
    past measurement and not a measurement of this machine.
    """

    runtime: str
    model: str
    endpoint: str
    results: tuple[PromptResult, ...]
    memory: MemoryReading | None = None
    warmup: int = 0
    replayed_from: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.results:
            raise UsageError("a report needs at least one prompt result")
        seen: set[str] = set()
        for result in self.results:
            if result.prompt in seen:
                raise UsageError(f"duplicate prompt {result.prompt!r} in report")
            seen.add(result.prompt)

    @property
    def is_replay(self) -> bool:
        return self.replayed_from is not None

    @property
    def total_samples(self) -> int:
        return sum(len(result.samples) for result in self.results)

    def result(self, prompt: str) -> PromptResult:
        """The result for one prompt, by name."""
        for candidate in self.results:
            if candidate.prompt == prompt:
                return candidate
        known = ", ".join(candidate.prompt for candidate in self.results)
        raise UsageError(f"no prompt {prompt!r} in this report; it has: {known}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "model-bench/report/1",
            "runtime": self.runtime,
            "model": self.model,
            "endpoint": self.endpoint,
            "warmup": self.warmup,
            "replayed_from": self.replayed_from,
            "memory": None if self.memory is None else self.memory.as_dict(),
            "notes": list(self.notes),
            "prompts": [result.as_dict() for result in self.results],
        }


def build_results(samples: Sequence[Sample]) -> tuple[PromptResult, ...]:
    """Group samples into one :class:`PromptResult` per prompt.

    Prompt order follows first appearance, so a report lists prompts in
    the order they were run rather than alphabetically.
    """
    order: list[str] = []
    grouped: dict[str, list[Sample]] = {}
    for sample in samples:
        if sample.prompt not in grouped:
            order.append(sample.prompt)
            grouped[sample.prompt] = []
        grouped[sample.prompt].append(sample)
    return tuple(PromptResult(name, tuple(grouped[name])) for name in order)
