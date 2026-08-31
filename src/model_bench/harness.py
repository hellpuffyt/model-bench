"""The measurement loop.

This module is the only place a duration is ever computed, and it is
deliberately dull: start a clock, ask the runtime to stream, note when
the first chunk with text arrives, note when the stream ends. Everything
interesting is in what it refuses to do.

**It never invents a measurement.** If the runtime cannot be reached, the
:class:`~model_bench.errors.RuntimeUnavailable` from ``probe`` propagates
and no report is produced at all. If a stream ends without ever
producing text, the sample is recorded with zero tokens and a note
saying so, rather than being dropped (which would flatter the average) or
smoothed over.

**It never counts what it did not see.** A token count is only labelled
``reported`` when the runtime's own protocol supplied one. Otherwise the
count is the number of chunks that carried text, labelled ``chunks``, and
the report prints the label next to the number.

**Warm-up runs are thrown away, and their number is recorded.** The first
generation after a model loads is dominated by loading it. Including it
would make every benchmark a measurement of disk speed; hiding that it
happened would make the numbers unreproducible.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .errors import UsageError
from .models import (
    BenchmarkReport,
    MemoryReading,
    PromptSpec,
    Sample,
    TokenSource,
    build_results,
)
from .prompts import deduplicated
from .recording import RecordedChunk, RecordedStream, Recording
from .runtime import Runtime

Observer = Callable[[str], None]


@dataclass(frozen=True)
class _Timed:
    """The raw timings of one generation, before they become a sample."""

    ttft_s: float
    total_s: float
    output_tokens: int
    token_source: TokenSource
    output_chars: int
    prompt_tokens: int | None
    chunks: tuple[RecordedChunk, ...]


@dataclass(frozen=True)
class RunOutcome:
    """What a run produces: the report, and the recording behind it."""

    report: BenchmarkReport
    recording: Recording


def _generate(runtime: Runtime, model: str, prompt: PromptSpec) -> _Timed:
    """Stream one completion, timing it."""
    clock = runtime.clock
    start = clock()
    first_text_at: float | None = None
    chunks: list[RecordedChunk] = []
    reported_output: int | None = None
    reported_prompt: int | None = None
    text_chunks = 0
    characters = 0

    for chunk in runtime.stream(model, prompt):
        moment = clock()
        if chunk.output_tokens is not None:
            reported_output = chunk.output_tokens
        if chunk.prompt_tokens is not None:
            reported_prompt = chunk.prompt_tokens
        if chunk.text:
            if first_text_at is None:
                first_text_at = moment
            text_chunks += 1
            characters += len(chunk.text)
            chunks.append(RecordedChunk(at_s=moment - start, text=chunk.text))

    total_s = clock() - start
    ttft_s = total_s if first_text_at is None else first_text_at - start
    if reported_output is None:
        tokens, source = text_chunks, TokenSource.CHUNKS
    else:
        tokens, source = reported_output, TokenSource.REPORTED
    return _Timed(
        ttft_s=ttft_s,
        total_s=total_s,
        output_tokens=tokens,
        token_source=source,
        output_chars=characters,
        prompt_tokens=reported_prompt,
        chunks=tuple(chunks),
    )


def run_benchmark(
    runtime: Runtime,
    model: str,
    prompts: Sequence[PromptSpec],
    *,
    repeat: int = 3,
    warmup: int = 1,
    observer: Observer | None = None,
    replayed_from: str | None = None,
    runtime_label: str | None = None,
) -> RunOutcome:
    """Run every prompt ``repeat`` times and summarise the result.

    ``warmup`` generations of the first prompt run first and are
    discarded. ``observer`` is called with a one-line description of each
    step, which is how ``--verbose`` reports progress without this module
    knowing what a terminal is.
    """
    if repeat < 1:
        raise UsageError(f"--repeat must be at least 1, got {repeat}")
    if warmup < 0:
        raise UsageError(f"--warmup cannot be negative, got {warmup}")
    if not prompts:
        raise UsageError("a benchmark needs at least one prompt")
    ordered = deduplicated(prompts, "prompts")

    say = observer if observer is not None else _silence
    runtime.probe()

    for index in range(warmup):
        say(f"warmup {index + 1}/{warmup} on prompt {ordered[0].name!r}")
        _generate(runtime, model, ordered[0])

    samples: list[Sample] = []
    streams: list[RecordedStream] = []
    notes: list[str] = []
    for prompt in ordered:
        for repetition in range(repeat):
            say(f"{prompt.name} run {repetition + 1}/{repeat}")
            timed = _generate(runtime, model, prompt)
            if timed.output_tokens == 0:
                notes.append(
                    f"prompt {prompt.name!r} repetition {repetition} produced no "
                    "output tokens; its throughput is excluded"
                )
            samples.append(
                Sample(
                    prompt=prompt.name,
                    repetition=repetition,
                    ttft_s=timed.ttft_s,
                    total_s=timed.total_s,
                    output_tokens=timed.output_tokens,
                    token_source=timed.token_source,
                    output_chars=timed.output_chars,
                    prompt_tokens=timed.prompt_tokens,
                )
            )
            streams.append(
                RecordedStream(
                    prompt=prompt.name,
                    repetition=repetition,
                    chunks=timed.chunks,
                    end_s=timed.total_s,
                    reported_output_tokens=(
                        timed.output_tokens
                        if timed.token_source is TokenSource.REPORTED
                        else None
                    ),
                    reported_prompt_tokens=timed.prompt_tokens,
                )
            )

    memory = _memory_or_none(runtime, model)
    label = runtime_label if runtime_label is not None else runtime.name
    report = BenchmarkReport(
        runtime=label,
        model=model,
        endpoint=runtime.endpoint,
        results=build_results(samples),
        memory=memory,
        warmup=warmup,
        replayed_from=replayed_from,
        notes=tuple(notes),
    )
    recording = Recording(
        runtime=label,
        model=model,
        endpoint=runtime.endpoint,
        streams=tuple(streams),
        memory=memory,
        warmup=warmup,
    )
    return RunOutcome(report=report, recording=recording)


def _memory_or_none(runtime: Runtime, model: str) -> MemoryReading | None:
    """Ask for a memory reading, but never fail a run over one.

    The timings are the measurement; memory is context. A server that
    answers ``/api/generate`` but not ``/api/ps`` should still produce a
    benchmark, with ``n/a`` where the number would be.
    """
    try:
        return runtime.memory(model)
    except Exception:  # deliberately non-fatal, see the docstring
        return None


def _silence(message: str) -> None:
    """The default observer: say nothing."""
