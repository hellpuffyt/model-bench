"""Turning a :class:`~model_bench.models.BenchmarkReport` into output.

Four formats, one rule: every number printed is one that was measured,
and anything that was not measured prints ``n/a`` rather than ``0``.
That is why ``tok/s`` can be blank for a prompt whose model emitted a
single token, and why the memory line disappears entirely on runtimes
that cannot report memory.

The header block is as much of the output as the table, because a
throughput figure without the model, the endpoint, the warm-up count and
the token-count provenance beside it is not reproducible and therefore
not worth much. When a report came from a recording, the header says so
on its own line — replayed numbers describe the machine that recorded
them, not the machine printing them.

Alignment is computed from the data rather than hard-coded, so the table
stays square when a model name or prompt name is long. The formats are
byte-stable for identical input, which is what lets CI assert exact
output against a committed fixture.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence

from . import __version__
from .errors import UsageError
from .models import BenchmarkReport, MemoryReading, PromptResult
from .stats import Summary

FORMATS = ("table", "json", "markdown", "csv")

_HEADERS = (
    "prompt",
    "runs",
    "ttft p50",
    "ttft p90",
    "tok/s mean",
    "tok/s sd",
    "total p50",
    "tokens",
)


def render(report: BenchmarkReport, fmt: str = "table") -> str:
    """Render ``report`` in one of :data:`FORMATS`."""
    if fmt == "table":
        return render_table(report)
    if fmt == "json":
        return render_json(report)
    if fmt == "markdown":
        return render_markdown(report)
    if fmt == "csv":
        return render_csv(report)
    raise UsageError(f"unknown format {fmt!r}; choose one of: {', '.join(FORMATS)}")


def human_bytes(count: int) -> str:
    """Binary-prefixed size, because that is how model weights are sized."""
    if count < 0:
        raise UsageError(f"a byte count cannot be negative, got {count}")
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(count)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.2f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")  # pragma: no cover


def describe_memory(memory: MemoryReading | None) -> str:
    """The memory line, or an explicit ``n/a``."""
    if memory is None:
        return "n/a (this runtime does not report it)"
    total = human_bytes(memory.total_bytes)
    if memory.vram_bytes is None:
        return f"{total} resident ({memory.source})"
    vram = human_bytes(memory.vram_bytes)
    return f"{total} resident, {vram} in VRAM ({memory.source})"


def _seconds(value: float) -> str:
    return f"{value:.3f}"


def _rate(value: float) -> str:
    return f"{value:.1f}"


def _optional_rate(summary: Summary | None, attribute: str) -> str:
    if summary is None:
        return "n/a"
    value = getattr(summary, attribute)
    if value is None:
        return "n/a"
    return _rate(float(value))


def _row(result: PromptResult) -> tuple[str, ...]:
    throughput = result.throughput
    return (
        result.prompt,
        str(len(result.samples)),
        _seconds(result.ttft.median),
        _seconds(result.ttft.p90),
        _optional_rate(throughput, "mean"),
        _optional_rate(throughput, "stdev"),
        _seconds(result.total.median),
        f"{result.output_tokens.median:.0f}",
    )


def _header_lines(report: BenchmarkReport) -> list[str]:
    lines = [
        f"model-bench {__version__}",
        f"runtime   {report.runtime}",
        f"model     {report.model}",
        f"endpoint  {report.endpoint}",
        f"memory    {describe_memory(report.memory)}",
        f"warmup    {report.warmup} discarded run(s)",
        f"tokens    {report.results[0].token_source} by the runtime",
    ]
    if report.replayed_from is not None:
        lines.insert(1, f"source    replay of {report.replayed_from} (not a live run)")
    return lines


def _widths(rows: Sequence[Sequence[str]]) -> list[int]:
    """Column widths: the widest cell, never narrower than its heading."""
    return [
        max([len(heading)] + [len(row[index]) for row in rows])
        for index, heading in enumerate(_HEADERS)
    ]


def render_table(report: BenchmarkReport) -> str:
    """A fixed-width table for a terminal."""
    rows = [_row(result) for result in report.results]
    widths = _widths(rows)
    lines = _header_lines(report)
    lines.append("")
    lines.append(_align(_HEADERS, widths))
    lines.append("  ".join("-" * width for width in widths))
    lines.extend(_align(row, widths) for row in rows)
    for note in report.notes:
        lines.append(f"note: {note}")
    return "\n".join(lines) + "\n"


def _align(row: Sequence[str], widths: Sequence[int]) -> str:
    cells = [row[0].ljust(widths[0])]
    cells.extend(
        value.rjust(width) for value, width in zip(row[1:], widths[1:], strict=False)
    )
    return "  ".join(cells).rstrip()


def render_markdown(report: BenchmarkReport) -> str:
    """A GitHub-flavoured table, for pasting into an issue or a PR."""
    lines = ["# model-bench", ""]
    for line in _header_lines(report)[1:]:
        label, _, value = line.partition("  ")
        lines.append(f"- **{label.strip()}**: {value.strip()}")
    lines.append("")
    lines.append("| " + " | ".join(_HEADERS) + " |")
    lines.append("| " + " | ".join("---" for _ in _HEADERS) + " |")
    for result in report.results:
        lines.append("| " + " | ".join(_row(result)) + " |")
    if report.notes:
        lines.append("")
        for note in report.notes:
            lines.append(f"> note: {note}")
    return "\n".join(lines) + "\n"


def render_json(report: BenchmarkReport) -> str:
    """The whole report, including every individual sample."""
    payload = report.as_dict()
    payload["tool_version"] = __version__
    return json.dumps(payload, indent=2) + "\n"


def render_csv(report: BenchmarkReport) -> str:
    """One row per *sample*, for loading into a spreadsheet.

    The other formats summarise; this one does not, because a reader who
    reaches for CSV wants to do their own statistics and a summary they
    cannot check is not much use to them.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "runtime",
            "model",
            "prompt",
            "repetition",
            "ttft_s",
            "total_s",
            "output_tokens",
            "token_source",
            "tokens_per_second",
        ]
    )
    for result in report.results:
        for sample in result.samples:
            rate = sample.tokens_per_second
            writer.writerow(
                [
                    report.runtime,
                    report.model,
                    sample.prompt,
                    sample.repetition,
                    f"{sample.ttft_s:.6f}",
                    f"{sample.total_s:.6f}",
                    sample.output_tokens,
                    sample.token_source.value,
                    "" if rate is None else f"{rate:.4f}",
                ]
            )
    return buffer.getvalue()
