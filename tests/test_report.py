"""The four output formats, and what they say when a number is missing."""

from __future__ import annotations

import csv
import io
import json

import pytest

from conftest import make_sample
from model_bench.errors import UsageError
from model_bench.models import (
    BenchmarkReport,
    MemoryReading,
    PromptResult,
    TokenSource,
)
from model_bench.report import (
    FORMATS,
    describe_memory,
    human_bytes,
    render,
    render_csv,
    render_json,
    render_markdown,
    render_table,
)


def report(**kwargs: object) -> BenchmarkReport:
    defaults: dict[str, object] = {
        "runtime": "ollama",
        "model": "llama3.2:3b",
        "endpoint": "http://127.0.0.1:11434/api/generate",
        "results": (
            PromptResult(
                "tiny",
                (
                    make_sample(
                        prompt="tiny", ttft_s=0.02, total_s=0.20, output_tokens=11
                    ),
                    make_sample(
                        prompt="tiny",
                        repetition=1,
                        ttft_s=0.04,
                        total_s=0.24,
                        output_tokens=11,
                    ),
                ),
            ),
        ),
    }
    defaults.update(kwargs)
    return BenchmarkReport(**defaults)  # type: ignore[arg-type]


# --- byte formatting --------------------------------------------------


def test_bytes_below_a_kibibyte_are_exact() -> None:
    assert human_bytes(512) == "512 B"


def test_zero_bytes() -> None:
    assert human_bytes(0) == "0 B"


def test_kibibytes_are_binary_not_decimal() -> None:
    assert human_bytes(1024) == "1.00 KiB"


def test_a_model_sized_number_reads_in_gibibytes() -> None:
    assert human_bytes(2_557_000_000) == "2.38 GiB"


def test_very_large_sizes_stop_at_tebibytes() -> None:
    assert human_bytes(5 * 1024**4) == "5.00 TiB"


def test_beyond_tebibytes_still_uses_tebibytes() -> None:
    assert human_bytes(2048 * 1024**4).endswith("TiB")


def test_a_negative_size_is_refused() -> None:
    with pytest.raises(UsageError, match="cannot be negative"):
        human_bytes(-1)


# --- memory description -----------------------------------------------


def test_absent_memory_says_why() -> None:
    assert describe_memory(None) == "n/a (this runtime does not report it)"


def test_memory_without_vram_reports_only_the_total() -> None:
    text = describe_memory(MemoryReading(1024, "/api/ps"))
    assert text == "1.00 KiB resident (/api/ps)"


def test_memory_with_vram_reports_both() -> None:
    text = describe_memory(MemoryReading(2048, "/api/ps", 1024))
    assert "2.00 KiB resident" in text
    assert "1.00 KiB in VRAM" in text


# --- table ------------------------------------------------------------


def test_the_table_leads_with_the_tool_version() -> None:
    assert render_table(report()).startswith("model-bench ")


def test_the_table_names_the_model_and_endpoint() -> None:
    text = render_table(report())
    assert "model     llama3.2:3b" in text
    assert "endpoint  http://127.0.0.1:11434/api/generate" in text


def test_the_table_states_where_token_counts_came_from() -> None:
    assert "tokens    reported by the runtime" in render_table(report())


def test_the_table_says_when_counts_were_only_estimated() -> None:
    results = (
        PromptResult(
            "tiny", (make_sample(prompt="tiny", token_source=TokenSource.CHUNKS),)
        ),
    )
    assert "tokens    chunks by the runtime" in render_table(report(results=results))


def test_the_table_reports_missing_memory_as_not_available() -> None:
    assert "memory    n/a" in render_table(report())


def test_the_table_shows_a_memory_reading_when_there_is_one() -> None:
    text = render_table(report(memory=MemoryReading(2_557_000_000, "/api/ps")))
    assert "memory    2.38 GiB resident (/api/ps)" in text


def test_the_table_marks_a_replay_on_its_own_line() -> None:
    text = render_table(report(replayed_from="run.json"))
    assert "source    replay of run.json (not a live run)" in text
    assert text.index("source") < text.index("runtime")


def test_a_live_table_has_no_source_line() -> None:
    assert "source" not in render_table(report())


def test_the_table_has_a_header_row_and_a_rule() -> None:
    lines = render_table(report()).splitlines()
    header = next(line for line in lines if line.startswith("prompt"))
    rule = lines[lines.index(header) + 1]
    assert set(rule.replace(" ", "")) == {"-"}


def test_the_table_columns_line_up_with_the_headings() -> None:
    lines = render_table(report()).splitlines()
    header = next(line for line in lines if line.startswith("prompt"))
    row = lines[-1]
    assert header.index("runs") + len("runs") == row.index("2") + 1


def test_the_table_widens_for_a_long_prompt_name() -> None:
    long_name = "a-very-long-prompt-name"
    results = (PromptResult(long_name, (make_sample(prompt=long_name),)),)
    lines = render_table(report(results=results)).splitlines()
    header = next(line for line in lines if line.startswith("prompt"))
    assert header.index("runs") == len(long_name) + 2


def test_the_table_prints_medians_to_three_decimals() -> None:
    row = render_table(report()).splitlines()[-1]
    assert "0.030" in row  # median of 0.02 and 0.04


def test_the_table_prints_n_a_for_unmeasurable_throughput() -> None:
    results = (PromptResult("tiny", (make_sample(prompt="tiny", output_tokens=1),)),)
    assert "n/a" in render_table(report(results=results)).splitlines()[-1]


def test_the_table_prints_n_a_for_a_missing_standard_deviation() -> None:
    results = (PromptResult("tiny", (make_sample(prompt="tiny"),)),)
    row = render_table(report(results=results)).splitlines()[-1]
    assert row.split().count("n/a") == 1


def test_the_table_appends_notes() -> None:
    text = render_table(report(notes=("careful with this one",)))
    assert text.rstrip().endswith("note: careful with this one")


def test_the_table_ends_with_a_newline() -> None:
    assert render_table(report()).endswith("\n")


def test_the_table_is_byte_stable() -> None:
    # CI asserts exact output; identical input must render identically.
    assert render_table(report()) == render_table(report())


# --- markdown ---------------------------------------------------------


def test_markdown_has_a_heading() -> None:
    assert render_markdown(report()).startswith("# model-bench\n")


def test_markdown_lists_the_metadata_as_bullets() -> None:
    text = render_markdown(report())
    assert "- **model**: llama3.2:3b" in text


def test_markdown_marks_a_replay() -> None:
    assert "- **source**: replay of run.json" in render_markdown(
        report(replayed_from="run.json")
    )


def test_markdown_builds_a_pipe_table() -> None:
    lines = render_markdown(report()).splitlines()
    header = next(line for line in lines if line.startswith("| prompt"))
    assert lines[lines.index(header) + 1].startswith("| ---")


def test_markdown_quotes_notes() -> None:
    assert "> note: hmm" in render_markdown(report(notes=("hmm",)))


def test_markdown_without_notes_has_no_quote_block() -> None:
    assert ">" not in render_markdown(report())


# --- json -------------------------------------------------------------


def test_json_is_parseable_and_declares_its_schema() -> None:
    payload = json.loads(render_json(report()))
    assert payload["schema"] == "model-bench/report/1"
    assert payload["tool_version"]


def test_json_includes_every_individual_sample() -> None:
    payload = json.loads(render_json(report()))
    assert len(payload["prompts"][0]["samples"]) == 2


def test_json_nulls_a_missing_memory_reading() -> None:
    assert json.loads(render_json(report()))["memory"] is None


def test_json_records_a_replay_source() -> None:
    payload = json.loads(render_json(report(replayed_from="run.json")))
    assert payload["replayed_from"] == "run.json"


# --- csv --------------------------------------------------------------


def test_csv_has_one_row_per_sample_not_per_prompt() -> None:
    rows = list(csv.reader(io.StringIO(render_csv(report()))))
    assert rows[0][0] == "runtime"
    assert len(rows) == 3


def test_csv_carries_the_runtime_and_model_on_every_row() -> None:
    rows = list(csv.reader(io.StringIO(render_csv(report()))))
    assert rows[1][:3] == ["ollama", "llama3.2:3b", "tiny"]


def test_csv_leaves_unmeasurable_throughput_blank() -> None:
    # Blank, not zero: zero would be a claim about the runtime.
    results = (PromptResult("tiny", (make_sample(prompt="tiny", output_tokens=1),)),)
    rows = list(csv.reader(io.StringIO(render_csv(report(results=results)))))
    assert rows[1][-1] == ""


def test_csv_records_the_token_source_per_sample() -> None:
    rows = list(csv.reader(io.StringIO(render_csv(report()))))
    assert rows[1][7] == "reported"


def test_csv_uses_unix_line_endings_everywhere() -> None:
    # csv.writer defaults to \r\n, which would make CI comparisons
    # platform-dependent.
    assert "\r" not in render_csv(report())


# --- dispatch ---------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS)
def test_every_advertised_format_renders(fmt: str) -> None:
    assert render(report(), fmt)


def test_the_default_format_is_the_table() -> None:
    assert render(report()) == render_table(report())


def test_an_unknown_format_lists_the_real_ones() -> None:
    with pytest.raises(UsageError, match="choose one of: table, json"):
        render(report(), "yaml")
