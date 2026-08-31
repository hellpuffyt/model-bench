"""Samples, results and reports — including what they refuse to be."""

from __future__ import annotations

import pytest

from conftest import make_sample
from model_bench.errors import UsageError
from model_bench.models import (
    BenchmarkReport,
    Chunk,
    MemoryReading,
    PromptResult,
    PromptSpec,
    TokenSource,
    build_results,
)


def test_token_source_str_is_the_bare_word() -> None:
    assert str(TokenSource.REPORTED) == "reported"
    assert str(TokenSource.CHUNKS) == "chunks"


def test_chunk_defaults_to_not_done_and_no_counts() -> None:
    chunk = Chunk(text="hi")
    assert chunk.done is False
    assert chunk.output_tokens is None
    assert chunk.prompt_tokens is None


def test_memory_resident_subtracts_vram() -> None:
    reading = MemoryReading(total_bytes=1000, vram_bytes=600, source="x")
    assert reading.resident_bytes == 400


def test_memory_resident_is_the_total_when_vram_is_unknown() -> None:
    assert MemoryReading(total_bytes=1000, source="x").resident_bytes == 1000


def test_memory_resident_never_goes_negative() -> None:
    # Ollama can report size_vram equal to or above size; clamping keeps
    # the derived figure meaningful rather than nonsensical.
    reading = MemoryReading(total_bytes=100, vram_bytes=500, source="x")
    assert reading.resident_bytes == 0


def test_memory_as_dict_round_trips_the_source() -> None:
    payload = MemoryReading(total_bytes=1, vram_bytes=None, source="/api/ps").as_dict()
    assert payload == {"total_bytes": 1, "vram_bytes": None, "source": "/api/ps"}


def test_prompt_spec_rejects_a_blank_name() -> None:
    with pytest.raises(UsageError, match="non-empty name"):
        PromptSpec(name="   ", text="hello")


def test_prompt_spec_rejects_empty_text() -> None:
    with pytest.raises(UsageError, match="no text"):
        PromptSpec(name="p", text="")


def test_prompt_spec_rejects_a_zero_token_cap() -> None:
    with pytest.raises(UsageError, match="at least 1"):
        PromptSpec(name="p", text="hi", max_tokens=0)


def test_prompt_spec_as_dict() -> None:
    assert PromptSpec("p", "hi", 4).as_dict() == {
        "name": "p",
        "text": "hi",
        "max_tokens": 4,
    }


def test_sample_rejects_negative_time() -> None:
    with pytest.raises(UsageError, match="negative elapsed"):
        make_sample(ttft_s=-0.1)


def test_sample_rejects_finishing_before_its_first_token() -> None:
    with pytest.raises(UsageError, match="before its first token"):
        make_sample(ttft_s=2.0, total_s=1.0)


def test_sample_rejects_a_negative_token_count() -> None:
    with pytest.raises(UsageError, match="negative token count"):
        make_sample(output_tokens=-1)


def test_sample_allows_ttft_equal_to_total() -> None:
    # A stream that produced nothing has ttft == total by definition.
    assert make_sample(ttft_s=1.0, total_s=1.0, output_tokens=0).decode_s == 0.0


def test_decode_time_excludes_the_first_token_wait() -> None:
    assert make_sample(ttft_s=0.25, total_s=1.25).decode_s == pytest.approx(1.0)


def test_tokens_per_second_excludes_the_first_token() -> None:
    # 101 tokens in 1.0 s of decode: the first arrived at ttft, so 100
    # were produced during the measured decode window.
    sample = make_sample(ttft_s=0.5, total_s=1.5, output_tokens=101)
    assert sample.tokens_per_second == pytest.approx(100.0)


def test_tokens_per_second_is_none_for_a_single_token() -> None:
    assert make_sample(output_tokens=1).tokens_per_second is None


def test_tokens_per_second_is_none_for_zero_tokens() -> None:
    assert make_sample(output_tokens=0).tokens_per_second is None


def test_tokens_per_second_is_none_when_decode_took_no_time() -> None:
    # Rather than an infinity derived from timer resolution.
    sample = make_sample(ttft_s=1.0, total_s=1.0, output_tokens=50)
    assert sample.tokens_per_second is None


def test_overall_tokens_per_second_includes_the_first_token_wait() -> None:
    sample = make_sample(ttft_s=0.5, total_s=2.0, output_tokens=100)
    assert sample.overall_tokens_per_second == pytest.approx(50.0)


def test_overall_tokens_per_second_is_none_without_tokens() -> None:
    assert make_sample(output_tokens=0).overall_tokens_per_second is None


def test_overall_tokens_per_second_is_none_without_elapsed_time() -> None:
    sample = make_sample(ttft_s=0.0, total_s=0.0, output_tokens=5)
    assert sample.overall_tokens_per_second is None


def test_sample_as_dict_carries_the_token_source() -> None:
    payload = make_sample(token_source=TokenSource.CHUNKS).as_dict()
    assert payload["token_source"] == "chunks"
    assert payload["tokens_per_second"] is not None


def test_prompt_result_rejects_no_samples() -> None:
    with pytest.raises(UsageError, match="no samples"):
        PromptResult("p", ())


def test_prompt_result_summarises_ttft() -> None:
    result = PromptResult(
        "p",
        (
            make_sample(ttft_s=0.1, repetition=0),
            make_sample(ttft_s=0.3, repetition=1),
        ),
    )
    assert result.ttft.median == pytest.approx(0.2)
    assert result.total.count == 2


def test_prompt_result_throughput_skips_unmeasurable_samples() -> None:
    result = PromptResult(
        "p",
        (
            make_sample(ttft_s=0.0, total_s=1.0, output_tokens=101),
            make_sample(output_tokens=1, repetition=1),
        ),
    )
    throughput = result.throughput
    assert throughput is not None
    assert throughput.count == 1
    assert throughput.mean == pytest.approx(100.0)


def test_prompt_result_throughput_is_none_when_nothing_is_measurable() -> None:
    result = PromptResult("p", (make_sample(output_tokens=0),))
    assert result.throughput is None


def test_prompt_result_token_source_degrades_to_chunks() -> None:
    # One estimated count makes the whole row an estimate.
    result = PromptResult(
        "p",
        (
            make_sample(token_source=TokenSource.REPORTED),
            make_sample(repetition=1, token_source=TokenSource.CHUNKS),
        ),
    )
    assert result.token_source is TokenSource.CHUNKS


def test_prompt_result_token_source_stays_reported_when_all_are() -> None:
    result = PromptResult("p", (make_sample(), make_sample(repetition=1)))
    assert result.token_source is TokenSource.REPORTED


def test_prompt_result_as_dict_includes_every_sample() -> None:
    result = PromptResult("p", (make_sample(), make_sample(repetition=1)))
    payload = result.as_dict()
    assert payload["repetitions"] == 2
    assert len(payload["samples"]) == 2
    assert payload["tokens_per_second"]["count"] == 2


def test_prompt_result_as_dict_nulls_unmeasurable_throughput() -> None:
    payload = PromptResult("p", (make_sample(output_tokens=0),)).as_dict()
    assert payload["tokens_per_second"] is None


def _report(**kwargs: object) -> BenchmarkReport:
    defaults: dict[str, object] = {
        "runtime": "scripted",
        "model": "m",
        "endpoint": "http://x",
        "results": (PromptResult("p", (make_sample(),)),),
    }
    defaults.update(kwargs)
    return BenchmarkReport(**defaults)  # type: ignore[arg-type]


def test_report_rejects_no_results() -> None:
    with pytest.raises(UsageError, match="at least one prompt"):
        _report(results=())


def test_report_rejects_duplicate_prompts() -> None:
    duplicate = (PromptResult("p", (make_sample(),)), PromptResult("p", (make_sample(),)))
    with pytest.raises(UsageError, match="duplicate prompt"):
        _report(results=duplicate)


def test_report_is_not_a_replay_by_default() -> None:
    assert _report().is_replay is False


def test_report_knows_it_is_a_replay() -> None:
    assert _report(replayed_from="run.json").is_replay is True


def test_report_counts_all_samples() -> None:
    results = (
        PromptResult("a", (make_sample(prompt="a"), make_sample(prompt="a"))),
        PromptResult("b", (make_sample(prompt="b"),)),
    )
    assert _report(results=results).total_samples == 3


def test_report_looks_up_a_prompt_by_name() -> None:
    assert _report().result("p").prompt == "p"


def test_report_lists_what_it_has_for_an_unknown_prompt() -> None:
    with pytest.raises(UsageError, match="it has: p"):
        _report().result("nope")


def test_report_as_dict_declares_its_schema() -> None:
    payload = _report(replayed_from="run.json", notes=("careful",)).as_dict()
    assert payload["schema"] == "model-bench/report/1"
    assert payload["replayed_from"] == "run.json"
    assert payload["notes"] == ["careful"]
    assert payload["memory"] is None


def test_report_as_dict_includes_memory_when_present() -> None:
    memory = MemoryReading(total_bytes=5, source="s")
    assert _report(memory=memory).as_dict()["memory"]["total_bytes"] == 5


def test_build_results_groups_by_prompt() -> None:
    samples = [
        make_sample(prompt="a", repetition=0),
        make_sample(prompt="b", repetition=0),
        make_sample(prompt="a", repetition=1),
    ]
    results = build_results(samples)
    assert [result.prompt for result in results] == ["a", "b"]
    assert len(results[0].samples) == 2


def test_build_results_preserves_first_appearance_order() -> None:
    samples = [make_sample(prompt="z"), make_sample(prompt="a")]
    assert [result.prompt for result in build_results(samples)] == ["z", "a"]


def test_build_results_of_nothing_is_empty() -> None:
    assert build_results([]) == ()
