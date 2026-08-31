"""The measurement loop, against runtimes with a clock the test controls."""

from __future__ import annotations

import pytest

from conftest import ScriptedRuntime
from model_bench.errors import RuntimeUnavailable, UsageError
from model_bench.harness import run_benchmark
from model_bench.models import MemoryReading, PromptSpec, TokenSource

ONE = (PromptSpec("p", "hello", 8),)


def test_a_failed_probe_stops_the_run_before_any_measurement() -> None:
    # No report at all beats a report with an estimate in it.
    runtime = ScriptedRuntime(
        [[(0.1, "a")]], probe_error=RuntimeUnavailable("x", "http://h", "refused")
    )
    with pytest.raises(RuntimeUnavailable):
        run_benchmark(runtime, "m", ONE)
    assert runtime.calls == []


def test_time_to_first_token_is_the_first_chunk_that_carried_text() -> None:
    runtime = ScriptedRuntime([[(0.0, ""), (0.4, "hi"), (0.6, "!")]], end_gap=0.1)
    report = run_benchmark(runtime, "m", ONE, repeat=1, warmup=0).report
    assert report.result("p").ttft.median == pytest.approx(0.4)


def test_total_time_runs_to_the_end_of_the_stream() -> None:
    runtime = ScriptedRuntime([[(0.1, "a"), (0.5, "b")]], end_gap=0.25)
    report = run_benchmark(runtime, "m", ONE, repeat=1, warmup=0).report
    assert report.result("p").total.median == pytest.approx(0.75)


def test_a_reported_count_is_labelled_reported() -> None:
    runtime = ScriptedRuntime([[(0.1, "a")]], reported_tokens=42)
    report = run_benchmark(runtime, "m", ONE, repeat=1, warmup=0).report
    result = report.result("p")
    assert result.token_source is TokenSource.REPORTED
    assert result.output_tokens.median == 42


def test_an_unreported_count_falls_back_to_counting_chunks() -> None:
    runtime = ScriptedRuntime([[(0.1, "a"), (0.2, "b"), (0.3, "c")]])
    report = run_benchmark(runtime, "m", ONE, repeat=1, warmup=0).report
    result = report.result("p")
    assert result.token_source is TokenSource.CHUNKS
    assert result.output_tokens.median == 3


def test_empty_chunks_are_not_counted_as_tokens() -> None:
    runtime = ScriptedRuntime([[(0.1, "a"), (0.2, ""), (0.3, "b")]])
    report = run_benchmark(runtime, "m", ONE, repeat=1, warmup=0).report
    assert report.result("p").output_tokens.median == 2


def test_output_characters_are_summed() -> None:
    runtime = ScriptedRuntime([[(0.1, "abc"), (0.2, "de")]])
    outcome = run_benchmark(runtime, "m", ONE, repeat=1, warmup=0)
    assert outcome.report.result("p").samples[0].output_chars == 5


def test_a_prompt_token_count_is_carried_through() -> None:
    runtime = ScriptedRuntime([[(0.1, "a")]], prompt_tokens=11)
    outcome = run_benchmark(runtime, "m", ONE, repeat=1, warmup=0)
    assert outcome.report.result("p").samples[0].prompt_tokens == 11


def test_a_stream_with_no_text_is_recorded_with_a_note() -> None:
    # Dropping it would flatter the average; hiding it would mislead.
    runtime = ScriptedRuntime([[(0.3, "")]])
    report = run_benchmark(runtime, "m", ONE, repeat=1, warmup=0).report
    assert report.result("p").samples[0].output_tokens == 0
    assert report.notes and "produced no output tokens" in report.notes[0]


def test_a_stream_with_no_text_has_ttft_equal_to_total() -> None:
    runtime = ScriptedRuntime([[(0.3, "")]], end_gap=0.2)
    sample = run_benchmark(runtime, "m", ONE, repeat=1, warmup=0).report.result(
        "p"
    ).samples[0]
    assert sample.ttft_s == sample.total_s


def test_warmup_runs_happen_and_are_discarded() -> None:
    runtime = ScriptedRuntime([[(9.0, "slow")], [(0.1, "fast")]])
    outcome = run_benchmark(runtime, "m", ONE, repeat=1, warmup=1)
    assert len(runtime.calls) == 2
    assert outcome.report.total_samples == 1
    assert outcome.report.result("p").ttft.median == pytest.approx(0.1)


def test_the_warmup_count_is_recorded_in_the_report() -> None:
    runtime = ScriptedRuntime([[(0.1, "a")]])
    outcome = run_benchmark(runtime, "m", ONE, repeat=1, warmup=2)
    assert outcome.report.warmup == 2
    assert outcome.recording.warmup == 2


def test_warmup_uses_the_first_prompt() -> None:
    runtime = ScriptedRuntime([[(0.1, "a")]])
    prompts = (PromptSpec("first", "x"), PromptSpec("second", "y"))
    run_benchmark(runtime, "m", prompts, repeat=1, warmup=1)
    assert runtime.calls[0] == ("m", "first")


def test_zero_warmup_measures_immediately() -> None:
    runtime = ScriptedRuntime([[(0.1, "a")]])
    run_benchmark(runtime, "m", ONE, repeat=1, warmup=0)
    assert len(runtime.calls) == 1


def test_every_prompt_is_run_the_requested_number_of_times() -> None:
    runtime = ScriptedRuntime([[(0.1, "a")]])
    prompts = (PromptSpec("a", "x"), PromptSpec("b", "y"))
    outcome = run_benchmark(runtime, "m", prompts, repeat=3, warmup=0)
    assert outcome.report.total_samples == 6
    assert [result.prompt for result in outcome.report.results] == ["a", "b"]


def test_repetitions_are_numbered_from_zero() -> None:
    runtime = ScriptedRuntime([[(0.1, "a")]])
    outcome = run_benchmark(runtime, "m", ONE, repeat=3, warmup=0)
    numbers = [s.repetition for s in outcome.report.result("p").samples]
    assert numbers == [0, 1, 2]


def test_repeat_below_one_is_refused() -> None:
    with pytest.raises(UsageError, match="--repeat must be at least 1"):
        run_benchmark(ScriptedRuntime([[(0.1, "a")]]), "m", ONE, repeat=0)


def test_negative_warmup_is_refused() -> None:
    with pytest.raises(UsageError, match="--warmup cannot be negative"):
        run_benchmark(ScriptedRuntime([[(0.1, "a")]]), "m", ONE, warmup=-1)


def test_no_prompts_is_refused() -> None:
    with pytest.raises(UsageError, match="at least one prompt"):
        run_benchmark(ScriptedRuntime([[(0.1, "a")]]), "m", ())


def test_duplicate_prompt_names_are_refused() -> None:
    prompts = (PromptSpec("p", "x"), PromptSpec("p", "y"))
    with pytest.raises(UsageError, match="duplicate prompt name"):
        run_benchmark(ScriptedRuntime([[(0.1, "a")]]), "m", prompts)


def test_the_observer_sees_warmup_and_every_run() -> None:
    messages: list[str] = []
    runtime = ScriptedRuntime([[(0.1, "a")]])
    run_benchmark(
        runtime, "m", ONE, repeat=2, warmup=1, observer=messages.append
    )
    assert messages == ["warmup 1/1 on prompt 'p'", "p run 1/2", "p run 2/2"]


def test_no_observer_is_fine() -> None:
    runtime = ScriptedRuntime([[(0.1, "a")]])
    assert run_benchmark(runtime, "m", ONE, repeat=1, warmup=0).report is not None


def test_memory_is_attached_to_the_report_and_the_recording() -> None:
    reading = MemoryReading(2048, "/api/ps", 1024)
    runtime = ScriptedRuntime([[(0.1, "a")]], memory=reading)
    outcome = run_benchmark(runtime, "m", ONE, repeat=1, warmup=0)
    assert outcome.report.memory == reading
    assert outcome.recording.memory == reading


def test_a_memory_lookup_failure_does_not_fail_the_run() -> None:
    # The timings are the measurement; memory is context.
    runtime = ScriptedRuntime([[(0.1, "a")]], memory_error=OSError("no /api/ps"))
    outcome = run_benchmark(runtime, "m", ONE, repeat=1, warmup=0)
    assert outcome.report.memory is None


def test_the_report_uses_the_runtime_name_by_default() -> None:
    outcome = run_benchmark(ScriptedRuntime([[(0.1, "a")]]), "m", ONE, warmup=0)
    assert outcome.report.runtime == "scripted"


def test_a_runtime_label_can_override_the_name() -> None:
    outcome = run_benchmark(
        ScriptedRuntime([[(0.1, "a")]]), "m", ONE, warmup=0, runtime_label="ollama"
    )
    assert outcome.report.runtime == "ollama"


def test_the_endpoint_is_carried_into_both_outputs() -> None:
    runtime = ScriptedRuntime([[(0.1, "a")]], endpoint="http://elsewhere/gen")
    outcome = run_benchmark(runtime, "m", ONE, repeat=1, warmup=0)
    assert outcome.report.endpoint == "http://elsewhere/gen"
    assert outcome.recording.endpoint == "http://elsewhere/gen"


def test_replayed_from_is_absent_for_a_live_run() -> None:
    outcome = run_benchmark(ScriptedRuntime([[(0.1, "a")]]), "m", ONE, warmup=0)
    assert outcome.report.is_replay is False


def test_replayed_from_is_recorded_when_given() -> None:
    outcome = run_benchmark(
        ScriptedRuntime([[(0.1, "a")]]), "m", ONE, warmup=0, replayed_from="run.json"
    )
    assert outcome.report.replayed_from == "run.json"


def test_the_recording_captures_only_chunks_that_carried_text() -> None:
    runtime = ScriptedRuntime([[(0.1, "a"), (0.2, ""), (0.3, "b")]], end_gap=0.1)
    recording = run_benchmark(runtime, "m", ONE, repeat=1, warmup=0).recording
    stream = recording.streams[0]
    assert [chunk.text for chunk in stream.chunks] == ["a", "b"]
    assert stream.chunks[0].at_s == pytest.approx(0.1)
    assert stream.end_s == pytest.approx(0.4)


def test_the_recording_omits_a_token_count_it_only_guessed() -> None:
    # Replaying it must not upgrade an estimate into a reported figure.
    runtime = ScriptedRuntime([[(0.1, "a")]])
    recording = run_benchmark(runtime, "m", ONE, repeat=1, warmup=0).recording
    assert recording.streams[0].reported_output_tokens is None


def test_the_recording_keeps_a_reported_token_count() -> None:
    runtime = ScriptedRuntime([[(0.1, "a")]], reported_tokens=17, prompt_tokens=5)
    recording = run_benchmark(runtime, "m", ONE, repeat=1, warmup=0).recording
    assert recording.streams[0].reported_output_tokens == 17
    assert recording.streams[0].reported_prompt_tokens == 5


def test_the_recording_excludes_warmup_runs() -> None:
    runtime = ScriptedRuntime([[(0.1, "a")]])
    recording = run_benchmark(runtime, "m", ONE, repeat=2, warmup=3).recording
    assert len(recording.streams) == 2


def test_a_run_is_replayable_from_its_own_recording() -> None:
    from model_bench.runtimes import ReplayRuntime

    runtime = ScriptedRuntime(
        [[(0.1, "a"), (0.4, "b"), (0.9, "c")]], end_gap=0.1, reported_tokens=3
    )
    original = run_benchmark(runtime, "m", ONE, repeat=2, warmup=0)
    replayed = run_benchmark(
        ReplayRuntime(original.recording),
        "m",
        ONE,
        repeat=2,
        warmup=0,
        replayed_from="memory",
    )
    assert replayed.report.result("p").ttft.median == pytest.approx(
        original.report.result("p").ttft.median
    )
    assert replayed.report.result("p").total.median == pytest.approx(
        original.report.result("p").total.median
    )
