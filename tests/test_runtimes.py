"""The three backends, against fake sockets and recorded streams."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from conftest import FakeOpener, ndjson, sse
from model_bench.errors import ProtocolError, RecordingError, RuntimeUnavailable
from model_bench.models import Chunk, PromptSpec
from model_bench.recording import RecordedChunk, RecordedStream, Recording
from model_bench.runtime import Runtime
from model_bench.runtimes import (
    DEFAULT_ENDPOINTS,
    LIVE_RUNTIMES,
    OllamaRuntime,
    OpenAICompatRuntime,
    ReplayRuntime,
    VirtualClock,
)
from model_bench.runtimes.ollama import read_memory

PROMPT = PromptSpec("p", "say something", max_tokens=32)


# --- registry ---------------------------------------------------------


def test_replay_is_not_a_selectable_live_runtime() -> None:
    # A recording must never be reachable through the flag that means
    # "measure this machine".
    assert "replay" not in LIVE_RUNTIMES


def test_every_live_runtime_has_a_default_endpoint() -> None:
    assert set(DEFAULT_ENDPOINTS) == set(LIVE_RUNTIMES)


# --- ollama -----------------------------------------------------------


def test_ollama_endpoint_is_the_generate_route() -> None:
    assert OllamaRuntime("http://h:1/").endpoint == "http://h:1/api/generate"


def test_ollama_strips_trailing_slashes_from_the_host() -> None:
    assert OllamaRuntime("http://h:1///").host == "http://h:1"


def test_ollama_probe_accepts_a_model_list() -> None:
    OllamaRuntime("http://h", opener=FakeOpener('{"models": []}')).probe()


def test_ollama_probe_rejects_something_else_listening_on_the_port() -> None:
    opener = FakeOpener('{"hello": "world"}')
    with pytest.raises(RuntimeUnavailable, match="did not look like an Ollama"):
        OllamaRuntime("http://h", opener=opener).probe()


def test_ollama_probe_rejects_a_json_array() -> None:
    with pytest.raises(RuntimeUnavailable):
        OllamaRuntime("http://h", opener=FakeOpener("[]")).probe()


def test_ollama_lists_available_models() -> None:
    body = '{"models": [{"name": "a"}, {"name": "b"}]}'
    runtime = OllamaRuntime("http://h", opener=FakeOpener(body))
    assert runtime.available_models() == ["a", "b"]


def test_ollama_model_list_skips_malformed_entries() -> None:
    body = '{"models": [{"name": "a"}, 7, {"size": 1}]}'
    runtime = OllamaRuntime("http://h", opener=FakeOpener(body))
    assert runtime.available_models() == ["a"]


def test_ollama_model_list_is_empty_for_a_non_object_response() -> None:
    assert OllamaRuntime("http://h", opener=FakeOpener("[]")).available_models() == []


def test_ollama_model_list_is_empty_when_models_is_not_a_list() -> None:
    opener = FakeOpener('{"models": 3}')
    assert OllamaRuntime("http://h", opener=opener).available_models() == []


def test_ollama_streams_text_chunks_then_a_final_count() -> None:
    body = ndjson(
        {"response": "Hel", "done": False},
        {"response": "lo", "done": False},
        {"response": "", "done": True, "eval_count": 7, "prompt_eval_count": 3},
    )
    runtime = OllamaRuntime("http://h", opener=FakeOpener(body))
    chunks = list(runtime.stream("m", PROMPT))
    assert [chunk.text for chunk in chunks] == ["Hel", "lo", ""]
    assert chunks[-1].done is True
    assert chunks[-1].output_tokens == 7
    assert chunks[-1].prompt_tokens == 3


def test_ollama_pins_temperature_and_seed_for_repeatable_work() -> None:
    opener = FakeOpener(ndjson({"response": "x", "done": True}))
    runtime = OllamaRuntime("http://h", seed=42, opener=opener)
    list(runtime.stream("m", PROMPT))
    options = opener.payload["options"]
    assert options == {"temperature": 0, "seed": 42, "num_predict": 32}
    assert opener.payload["stream"] is True
    assert opener.payload["prompt"] == "say something"


def test_ollama_reports_a_mid_stream_error_rather_than_a_fast_empty_run() -> None:
    body = ndjson({"error": 'model "ghost" not found'})
    runtime = OllamaRuntime("http://h", opener=FakeOpener(body))
    with pytest.raises(RuntimeUnavailable, match="ghost"):
        list(runtime.stream("ghost", PROMPT))


def test_ollama_rejects_a_non_json_line() -> None:
    runtime = OllamaRuntime("http://h", opener=FakeOpener("not json\n"))
    with pytest.raises(ProtocolError, match="not JSON"):
        list(runtime.stream("m", PROMPT))


def test_ollama_rejects_a_json_line_that_is_not_an_object() -> None:
    runtime = OllamaRuntime("http://h", opener=FakeOpener("[1, 2]\n"))
    with pytest.raises(ProtocolError, match="expected a JSON object"):
        list(runtime.stream("m", PROMPT))


def test_ollama_ignores_a_boolean_where_a_count_belongs() -> None:
    # True is an int in Python; treating it as a token count would be
    # a silent one-token benchmark.
    body = ndjson({"response": "", "done": True, "eval_count": True})
    runtime = OllamaRuntime("http://h", opener=FakeOpener(body))
    assert list(runtime.stream("m", PROMPT))[-1].output_tokens is None


def test_ollama_ignores_a_float_token_count() -> None:
    body = ndjson({"response": "", "done": True, "eval_count": 4.5})
    runtime = OllamaRuntime("http://h", opener=FakeOpener(body))
    assert list(runtime.stream("m", PROMPT))[-1].output_tokens is None


def test_ollama_memory_finds_the_loaded_model() -> None:
    body = '{"models": [{"name": "m", "size": 2048, "size_vram": 1024}]}'
    runtime = OllamaRuntime("http://h", opener=FakeOpener(body))
    reading = runtime.memory("m")
    assert reading is not None
    assert reading.total_bytes == 2048
    assert reading.vram_bytes == 1024
    assert reading.source == "http://h/api/ps"


def test_ollama_memory_is_none_when_the_model_is_not_resident() -> None:
    body = '{"models": [{"name": "other", "size": 1}]}'
    assert read_memory("http://h", "m", opener=FakeOpener(body)) is None


def test_ollama_memory_is_none_when_nothing_is_resident() -> None:
    assert read_memory("http://h", "m", opener=FakeOpener('{"models": []}')) is None


def test_ollama_memory_is_none_without_a_size() -> None:
    body = '{"models": [{"name": "m"}]}'
    assert read_memory("http://h", "m", opener=FakeOpener(body)) is None


def test_ollama_memory_is_none_for_a_non_object_response() -> None:
    assert read_memory("http://h", "m", opener=FakeOpener("[]")) is None


def test_ollama_memory_is_none_when_models_is_not_a_list() -> None:
    body = '{"models": "lots"}'
    assert read_memory("http://h", "m", opener=FakeOpener(body)) is None


def test_ollama_memory_skips_malformed_entries() -> None:
    body = '{"models": [5, {"name": "m", "size": 8}]}'
    reading = read_memory("http://h", "m", opener=FakeOpener(body))
    assert reading is not None and reading.total_bytes == 8


def test_ollama_memory_without_vram_reports_only_a_total() -> None:
    body = '{"models": [{"name": "m", "size": 8}]}'
    reading = read_memory("http://h", "m", opener=FakeOpener(body))
    assert reading is not None and reading.vram_bytes is None


# --- openai-compatible ------------------------------------------------


def test_openai_endpoint_is_the_chat_completions_route() -> None:
    runtime = OpenAICompatRuntime("http://h/v1/")
    assert runtime.endpoint == "http://h/v1/chat/completions"


def test_openai_sends_no_authorization_without_a_key() -> None:
    opener = FakeOpener(sse("[DONE]"))
    list(OpenAICompatRuntime("http://h/v1", opener=opener).stream("m", PROMPT))
    assert opener.requests[0].get_header("Authorization") is None


def test_openai_sends_a_bearer_token_when_given_one() -> None:
    opener = FakeOpener(sse("[DONE]"))
    runtime = OpenAICompatRuntime("http://h/v1", api_key="secret", opener=opener)
    list(runtime.stream("m", PROMPT))
    assert opener.requests[0].get_header("Authorization") == "Bearer secret"


def test_openai_probe_accepts_a_model_list() -> None:
    OpenAICompatRuntime("http://h/v1", opener=FakeOpener('{"data": []}')).probe()


def test_openai_probe_rejects_an_unrecognisable_server() -> None:
    opener = FakeOpener('{"models": []}')
    with pytest.raises(RuntimeUnavailable, match="OpenAI-compatible"):
        OpenAICompatRuntime("http://h/v1", opener=opener).probe()


def test_openai_asks_for_usage_and_caps_the_output() -> None:
    opener = FakeOpener(sse("[DONE]"))
    list(OpenAICompatRuntime("http://h/v1", opener=opener).stream("m", PROMPT))
    body = opener.payload
    assert body["stream_options"] == {"include_usage": True}
    assert body["max_tokens"] == 32
    assert body["temperature"] == 0
    assert body["messages"] == [{"role": "user", "content": "say something"}]


def test_openai_streams_deltas() -> None:
    body = sse(
        '{"choices": [{"delta": {"content": "He"}}]}',
        '{"choices": [{"delta": {"content": "llo"}, "finish_reason": null}]}',
        "[DONE]",
    )
    runtime = OpenAICompatRuntime("http://h/v1", opener=FakeOpener(body))
    chunks = list(runtime.stream("m", PROMPT))
    assert [chunk.text for chunk in chunks] == ["He", "llo", ""]
    assert chunks[-1].done is True


def test_openai_uses_usage_when_the_server_sends_it() -> None:
    body = sse(
        '{"choices": [{"delta": {"content": "hi"}}]}',
        '{"choices": [], "usage": {"completion_tokens": 12, "prompt_tokens": 4}}',
        "[DONE]",
    )
    runtime = OpenAICompatRuntime("http://h/v1", opener=FakeOpener(body))
    chunks = list(runtime.stream("m", PROMPT))
    assert chunks[1].output_tokens == 12
    assert chunks[1].prompt_tokens == 4


def test_openai_reports_no_count_when_usage_is_absent() -> None:
    body = sse('{"choices": [{"delta": {"content": "hi"}}]}', "[DONE]")
    runtime = OpenAICompatRuntime("http://h/v1", opener=FakeOpener(body))
    assert all(chunk.output_tokens is None for chunk in runtime.stream("m", PROMPT))


def test_openai_ignores_keepalive_comments() -> None:
    body = ": keep-alive\ndata: [DONE]\n"
    runtime = OpenAICompatRuntime("http://h/v1", opener=FakeOpener(body))
    assert len(list(runtime.stream("m", PROMPT))) == 1


def test_openai_ignores_non_data_sse_fields() -> None:
    body = "event: message\nid: 7\ndata: [DONE]\n"
    runtime = OpenAICompatRuntime("http://h/v1", opener=FakeOpener(body))
    assert len(list(runtime.stream("m", PROMPT))) == 1


def test_openai_marks_a_finish_reason_as_done() -> None:
    body = sse('{"choices": [{"delta": {}, "finish_reason": "stop"}]}')
    runtime = OpenAICompatRuntime("http://h/v1", opener=FakeOpener(body))
    assert list(runtime.stream("m", PROMPT))[0].done is True


def test_openai_tolerates_an_empty_choices_list() -> None:
    body = sse('{"choices": []}')
    runtime = OpenAICompatRuntime("http://h/v1", opener=FakeOpener(body))
    assert list(runtime.stream("m", PROMPT))[0].text == ""


def test_openai_tolerates_a_choice_that_is_not_an_object() -> None:
    body = sse('{"choices": [42]}')
    runtime = OpenAICompatRuntime("http://h/v1", opener=FakeOpener(body))
    assert list(runtime.stream("m", PROMPT))[0].text == ""


def test_openai_tolerates_a_non_string_delta_content() -> None:
    body = sse('{"choices": [{"delta": {"content": null}}]}')
    runtime = OpenAICompatRuntime("http://h/v1", opener=FakeOpener(body))
    assert list(runtime.stream("m", PROMPT))[0].text == ""


def test_openai_tolerates_a_missing_delta() -> None:
    body = sse('{"choices": [{"index": 0}]}')
    runtime = OpenAICompatRuntime("http://h/v1", opener=FakeOpener(body))
    assert list(runtime.stream("m", PROMPT))[0].text == ""


def test_openai_ignores_usage_that_is_not_an_object() -> None:
    body = sse('{"choices": [], "usage": 9}')
    runtime = OpenAICompatRuntime("http://h/v1", opener=FakeOpener(body))
    assert list(runtime.stream("m", PROMPT))[0].output_tokens is None


def test_openai_raises_on_a_structured_error() -> None:
    body = sse('{"error": {"message": "model not found", "type": "invalid"}}')
    runtime = OpenAICompatRuntime("http://h/v1", opener=FakeOpener(body))
    with pytest.raises(RuntimeUnavailable, match="model not found"):
        list(runtime.stream("m", PROMPT))


def test_openai_raises_on_a_bare_string_error() -> None:
    body = sse('{"error": "overloaded"}')
    runtime = OpenAICompatRuntime("http://h/v1", opener=FakeOpener(body))
    with pytest.raises(RuntimeUnavailable, match="overloaded"):
        list(runtime.stream("m", PROMPT))


def test_openai_rejects_a_non_json_event() -> None:
    runtime = OpenAICompatRuntime("http://h/v1", opener=FakeOpener("data: {oops\n"))
    with pytest.raises(ProtocolError, match="not JSON"):
        list(runtime.stream("m", PROMPT))


def test_openai_rejects_an_event_that_is_not_an_object() -> None:
    runtime = OpenAICompatRuntime("http://h/v1", opener=FakeOpener("data: [1]\n"))
    with pytest.raises(ProtocolError, match="expected a JSON object"):
        list(runtime.stream("m", PROMPT))


def test_openai_cannot_report_memory() -> None:
    # The API simply does not expose it; None becomes "n/a", not zero.
    assert OpenAICompatRuntime("http://h/v1").memory("m") is None


# --- replay -----------------------------------------------------------


def _recording(*streams: RecordedStream) -> Recording:
    return Recording(
        runtime="ollama",
        model="m",
        endpoint="http://h/api/generate",
        streams=streams,
    )


def _stream(prompt: str, repetition: int = 0, tokens: int | None = 5) -> RecordedStream:
    return RecordedStream(
        prompt=prompt,
        repetition=repetition,
        chunks=(RecordedChunk(0.1, "a"), RecordedChunk(0.2, "b")),
        end_s=0.5,
        reported_output_tokens=tokens,
    )


def test_virtual_clock_starts_where_it_is_told() -> None:
    assert VirtualClock(3.0)() == 3.0


def test_virtual_clock_moves_forward_on_request() -> None:
    clock = VirtualClock()
    clock.advance_to(2.5)
    assert clock() == 2.5


def test_virtual_clock_refuses_to_go_backwards() -> None:
    clock = VirtualClock(5.0)
    with pytest.raises(RecordingError, match="backwards"):
        clock.advance_to(4.0)


def test_replay_reports_the_recorded_endpoint() -> None:
    assert ReplayRuntime(_recording(_stream("p"))).endpoint == "http://h/api/generate"


def test_replay_remembers_which_runtime_produced_the_recording() -> None:
    assert ReplayRuntime(_recording(_stream("p"))).source_runtime == "ollama"


def test_replay_yields_recorded_text_then_a_final_count() -> None:
    runtime = ReplayRuntime(_recording(_stream("p")))
    chunks = list(runtime.stream("m", PromptSpec("p", "x")))
    assert [chunk.text for chunk in chunks] == ["a", "b", ""]
    assert chunks[-1].output_tokens == 5


def test_replay_advances_its_clock_to_the_recorded_offsets() -> None:
    runtime = ReplayRuntime(_recording(_stream("p")))
    observed = []
    for _ in runtime.stream("m", PromptSpec("p", "x")):
        observed.append(runtime.clock())
    assert observed == [0.1, 0.2, 0.5]


def test_replay_consumes_repetitions_in_order() -> None:
    recording = _recording(_stream("p", 0), _stream("p", 1))
    runtime = ReplayRuntime(recording)
    assert runtime.remaining("p") == 2
    list(runtime.stream("m", PromptSpec("p", "x")))
    assert runtime.remaining("p") == 1


def test_replay_clock_continues_across_repetitions() -> None:
    runtime = ReplayRuntime(_recording(_stream("p", 0), _stream("p", 1)))
    list(runtime.stream("m", PromptSpec("p", "x")))
    list(runtime.stream("m", PromptSpec("p", "x")))
    assert runtime.clock() == pytest.approx(1.0)


def test_replay_refuses_to_invent_a_run_it_does_not_have() -> None:
    runtime = ReplayRuntime(_recording(_stream("p")))
    list(runtime.stream("m", PromptSpec("p", "x")))
    with pytest.raises(RecordingError, match="no unplayed run"):
        list(runtime.stream("m", PromptSpec("p", "x")))


def test_replay_names_the_prompts_it_does_have() -> None:
    runtime = ReplayRuntime(_recording(_stream("p")))
    with pytest.raises(RecordingError, match="it contains: p"):
        list(runtime.stream("m", PromptSpec("other", "x")))


def test_replay_returns_the_recorded_memory() -> None:
    recording = _recording(_stream("p"))
    assert ReplayRuntime(recording).memory("m") is None


def test_replay_needs_no_probe() -> None:
    ReplayRuntime(_recording(_stream("p"))).probe()


def test_openai_ignores_a_non_integer_token_count() -> None:
    body = sse('{"choices": [], "usage": {"completion_tokens": "12"}}')
    runtime = OpenAICompatRuntime("http://h/v1", opener=FakeOpener(body))
    assert list(runtime.stream("m", PROMPT))[0].output_tokens is None


def test_a_runtime_times_itself_with_the_real_clock_by_default() -> None:
    # Only the replay runtime substitutes a clock; everything else must
    # be measured against wall time, or the numbers mean nothing.
    import time

    class Bare(Runtime):
        @property
        def endpoint(self) -> str:
            return "http://bare"

        def stream(self, model: str, prompt: PromptSpec) -> Iterator[Chunk]:
            yield Chunk(text="x")

    runtime = Bare()
    assert runtime.clock is time.perf_counter
    assert runtime.memory("m") is None
    assert runtime.probe() is None
