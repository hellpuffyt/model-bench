"""Recordings: what they must contain to be replayable, and what they reject."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_bench.errors import RecordingError
from model_bench.models import MemoryReading
from model_bench.recording import (
    SCHEMA,
    RecordedChunk,
    RecordedStream,
    Recording,
    load,
    loads,
)

MINIMAL = {
    "schema": SCHEMA,
    "runtime": "ollama",
    "model": "m",
    "endpoint": "http://h",
    "streams": [
        {
            "prompt": "p",
            "repetition": 0,
            "end_s": 1.0,
            "chunks": [{"at_s": 0.1, "text": "a"}, {"at_s": 0.5, "text": "b"}],
            "reported_output_tokens": 9,
        }
    ],
}


def dumped(**overrides: object) -> str:
    document = json.loads(json.dumps(MINIMAL))
    document.update(overrides)
    return json.dumps(document)


def test_loads_a_minimal_recording() -> None:
    recording = loads(dumped())
    assert recording.runtime == "ollama"
    assert recording.model == "m"
    assert recording.endpoint == "http://h"
    assert len(recording.streams) == 1


def test_chunks_keep_their_offsets_and_text() -> None:
    stream = loads(dumped()).streams[0]
    assert stream.chunks == (RecordedChunk(0.1, "a"), RecordedChunk(0.5, "b"))
    assert stream.end_s == 1.0
    assert stream.reported_output_tokens == 9


def test_missing_fields_default_rather_than_crash() -> None:
    document = json.loads(json.dumps(MINIMAL))
    del document["runtime"]
    del document["model"]
    del document["endpoint"]
    recording = loads(json.dumps(document))
    assert recording.runtime == "unknown"
    assert recording.model == "unknown"


def test_prompts_are_listed_in_first_appearance_order() -> None:
    document = json.loads(json.dumps(MINIMAL))
    document["streams"] = [
        {"prompt": "z", "end_s": 1.0, "chunks": []},
        {"prompt": "a", "end_s": 1.0, "chunks": []},
        {"prompt": "z", "end_s": 1.0, "chunks": []},
    ]
    assert loads(json.dumps(document)).prompts == ("z", "a")


def test_streams_for_selects_one_prompt() -> None:
    document = json.loads(json.dumps(MINIMAL))
    document["streams"] = [
        {"prompt": "z", "end_s": 1.0, "chunks": []},
        {"prompt": "a", "end_s": 1.0, "chunks": []},
    ]
    assert len(loads(json.dumps(document)).streams_for("z")) == 1


def test_a_recording_may_carry_a_memory_reading() -> None:
    recording = loads(
        dumped(memory={"total_bytes": 100, "vram_bytes": 60, "source": "/api/ps"})
    )
    assert recording.memory == MemoryReading(100, "/api/ps", 60)


def test_memory_may_be_absent() -> None:
    assert loads(dumped(memory=None)).memory is None


def test_memory_without_a_total_is_rejected() -> None:
    with pytest.raises(RecordingError, match="total_bytes is required"):
        loads(dumped(memory={"source": "x"}))


def test_memory_that_is_not_an_object_is_rejected() -> None:
    with pytest.raises(RecordingError, match="memory must be a JSON object"):
        loads(dumped(memory=[1]))


def test_a_non_integer_memory_total_is_rejected() -> None:
    with pytest.raises(RecordingError, match="must be an integer"):
        loads(dumped(memory={"total_bytes": "lots"}))


def test_an_unknown_schema_is_refused() -> None:
    with pytest.raises(RecordingError, match="unknown recording schema"):
        loads(dumped(schema="model-bench/recording/99"))


def test_a_missing_schema_is_refused() -> None:
    document = json.loads(json.dumps(MINIMAL))
    del document["schema"]
    with pytest.raises(RecordingError, match="unknown recording schema"):
        loads(json.dumps(document))


def test_invalid_json_is_refused() -> None:
    with pytest.raises(RecordingError, match="not valid JSON"):
        loads("{oops")


def test_a_top_level_array_is_refused() -> None:
    with pytest.raises(RecordingError, match="must be a JSON object"):
        loads("[1, 2]")


def test_streams_must_be_a_list() -> None:
    with pytest.raises(RecordingError, match="streams must be a list"):
        loads(dumped(streams={"prompt": "p"}))


def test_a_string_is_not_a_list_of_streams() -> None:
    with pytest.raises(RecordingError, match="streams must be a list"):
        loads(dumped(streams="p"))


def test_an_empty_recording_is_refused() -> None:
    # There is nothing to replay, so failing beats producing an empty table.
    with pytest.raises(RecordingError, match="no streams"):
        loads(dumped(streams=[]))


def test_a_stream_must_be_an_object() -> None:
    with pytest.raises(RecordingError, match=r"streams\[0\] must be a JSON object"):
        loads(dumped(streams=[7]))


def test_a_stream_must_name_its_prompt() -> None:
    with pytest.raises(RecordingError, match="missing 'prompt'"):
        loads(dumped(streams=[{"end_s": 1.0, "chunks": []}]))


def test_a_stream_must_have_an_end_time() -> None:
    with pytest.raises(RecordingError, match="missing 'end_s'"):
        loads(dumped(streams=[{"prompt": "p", "chunks": []}]))


def test_chunks_must_be_a_list() -> None:
    with pytest.raises(RecordingError, match="chunks must be a list"):
        loads(dumped(streams=[{"prompt": "p", "end_s": 1.0, "chunks": "abc"}]))


def test_a_chunk_needs_a_numeric_offset() -> None:
    stream = {"prompt": "p", "end_s": 1.0, "chunks": [{"at_s": "soon", "text": "a"}]}
    with pytest.raises(RecordingError, match="must be a number"):
        loads(dumped(streams=[stream]))


def test_a_boolean_is_not_a_numeric_offset() -> None:
    stream = {"prompt": "p", "end_s": 1.0, "chunks": [{"at_s": True, "text": "a"}]}
    with pytest.raises(RecordingError, match="must be a number"):
        loads(dumped(streams=[stream]))


def test_chunk_times_may_not_go_backwards() -> None:
    # Out-of-order offsets would produce a negative time to first token.
    stream = {
        "prompt": "p",
        "end_s": 1.0,
        "chunks": [{"at_s": 0.5, "text": "a"}, {"at_s": 0.1, "text": "b"}],
    }
    with pytest.raises(RecordingError, match="goes backwards"):
        loads(dumped(streams=[stream]))


def test_chunk_times_may_repeat() -> None:
    # Two chunks in the same clock tick is normal on a coarse timer.
    stream = {
        "prompt": "p",
        "end_s": 1.0,
        "chunks": [{"at_s": 0.5, "text": "a"}, {"at_s": 0.5, "text": "b"}],
    }
    assert len(loads(dumped(streams=[stream])).streams[0].chunks) == 2


def test_the_end_must_not_precede_the_last_chunk() -> None:
    stream = {"prompt": "p", "end_s": 0.2, "chunks": [{"at_s": 0.5, "text": "a"}]}
    with pytest.raises(RecordingError, match="is before its last chunk"):
        loads(dumped(streams=[stream]))


def test_a_missing_chunk_text_becomes_empty() -> None:
    stream = {"prompt": "p", "end_s": 1.0, "chunks": [{"at_s": 0.5}]}
    assert loads(dumped(streams=[stream])).streams[0].chunks[0].text == ""


def test_a_chunk_must_be_an_object() -> None:
    stream = {"prompt": "p", "end_s": 1.0, "chunks": ["a"]}
    with pytest.raises(RecordingError, match=r"chunks\[0\] must be a JSON object"):
        loads(dumped(streams=[stream]))


def test_a_missing_repetition_defaults_to_zero() -> None:
    stream = {"prompt": "p", "end_s": 1.0, "chunks": []}
    assert loads(dumped(streams=[stream])).streams[0].repetition == 0


def test_a_non_integer_repetition_is_rejected() -> None:
    stream = {"prompt": "p", "end_s": 1.0, "chunks": [], "repetition": "first"}
    with pytest.raises(RecordingError, match="must be an integer"):
        loads(dumped(streams=[stream]))


def test_round_trip_through_json_preserves_everything() -> None:
    original = Recording(
        runtime="ollama",
        model="m",
        endpoint="http://h",
        streams=(
            RecordedStream(
                prompt="p",
                repetition=2,
                chunks=(RecordedChunk(0.25, "hi"),),
                end_s=0.75,
                reported_output_tokens=4,
                reported_prompt_tokens=3,
            ),
        ),
        memory=MemoryReading(10, "/api/ps", 5),
        warmup=2,
    )
    assert loads(original.to_json()) == original


def test_to_json_ends_with_a_newline() -> None:
    assert loads(dumped()).to_json().endswith("}\n")


def test_save_and_load_a_file(tmp_path: Path) -> None:
    recording = loads(dumped())
    destination = tmp_path / "run.json"
    recording.save(destination)
    assert load(destination) == recording


def test_loading_a_missing_file_says_so(tmp_path: Path) -> None:
    with pytest.raises(RecordingError, match="cannot read recording"):
        load(tmp_path / "absent.json")


def test_a_bad_file_error_names_the_file(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(RecordingError, match="broken.json"):
        load(broken)


def test_the_committed_fixture_is_a_valid_recording(recording_path: Path) -> None:
    recording = load(recording_path)
    assert recording.runtime == "ollama"
    assert recording.model == "llama3.2:3b"
    assert recording.prompts == ("tiny", "short")
    assert len(recording.streams) == 6


def test_the_committed_fixture_has_reported_token_counts(recording_path: Path) -> None:
    # It came from Ollama, which counts tokens itself; nothing is guessed.
    recording = load(recording_path)
    assert all(s.reported_output_tokens is not None for s in recording.streams)
