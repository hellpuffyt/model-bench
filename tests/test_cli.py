"""The console script, end to end, including its failure output."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from conftest import ScriptedRuntime
from model_bench.cli import (
    build_parser,
    main,
    make_runtime,
    read_api_key,
    replay_report,
    resolve_prompts,
)
from model_bench.errors import RuntimeUnavailable, UsageError
from model_bench.recording import RecordedChunk, RecordedStream, Recording, load
from model_bench.runtimes import OllamaRuntime, OpenAICompatRuntime


def run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


# --- argument parsing -------------------------------------------------


def test_a_subcommand_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_run_requires_a_model() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run"])


def test_run_defaults_to_ollama_with_three_repetitions() -> None:
    arguments = build_parser().parse_args(["run", "--model", "m"])
    assert arguments.runtime == "ollama"
    assert arguments.repeat == 3
    assert arguments.warmup == 1
    assert arguments.format == "table"


def test_an_unknown_runtime_is_rejected_by_the_parser() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--model", "m", "--runtime", "vllm-ish"])


def test_an_unknown_format_is_rejected_by_the_parser() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["replay", "f.json", "--format", "yaml"])


def test_version_prints_and_exits() -> None:
    with pytest.raises(SystemExit) as caught:
        build_parser().parse_args(["--version"])
    assert caught.value.code == 0


def test_replay_takes_a_recording_path() -> None:
    assert build_parser().parse_args(["replay", "r.json"]).recording == "r.json"


# --- runtime construction ---------------------------------------------


def test_make_runtime_builds_ollama_at_its_default_host() -> None:
    runtime = make_runtime("ollama")
    assert isinstance(runtime, OllamaRuntime)
    assert runtime.host == "http://127.0.0.1:11434"


def test_make_runtime_honours_an_explicit_endpoint() -> None:
    runtime = make_runtime("ollama", endpoint="http://box:9999")
    assert isinstance(runtime, OllamaRuntime)
    assert runtime.host == "http://box:9999"


def test_make_runtime_builds_the_openai_adapter_with_its_key() -> None:
    runtime = make_runtime("openai", api_key="k")
    assert isinstance(runtime, OpenAICompatRuntime)
    assert runtime.api_key == "k"


def test_make_runtime_passes_the_timeout_through() -> None:
    runtime = make_runtime("ollama", timeout=7.5)
    assert isinstance(runtime, OllamaRuntime)
    assert runtime.timeout == 7.5


def test_make_runtime_rejects_an_unknown_name() -> None:
    with pytest.raises(UsageError, match="unknown runtime 'llamafile'"):
        make_runtime("llamafile")


def test_make_runtime_rejects_an_unknown_name_even_with_an_endpoint() -> None:
    with pytest.raises(UsageError, match="unknown runtime"):
        make_runtime("llamafile", endpoint="http://h")


# --- api keys ---------------------------------------------------------


def test_no_key_is_read_when_none_was_asked_for() -> None:
    assert read_api_key(None) is None


def test_a_key_is_read_from_the_named_variable() -> None:
    assert read_api_key("TOKEN", {"TOKEN": "abc"}) == "abc"


def test_an_unset_variable_is_an_error_not_a_silent_anonymous_call() -> None:
    with pytest.raises(UsageError, match="is empty or unset"):
        read_api_key("TOKEN", {})


def test_an_empty_variable_is_an_error() -> None:
    with pytest.raises(UsageError, match="is empty or unset"):
        read_api_key("TOKEN", {"TOKEN": ""})


def test_the_real_environment_is_used_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_BENCH_TEST_KEY", "from-env")
    assert read_api_key("MODEL_BENCH_TEST_KEY") == "from-env"


# --- prompt resolution ------------------------------------------------


def test_prompts_come_from_the_named_set_by_default() -> None:
    assert [spec.name for spec in resolve_prompts(None, "quick")] == ["tiny", "short"]


def test_a_prompt_file_overrides_the_set(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(json.dumps([{"name": "custom", "text": "hi"}]), encoding="utf-8")
    assert [spec.name for spec in resolve_prompts(str(path), "all")] == ["custom"]


# --- the prompts subcommand -------------------------------------------


def test_prompts_lists_the_whole_suite() -> None:
    code, out, _ = run("prompts")
    assert code == 0
    assert "tiny (max 16 tokens)" in out
    assert "long (max 512 tokens)" in out


def test_prompts_can_be_narrowed_to_a_set() -> None:
    code, out, _ = run("prompts", "--set", "tiny")
    assert code == 0
    assert "long (" not in out


def test_prompts_prints_the_text_that_will_be_sent() -> None:
    _, out, _ = run("prompts", "--set", "tiny")
    assert "Reply with exactly the word: pong" in out


# --- the replay subcommand --------------------------------------------


def test_replaying_the_committed_fixture_reproduces_its_numbers(
    recording_path: Path,
) -> None:
    code, out, err = run("replay", str(recording_path))
    assert code == 0, err
    assert "source    replay of" in out
    assert "model     llama3.2:3b" in out
    lines = [line for line in out.splitlines() if line.startswith(("tiny", "short"))]
    assert len(lines) == 2


def test_replay_is_deterministic(recording_path: Path) -> None:
    # Same file in, identical bytes out — which is what CI asserts.
    assert run("replay", str(recording_path)) == run("replay", str(recording_path))


def test_replay_reports_the_recorded_memory(recording_path: Path) -> None:
    _, out, _ = run("replay", str(recording_path))
    assert "GiB resident" in out
    assert "/api/ps" in out


def test_replay_can_emit_csv(recording_path: Path) -> None:
    code, out, _ = run("replay", str(recording_path), "--format", "csv")
    assert code == 0
    assert out.splitlines()[0].startswith("runtime,model,prompt")
    assert len(out.splitlines()) == 7


def test_replay_can_emit_json(recording_path: Path) -> None:
    code, out, _ = run("replay", str(recording_path), "--format", "json")
    payload = json.loads(out)
    assert code == 0
    assert payload["replayed_from"].endswith(".json")


def test_replay_writes_to_a_file_when_asked(
    recording_path: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "out.md"
    code, out, _ = run(
        "replay", str(recording_path), "--format", "markdown", "-o", str(destination)
    )
    assert code == 0
    assert out == ""
    assert destination.read_text(encoding="utf-8").startswith("# model-bench")


def test_replaying_a_missing_file_fails_cleanly(tmp_path: Path) -> None:
    code, out, err = run("replay", str(tmp_path / "nope.json"))
    assert code == 1
    assert out == ""
    assert "cannot read recording" in err


def test_replaying_a_broken_file_fails_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{", encoding="utf-8")
    code, _, err = run("replay", str(path))
    assert code == 1
    assert "not valid JSON" in err


def _uneven() -> Recording:
    def stream(prompt: str, repetition: int) -> RecordedStream:
        return RecordedStream(
            prompt=prompt,
            repetition=repetition,
            chunks=(RecordedChunk(0.1, "a"),),
            end_s=0.2,
        )

    return Recording(
        runtime="ollama",
        model="m",
        endpoint="http://h",
        streams=(stream("a", 0), stream("a", 1), stream("b", 0)),
    )


def test_an_uneven_recording_is_refused_rather_than_averaged() -> None:
    with pytest.raises(UsageError, match="uneven number of runs"):
        replay_report(_uneven(), "run.json")


def test_the_uneven_message_shows_the_counts() -> None:
    with pytest.raises(UsageError, match="a=2, b=1"):
        replay_report(_uneven(), "run.json")


def test_an_uneven_recording_fails_through_the_cli(tmp_path: Path) -> None:
    path = tmp_path / "uneven.json"
    _uneven().save(path)
    code, _, err = run("replay", str(path))
    assert code == 1
    assert "uneven number of runs" in err


def test_a_replayed_report_keeps_the_original_runtime_name(
    recording_path: Path,
) -> None:
    report = replay_report(load(recording_path), "fixture")
    assert report.runtime == "ollama"
    assert report.replayed_from == "fixture"


def test_a_replayed_report_has_no_warmup(recording_path: Path) -> None:
    # The warm-up was done when the recording was made, not now.
    assert replay_report(load(recording_path), "fixture").warmup == 0


# --- the run subcommand -----------------------------------------------


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, runtime: Any) -> None:
    monkeypatch.setattr("model_bench.cli.make_runtime", lambda *a, **k: runtime)


def test_a_run_prints_a_table(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_runtime(monkeypatch, ScriptedRuntime([[(0.1, "a"), (0.2, "b")]]))
    code, out, _ = run("run", "--model", "m", "--prompt-set", "tiny", "--repeat", "2")
    assert code == 0
    assert "model     m" in out
    assert "tiny" in out


def test_a_run_can_record_its_timings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_runtime(monkeypatch, ScriptedRuntime([[(0.1, "a")]], reported_tokens=4))
    destination = tmp_path / "run.json"
    code, _, _ = run(
        "run",
        "--model",
        "m",
        "--prompt-set",
        "tiny",
        "--repeat",
        "2",
        "--warmup",
        "0",
        "--record",
        str(destination),
    )
    assert code == 0
    recording = load(destination)
    assert len(recording.streams) == 2
    assert recording.model == "m"


def test_a_recorded_run_can_be_replayed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_runtime(
        monkeypatch, ScriptedRuntime([[(0.1, "a"), (0.5, "b")]], reported_tokens=4)
    )
    destination = tmp_path / "run.json"
    run(
        "run",
        "--model",
        "m",
        "--prompt-set",
        "tiny",
        "--repeat",
        "2",
        "--warmup",
        "0",
        "--record",
        str(destination),
        "--format",
        "csv",
        "-o",
        str(tmp_path / "live.csv"),
    )
    code, replayed, _ = run("replay", str(destination), "--format", "csv")
    assert code == 0
    live = (tmp_path / "live.csv").read_text(encoding="utf-8")
    assert replayed.split("\n")[1:] == live.split("\n")[1:]


def test_verbose_logs_progress_to_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_runtime(monkeypatch, ScriptedRuntime([[(0.1, "a")]]))
    code, out, err = run("run", "--model", "m", "--prompt-set", "tiny", "-v")
    assert code == 0
    assert "model-bench: tiny run 1/3" in err
    assert "model-bench" in out


def test_quiet_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_runtime(monkeypatch, ScriptedRuntime([[(0.1, "a")]]))
    _, _, err = run("run", "--model", "m", "--prompt-set", "tiny")
    assert err == ""


def test_an_unreachable_runtime_reports_that_nothing_was_estimated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = ScriptedRuntime(
        [[(0.1, "a")]],
        probe_error=RuntimeUnavailable("ollama", "http://127.0.0.1:11434", "refused"),
    )
    _patch_runtime(monkeypatch, runtime)
    code, out, err = run("run", "--model", "m")
    assert code == 1
    assert out == ""
    assert "is unavailable: refused" in err
    assert "nothing was estimated in its place" in err


def test_a_usage_error_is_reported_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_runtime(monkeypatch, ScriptedRuntime([[(0.1, "a")]]))
    code, _, err = run("run", "--model", "m", "--repeat", "0")
    assert code == 1
    assert "--repeat must be at least 1" in err


def test_a_missing_prompt_file_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_runtime(monkeypatch, ScriptedRuntime([[(0.1, "a")]]))
    code, _, err = run("run", "--model", "m", "--prompts", "no-such-file.json")
    assert code == 1
    assert "cannot read prompt file" in err


def test_an_unwritable_output_path_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_runtime(monkeypatch, ScriptedRuntime([[(0.1, "a")]]))
    unwritable = tmp_path / "missing-dir" / "out.txt"
    code, _, err = run(
        "run", "--model", "m", "--prompt-set", "tiny", "-o", str(unwritable)
    )
    assert code == 1
    assert "model-bench:" in err


def test_a_custom_prompt_file_is_used(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_runtime(monkeypatch, ScriptedRuntime([[(0.1, "a")]]))
    path = tmp_path / "p.json"
    path.write_text(json.dumps([{"name": "mine", "text": "hi"}]), encoding="utf-8")
    code, out, _ = run("run", "--model", "m", "--prompts", str(path), "--repeat", "1")
    assert code == 0
    assert "mine" in out


def test_an_api_key_variable_that_is_unset_stops_the_run() -> None:
    code, _, err = run("run", "--model", "m", "--api-key-env", "NOT_SET_ANYWHERE_XYZ")
    assert code == 1
    assert "is empty or unset" in err
