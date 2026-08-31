"""The ``model-bench`` command line.

Three subcommands, and the split between the first two is a design
decision rather than tidiness:

``run``
    Talks to a live runtime and measures it. If the runtime is not
    there, this fails; it does not fall back to anything.

``replay``
    Re-derives a report from a recording made earlier. Everything it
    prints is labelled as a replay.

``prompts``
    Prints the fixed prompt suite, so a reader can see exactly what was
    asked of the model before believing any number.

An API key is taken from an environment variable named with
``--api-key-env``, never as an argument. Command lines end up in shell
history, in ``ps`` output and in CI logs; environment variables at least
do not end up in the first two by default.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from . import __version__
from .errors import ModelBenchError, RuntimeUnavailable, UsageError
from .harness import Observer, RunOutcome, run_benchmark
from .http import DEFAULT_TIMEOUT
from .models import BenchmarkReport, PromptSpec
from .prompts import BUILTIN_PROMPTS, SETS, load_prompts, prompt_set
from .recording import Recording, load
from .report import FORMATS, render
from .runtime import Runtime
from .runtimes import (
    DEFAULT_ENDPOINTS,
    LIVE_RUNTIMES,
    OllamaRuntime,
    OpenAICompatRuntime,
    ReplayRuntime,
)

PROGRAM = "model-bench"


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, kept separate so tests can inspect it."""
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=(
            "Benchmark local LLM inference: time to first token, tokens per "
            "second and the runtime's reported memory, at fixed prompts."
        ),
        epilog=(
            "No runtime reachable means no numbers: model-bench reports the "
            "failure rather than estimating anything."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"{PROGRAM} {__version__}"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("run", help="benchmark a live runtime")
    run.add_argument(
        "--runtime",
        choices=LIVE_RUNTIMES,
        default="ollama",
        help="backend to measure (default: ollama)",
    )
    run.add_argument(
        "--model", required=True, help="model name as the runtime knows it"
    )
    run.add_argument(
        "--endpoint",
        help="base URL of the runtime; sensible per-runtime default when omitted",
    )
    run.add_argument(
        "--prompt-set",
        default="all",
        choices=sorted(SETS),
        help="which built-in prompts to run (default: all)",
    )
    run.add_argument(
        "--prompts", help="JSON file of custom prompts, instead of the built-in set"
    )
    run.add_argument(
        "--repeat", type=int, default=3, help="measured runs per prompt (default: 3)"
    )
    run.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="discarded runs before measuring, to exclude model load (default: 1)",
    )
    run.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"per-request timeout in seconds (default: {DEFAULT_TIMEOUT:g})",
    )
    run.add_argument(
        "--seed", type=int, default=0, help="sampling seed, for repeatable work"
    )
    run.add_argument(
        "--api-key-env",
        help="environment variable holding the API key (never pass the key itself)",
    )
    run.add_argument("--record", help="write the raw timings to this JSON file")
    run.add_argument("-v", "--verbose", action="store_true", help="log progress")
    _add_output_arguments(run)

    replay = subcommands.add_parser(
        "replay", help="re-derive a report from a recording made earlier"
    )
    replay.add_argument("recording", help="a JSON file written by --record")
    _add_output_arguments(replay)

    prompts = subcommands.add_parser("prompts", help="show the built-in prompt suite")
    prompts.add_argument(
        "--set", dest="prompt_set", default="all", choices=sorted(SETS)
    )
    return parser


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-f",
        "--format",
        choices=FORMATS,
        default="table",
        help="output format (default: table)",
    )
    parser.add_argument("-o", "--output", help="write the report here instead of stdout")


def make_runtime(
    name: str,
    *,
    endpoint: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    seed: int = 0,
    api_key: str | None = None,
) -> Runtime:
    """Build a live runtime by name."""
    host = endpoint or DEFAULT_ENDPOINTS.get(name)
    if host is None:
        raise UsageError(
            f"unknown runtime {name!r}; choose one of: {', '.join(LIVE_RUNTIMES)}"
        )
    if name == "ollama":
        return OllamaRuntime(host, timeout=timeout, seed=seed)
    if name == "openai":
        return OpenAICompatRuntime(host, api_key=api_key, timeout=timeout)
    raise UsageError(
        f"unknown runtime {name!r}; choose one of: {', '.join(LIVE_RUNTIMES)}"
    )


def resolve_prompts(
    prompt_file: str | None, set_name: str
) -> tuple[PromptSpec, ...]:
    """Custom prompts if a file was given, otherwise a built-in set."""
    if prompt_file:
        return load_prompts(prompt_file)
    return prompt_set(set_name)


def read_api_key(
    variable: str | None, environ: dict[str, str] | None = None
) -> str | None:
    """The API key from the environment, if one was asked for."""
    if not variable:
        return None
    source = os.environ if environ is None else environ
    value = source.get(variable)
    if not value:
        raise UsageError(
            f"environment variable {variable!r} is empty or unset, so there is "
            "no API key to send"
        )
    return value


def replay_report(recording: Recording, source: str) -> BenchmarkReport:
    """Rebuild a report by replaying every recorded stream.

    Every prompt must have the same number of recorded repetitions,
    because the report's per-prompt statistics are only comparable when
    they are computed over samples of the same size.
    """
    runtime = ReplayRuntime(recording)
    names = recording.prompts
    counts = {name: len(recording.streams_for(name)) for name in names}
    unique = set(counts.values())
    if len(unique) > 1:
        detail = ", ".join(f"{name}={counts[name]}" for name in names)
        raise UsageError(
            f"recording has an uneven number of runs per prompt ({detail}); "
            "model-bench cannot summarise it"
        )
    specs = [PromptSpec(name=name, text=f"(recorded prompt {name!r})") for name in names]
    outcome = run_benchmark(
        runtime,
        recording.model,
        specs,
        repeat=unique.pop(),
        warmup=0,
        replayed_from=source,
        runtime_label=recording.runtime,
    )
    return outcome.report


def _emit(text: str, destination: str | None, stdout: TextIO) -> None:
    if destination is None:
        stdout.write(text)
        return
    Path(destination).write_text(text, encoding="utf-8")


def _run_command(arguments: argparse.Namespace, stderr: TextIO) -> RunOutcome:
    prompts = resolve_prompts(arguments.prompts, arguments.prompt_set)
    api_key = read_api_key(arguments.api_key_env)
    runtime = make_runtime(
        arguments.runtime,
        endpoint=arguments.endpoint,
        timeout=arguments.timeout,
        seed=arguments.seed,
        api_key=api_key,
    )
    def log(message: str) -> None:
        stderr.write(f"{PROGRAM}: {message}\n")

    observer: Observer | None = log if arguments.verbose else None
    return run_benchmark(
        runtime,
        arguments.model,
        prompts,
        repeat=arguments.repeat,
        warmup=arguments.warmup,
        observer=observer,
    )


def _prompts_command(arguments: argparse.Namespace, stdout: TextIO) -> None:
    selected = {spec.name for spec in prompt_set(arguments.prompt_set)}
    for spec in BUILTIN_PROMPTS:
        if spec.name not in selected:
            continue
        stdout.write(f"{spec.name} (max {spec.max_tokens} tokens)\n")
        stdout.write(f"  {spec.text}\n")


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """The console script. Returns the process exit code."""
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    arguments = build_parser().parse_args(argv)

    try:
        if arguments.command == "prompts":
            _prompts_command(arguments, out)
            return 0
        if arguments.command == "replay":
            report = replay_report(load(arguments.recording), arguments.recording)
            _emit(render(report, arguments.format), arguments.output, out)
            return 0
        outcome = _run_command(arguments, err)
        if arguments.record:
            outcome.recording.save(arguments.record)
        _emit(render(outcome.report, arguments.format), arguments.output, out)
        return 0
    except RuntimeUnavailable as error:
        err.write(f"{PROGRAM}: {error}\n")
        err.write(
            f"{PROGRAM}: no measurement was taken; nothing was estimated in its "
            "place\n"
        )
        return 1
    except ModelBenchError as error:
        err.write(f"{PROGRAM}: {error}\n")
        return 1
    except OSError as error:
        err.write(f"{PROGRAM}: {error}\n")
        return 1
