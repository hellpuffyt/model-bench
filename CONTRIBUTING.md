# Contributing

Thanks for taking a look.

## Getting set up

```bash
git clone https://github.com/hellpuffyt/model-bench
cd model-bench
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

You do not need a model server to develop on this. The whole suite runs
against recorded streams and a clock the tests control.

## Before opening a pull request

All four must pass; CI runs the same commands.

```bash
pytest -q
ruff check .
mypy
pytest -q --cov=model_bench --cov-report=term-missing
```

## Design constraints

These are the rules the project is built around. A change that breaks one
of them is not a trade-off to discuss — it is a different tool.

- **No number is printed that was not measured.** If a runtime cannot be
  reached, the run fails and prints nothing on stdout. If a token count
  was inferred from chunk boundaries rather than reported by the server,
  the report says `tokens chunks`. If throughput cannot be computed
  honestly — fewer than two tokens, or a decode window inside the
  clock's resolution — the cell is `n/a`, never `0`. If memory is not
  exposed by the runtime's API, it is `n/a`, never a guess. This is
  enforced by tests in `tests/test_harness.py` and `tests/test_report.py`,
  and by the `dogfood` CI job.
- **A recording is never presented as a live run.** Replay has its own
  subcommand, and every replayed report carries a `source ... (not a live
  run)` line in every format. If you add a format, it prints that line.
- **Timing goes through `runtime.clock`, never `time.perf_counter`
  directly.** That indirection is the only reason replays are exact and
  the test suite has no `sleep` in it. Code that reads the wall clock
  itself cannot be replayed and cannot be tested deterministically.
- **No runtime dependencies.** `urllib.request` is the networking layer.
  A benchmarking tool should install anywhere without argument.

## Adding a runtime

1. Subclass `Runtime` in a new module under `src/model_bench/runtimes/`.
   Three things are required: `endpoint`, `stream()`, and `probe()` if
   reachability can be checked before measuring — which it almost always
   can, and should be, so a run fails in milliseconds rather than after
   the first prompt.
2. `stream()` yields `Chunk`. Set `output_tokens` on the final chunk
   **only** if the protocol actually reports a count. Leaving it `None`
   is not a failure; it makes the report say `chunks`, which is true.
   Setting it to a number you derived is the one thing that is not
   allowed.
3. Implement `memory()` only if the runtime exposes what the loaded model
   occupies. The default returns `None`, which prints `n/a`.
4. Map every transport failure to `RuntimeUnavailable` and every
   unparseable response to `ProtocolError`. Going through
   `model_bench.http` does this for you.
5. Register it in `runtimes/__init__.py` (`LIVE_RUNTIMES`,
   `DEFAULT_ENDPOINTS`) and in `make_runtime` in `cli.py`.
6. Test it against `FakeOpener` from `tests/conftest.py`, including a
   malformed line, a mid-stream error object, and a response with no
   usage block. Those three are what real servers actually do.

## Adding a statistic

Put it in `stats.py`, which is pure and takes floats. Then surface it in
`models.py` as a property that returns `None` when it cannot be computed,
and in `report.py` as a column that prints `n/a` for that `None`. The
`None` is not defensive programming; it is the whole point.

## Reporting a bug

Attach a recording. `model-bench run --record run.json` writes every
chunk timestamp of every repetition, and `model-bench replay run.json`
reproduces the report exactly on any machine — which means a bug in the
statistics, the aggregation or any output format can be reproduced from
the file alone, without your hardware or your model. Include the exact
command line; `--repeat` and `--warmup` change the numbers by design.

If the bug is in a runtime adapter rather than the numbers, the raw
protocol lines are what is needed instead: a few NDJSON or SSE lines the
server sent that model-bench mishandled.
