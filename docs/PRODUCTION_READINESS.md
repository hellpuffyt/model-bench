# Production readiness — model-bench

Audit date: 2026-10-22. Tested commit: see `git log` (follow-up after `d1a3b07`) (branch `audit/production-readiness`, based on `main` @ `d6561a9`).

## Scope

Stdlib-only CLI that benchmarks local LLM inference over HTTP (Ollama native API, OpenAI-compatible
APIs): time to first token, decode tokens/s, runtime-reported memory, at a fixed prompt suite; records
chunk timings and replays them through the same harness.

## What was and was not run

- **No real benchmark was run in this audit.** No local inference runtime (Ollama, llama.cpp server)
  is installed on the audit machine (`which ollama llama-server` empty; nothing on port 11434), and no
  model was downloaded (audit rule). No new numbers are reported anywhere.
- **Harness logic verified with a fake backend**, clearly labelled: the suite's `ScriptedRuntime`
  (scripted clock, no model) — 351 existing tests plus a new hand-computed row
  (`tests/test_harness.py::test_a_full_row_matches_a_hand_computation_on_a_fake_backend`: ttft 0.1 s,
  total 0.51 s, tok/s = 2/0.41).
- **Committed recording re-checked:** `model-bench replay fixtures/ollama-llama32-3b.json --format csv`
  equals the CI's expected CSV exactly; the table's tiny/short rows equal the README. Independent NumPy
  check of the summary statistics from the per-repetition CSV: tiny tok/s mean 179.3 sd 71.1, short
  176.3 sd 3.1; ttft p50/p90 0.023/0.025 and 0.021/0.041 — all match. Hand check: tiny rep 0 tok/s =
  (3−1)/(0.036630−0.022835) = 144.98.
- **Not benchmarked in this audit (exact list):** any live run of any runtime/model. **Not backed by a recording in the repo:** the README's `long` row, the `qwen2.5-coder:7b` row and the 1.663 s cold-start
  figure (no recording in the repo, other hardware). README now states this provenance.

## Clean-clone verification (commit `d1a3b07`)

```bash
git clone -b audit/production-readiness <local clone> model-bench && cd model-bench
uv venv --python 3.12 .venv && . .venv/bin/activate && uv pip install .
model-bench replay fixtures/ollama-llama32-3b.json --format csv   # identical to CI expectation
model-bench run --model llama3.2:3b --endpoint http://127.0.0.1:1 # exit 1, 0 bytes on stdout (no estimate)
model-bench prompts                                               # prompt suite listed
uv pip install -e ".[dev]" && pytest && ruff check . && mypy      # 353 passed / clean / clean
```

## Commands and results

| Command | Baseline (`d6561a9`) | After (`d1a3b07`) |
| --- | --- | --- |
| `pytest` | exit 0 — 351 passed, 0 failed, 0 skipped | exit 0 — 354 passed, 0 failed, 0 skipped |
| `ruff check .` / `mypy` | exit 0 / exit 0 | exit 0 / exit 0 |
| `uvx pip-audit` | — | no known vulnerabilities |

CI status — default branch: success (`d6561a9`, 2026-08-31); audit branch: pending until pushed.

## Bugs found and fixed

| Severity | Bug | Regression test |
| --- | --- | --- |
| Low (crash) | `replay` of a non-UTF-8 file → `UnicodeDecodeError` traceback | `tests/test_recording.py::test_a_binary_file_is_a_recording_error_not_a_traceback` |
| Docs | README `pip install model-bench` (not on PyPI); unrecorded results not marked as such | README |

Follow-up (owner request): replay warm-up reporting. The committed recording stores `"warmup": 1`, and the
README's live output says `warmup 1 discarded run(s)`, but `replay` printed `warmup 0`. The existing test
(`test_a_replayed_report_has_no_warmup`) reasoned that "the warm-up was done when the recording was made,
not now". That is true, but the report describes the *samples*, and those were measured after a warm-up of 1.
Classified as a bug and fixed: the replay report carries the recorded count, and the text header says
`warmup    1 discarded run(s) when recorded`. JSON `warmup` is now 1. The CSV that CI checks has no
warmup column and is unchanged. The old test was replaced by
`test_a_replayed_report_states_the_recorded_warmup` and `test_a_replay_header_says_the_warmup_happened_when_recorded`.
Both fail on the old code; the justification is written in the test. The README now shows the full real
replay output and says the live table's `long` row has no recording.

## Security review

API key read only from a named environment variable; not written to recordings/reports (grep of
`recording.py`/`report.py`). Default endpoint is loopback. No pickle; JSON recordings validated against
a schema. History secret scan: no hits. `pip-audit`: clean.

## Acceptance checklist

| # | Item | Expected | Evidence | Result |
| --- | --- | --- | --- | --- |
| 1 | tok/s, ttft, percentiles, sd correct | match hand/NumPy | above | PASS |
| 2 | Unreachable runtime → no numbers | exit 1, empty stdout | clean clone | PASS |
| 3 | Replay reproduces recording | CSV identical | clean clone | PASS |
| 4 | Replay labelled as replay | `source replay of …` | CI step / run | PASS |
| 4b | Replay states the recorded warm-up | `1 … when recorded` | new tests | PASS |
| 5 | Live benchmark against a real runtime | table | no runtime installed | NOT VERIFIED |

## Classification proposal

**Partially Verified** — harness and statistics verified with a fake backend and the committed
recording; no live benchmark could be run.
