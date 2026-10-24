# model-bench

Benchmark local LLM inference across runtimes — time to first token,
tokens per second, and the memory the runtime says the model occupies —
at fixed prompts, with no runtime dependencies.

Not published on PyPI; install from a clone (Python 3.10+, no runtime
dependencies):

```bash
git clone https://github.com/hellpuffyt/model-bench && cd model-bench
python -m venv .venv && . .venv/bin/activate
pip install .
```

```bash
model-bench run --runtime ollama --model llama3.2:3b --repeat 3
```

```
model-bench 0.1.0
runtime   ollama
model     llama3.2:3b
endpoint  http://127.0.0.1:11434/api/generate
memory    2.38 GiB resident, 2.38 GiB in VRAM (http://127.0.0.1:11434/api/ps)
warmup    1 discarded run(s)
tokens    reported by the runtime

prompt  runs  ttft p50  ttft p90  tok/s mean  tok/s sd  total p50  tokens
------  ----  --------  --------  ----------  --------  ---------  ------
tiny       3     0.023     0.025       179.3      71.1      0.033       3
short      3     0.021     0.041       176.3       3.1      0.750     128
long       3     0.029     0.061       173.2       0.5      2.989     512
```

## Why this exists

"How fast is this model on my machine" is usually answered with a
stopwatch and a screenshot, and the answer is usually not reproducible.
Three things go wrong, and model-bench is built around all three.

**The denominator is a guess.** Tokens per second is tokens ÷ seconds.
Seconds are easy. Tokens are not: an HTTP client sees chunks of text, and
chunks are not tokens. So every sample records where its count came from,
and the report says so on its own line — `tokens reported by the runtime`
when the server counted them (Ollama's `eval_count`, OpenAI's `usage`),
`tokens chunks` when they had to be inferred. A number derived from a
guess is never printed as though it were not.

**The first run is not a run.** The first generation after a model loads
is mostly disk. On the machine below, the `tiny` prompt's time to first
token was **1.663 s** with `--warmup 0` on a cold model and **0.023 s**
with the default `--warmup 1` — a factor of 72, none of it inference.
model-bench discards `--warmup` generations before measuring and prints
how many it discarded, so the figure is neither polluted by the load nor
quietly cleaned up.

**Nothing is estimated when nothing is measured.** If the runtime is not
reachable, there is no table:

```
$ model-bench run --model llama3.2:3b --endpoint http://127.0.0.1:1
model-bench: runtime 'ollama' at http://127.0.0.1:1/api/tags is unavailable: [Errno 111] Connection refused
model-bench: no measurement was taken; nothing was estimated in its place
```

Exit code 1, nothing on stdout. This is asserted in CI.

## What it measures

| Column | Meaning |
| --- | --- |
| `ttft p50` / `ttft p90` | seconds from request to the first chunk carrying text |
| `tok/s mean` | decode throughput: `(tokens − 1) ÷ (total − ttft)` |
| `tok/s sd` | sample standard deviation across repetitions |
| `total p50` | seconds to the end of the stream |
| `tokens` | median output tokens, counted as the `tokens` header says |

The first token is excluded from the throughput denominator because it
was produced during the prefill the `ttft` column already reports;
counting it in both would flatter every runtime by the same wrong amount.
A prompt that yielded fewer than two tokens, or finished inside the
clock's resolution, prints `n/a` rather than a number derived from timer
noise.

Memory is what the *runtime* reports about the model it has resident —
Ollama's `/api/ps` — not the RSS of the benchmark process, which would
measure Python. Runtimes that cannot report it print
`n/a (this runtime does not report it)`.

## Measured results

An RTX 4070 SUPER, i9-14900K, Ollama 0.33.2, `--repeat 3 --warmup 1`, the
built-in prompt suite. These are the tool's own output, not estimates.

> **Provenance.** The `tiny` and `short` rows for `llama3.2:3b` are backed by
> the committed recording: `model-bench replay fixtures/ollama-llama32-3b.json`
> reproduces them exactly (re-checked in the latest audit, see
> [docs/PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md)). The `long`
> row, the `qwen2.5-coder:7b` row and the cold-start 1.663 s figure have no
> recording in this repository and were not re-measured: the audit machine
> had no local inference runtime installed.

| Model | ttft p50 (short) | tok/s (short) | tok/s (long) | resident |
| --- | --- | --- | --- | --- |
| `llama3.2:3b` | 0.021 s | 176.3 ± 3.1 | 173.2 ± 0.5 | 2.38 GiB |
| `qwen2.5-coder:7b` | 0.027 s | 90.8 ± 0.7 | 90.5 ± 0.0 | 4.42 GiB |

Two things worth reading off that table. Throughput on the 7B model is
almost exactly half the 3B's, and its resident size is almost exactly
double — on a GPU with room for both, decode is memory-bandwidth bound
and the ratio falls out of the weights. And the standard deviation on the
`long` prompt is under 0.6% of the mean on both models, which is what
makes a 3-run benchmark worth quoting at all; the `tiny` prompt's
deviation is enormous by comparison (71 on a mean of 179), which is why
the suite has a `long` prompt.

## Recordings, and why replay exists

`--record` writes the timestamp of every chunk of every repetition to
JSON. `model-bench replay` re-derives the whole report from it:

```bash
model-bench run --model llama3.2:3b --record run.json
model-bench replay run.json
```

Replay is not a mock. It implements the same runtime interface the live
backends do and drives the same timing loop, statistics and formatter —
the only difference is that its clock is one it advances to each recorded
offset instead of `time.perf_counter`. So a replay is instantaneous,
identical on every platform, and reproduces the original to the last
decimal. Replaying [the committed recording](fixtures/ollama-llama32-3b.json)
of the run above prints the same `tiny` and `short` rows the live run
did, byte for byte — [CI asserts exactly that](.github/workflows/ci.yml)
on every push. The recording holds only those two prompts, so the live
table's `long` row has no recording behind it. This is the full replay output:

```
$ model-bench replay fixtures/ollama-llama32-3b.json
model-bench 0.1.0
source    replay of fixtures/ollama-llama32-3b.json (not a live run)
runtime   ollama
model     llama3.2:3b
endpoint  http://127.0.0.1:11434/api/generate
memory    2.38 GiB resident, 2.38 GiB in VRAM (http://127.0.0.1:11434/api/ps)
warmup    1 discarded run(s) when recorded
tokens    reported by the runtime

prompt  runs  ttft p50  ttft p90  tok/s mean  tok/s sd  total p50  tokens
------  ----  --------  --------  ----------  --------  ---------  ------
tiny       3     0.023     0.025       179.3      71.1      0.033       3
short      3     0.021     0.041       176.3       3.1      0.750     128
```

The `warmup` line describes the recorded run (the recording file stores
`"warmup": 1`): its samples were measured after one discarded generation.
The replay itself discards nothing, because there is no model to warm.

What replay is *for*: attaching evidence to a claim. A pull request that
says "this is 8% faster" can carry the recording, and a reviewer can
check the table instead of trusting it. Every replayed report is stamped:

```
source    replay of run.json (not a live run)
```

so a recording can never be passed off as a measurement of the machine
printing it. Replay also has its own subcommand rather than a
`--runtime replay` flag, for the same reason.

## The prompt suite

Fixed and committed, because a benchmark is only a comparison if the work
is identical. `model-bench prompts` prints exactly what will be sent.

| Prompt | Cap | What it exercises |
| --- | --- | --- |
| `tiny` | 16 tokens | almost pure time to first token |
| `short` | 128 tokens | the everyday case |
| `long` | 512 tokens | sustained decode, KV-cache growth |

Each prompt caps its own output. Without a cap a chattier model would
measure as slower for being chattier, and the benchmark would be about
verbosity. `--prompt-set quick` drops the long one; `--prompts FILE`
takes a JSON list of `{name, text, max_tokens}` instead.

## Runtimes

| `--runtime` | Speaks | Tokens | Memory |
| --- | --- | --- | --- |
| `ollama` | `POST /api/generate`, NDJSON | reported (`eval_count`) | `/api/ps` |
| `openai` | `POST /v1/chat/completions`, SSE | reported when the server sends `usage` | not exposed |

The `openai` adapter covers llama.cpp's server, vLLM, LM Studio, TGI and
anything else OpenAI-compatible. It requests `stream_options.include_usage`
and falls back to chunk counting — labelled as such — when the server
omits it, which several do.

Generation is pinned to `temperature: 0` with a fixed `--seed`. Sampling
temperature changes how many tokens come out, which would move the very
denominator being measured.

An API key is read from an environment variable named by `--api-key-env`,
never passed as an argument, because command lines end up in shell
history and CI logs:

```bash
model-bench run --runtime openai --endpoint http://localhost:8080/v1 \
  --model local --api-key-env LLAMA_API_KEY
```

## Output formats

`--format table` (default), `markdown` for pasting into a PR, `json` for
the full report including every individual sample, and `csv` — which is
one row per *sample*, not per prompt, because a reader who reaches for
CSV wants to do their own statistics. `-o FILE` writes instead of
printing.

## As a library

```python
from model_bench.harness import run_benchmark
from model_bench.prompts import prompt_set
from model_bench.report import render
from model_bench.runtimes import OllamaRuntime

outcome = run_benchmark(
    OllamaRuntime("http://127.0.0.1:11434"),
    model="llama3.2:3b",
    prompts=prompt_set("quick"),
    repeat=3,
    warmup=1,
)
print(render(outcome.report, "markdown"))
outcome.recording.save("run.json")
```

Anything implementing `model_bench.runtime.Runtime` — three methods —
can be benchmarked by the same harness. That is what makes the statistics
core testable without a model: the whole test suite runs with no server
and no `sleep`, against recorded streams and a clock the test controls.

## Development

```bash
pip install -e ".[dev]"
pytest -q                 # tests
ruff check .              # lint
mypy                      # strict types
pytest -q --cov=model_bench --cov-report=term-missing
```

Zero runtime dependencies: `urllib.request` is the whole networking
layer. CI runs the suite on Linux, macOS and Windows across Python
3.10–3.13.

## License

MIT — see [LICENSE](LICENSE).
