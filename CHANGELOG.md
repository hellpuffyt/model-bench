# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-08-31

First release.

### Added

- `model-bench run`: benchmark a live runtime at a fixed prompt suite,
  reporting time to first token (p50 and p90), decode throughput in
  tokens per second with its standard deviation, total wall time, and
  the memory the runtime says the loaded model occupies.
- Two live runtimes behind one interface: `ollama` (NDJSON over
  `/api/generate`, with token counts from `eval_count` and memory from
  `/api/ps`) and `openai` (server-sent events over
  `/v1/chat/completions`, covering llama.cpp, vLLM, LM Studio and TGI).
- Explicit token-count provenance. A count is labelled `reported` only
  when the runtime's protocol supplied one; otherwise it is `chunks` and
  the report says so.
- Honest failure: an unreachable runtime exits non-zero with nothing on
  stdout and an explicit "nothing was estimated in its place", rather
  than falling back to anything.
- Warm-up runs, discarded before measuring and counted in the report, so
  the first post-load generation does not turn the benchmark into a
  measurement of disk speed.
- `--record` and `model-bench replay`: every chunk timestamp written to
  JSON and replayed through the real harness against a virtual clock,
  reproducing the original report exactly on any machine. Replayed
  reports are stamped `(not a live run)` in every format.
- Four output formats — `table`, `markdown`, `json` (including every
  individual sample) and `csv` (one row per sample) — plus `-o FILE`.
- A committed prompt suite (`tiny`, `short`, `long`), each capping its
  own output so a chattier model does not measure as a slower one;
  `--prompt-set` to narrow it and `--prompts FILE` to replace it.
- `model-bench prompts`, printing exactly what will be sent.
- Statistics that decline to lie: `None` rather than zero or infinity for
  throughput that cannot be computed, sample (n−1) standard deviation,
  and no standard deviation at all from a single repetition.
- API keys read only from an environment variable named by
  `--api-key-env`, never from the command line.
- Zero runtime dependencies; `urllib.request` is the whole networking
  layer.

[0.1.0]: https://github.com/hellpuffyt/model-bench/releases/tag/v0.1.0
