"""model-bench — measure local LLM inference, honestly.

Three numbers, at fixed prompts, across interchangeable runtimes: time
to first token, decode throughput in tokens per second, and the memory
the runtime says the loaded model occupies.

The public entry points are :func:`model_bench.cli.main` for the command
line and :func:`model_bench.harness.run_benchmark` for use as a library.
Every backend implements :class:`model_bench.runtime.Runtime`, which is
what lets the statistics core be exercised end to end against recorded
runs rather than a live model.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
