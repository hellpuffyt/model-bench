"""The runtime interface every backend implements.

A runtime is the only part of model-bench that touches a model. It has
to answer three questions and nothing else:

* can I be reached at all (:meth:`Runtime.probe`),
* stream me a completion for this prompt (:meth:`Runtime.stream`),
* how much memory does the loaded model occupy (:meth:`Runtime.memory`).

Keeping the surface that small is what makes the statistics core
testable without a model: the harness only ever sees an iterator of
:class:`~model_bench.models.Chunk`, so a fake runtime that yields
recorded chunks is indistinguishable from a live one.

Timing goes through :attr:`Runtime.clock` rather than
:func:`time.perf_counter` directly. Live runtimes return the real clock;
the replay runtime returns a virtual one it advances itself, which is
what makes replaying a recording deterministic to the microsecond
instead of merely close.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator

from .models import Chunk, MemoryReading, PromptSpec

Clock = Callable[[], float]


class Runtime(ABC):
    """A thing that can generate tokens and be timed while doing so."""

    #: Short identifier used in reports and on the command line.
    name: str = "runtime"

    @property
    @abstractmethod
    def endpoint(self) -> str:
        """Where this runtime lives, for error messages and reports."""

    @property
    def clock(self) -> Clock:
        """The clock the harness should time this runtime with."""
        return time.perf_counter

    def probe(self) -> None:
        """Check the runtime is reachable, raising if it is not.

        The default does nothing, because a runtime that cannot fail to
        be reached (the replay runtime) has nothing to check. It is not
        abstract: forcing every backend to write an empty override would
        add noise without adding a check.
        """
        return None

    @abstractmethod
    def stream(self, model: str, prompt: PromptSpec) -> Iterator[Chunk]:
        """Yield chunks of a completion as they are produced."""

    def memory(self, model: str) -> MemoryReading | None:
        """What the runtime says the loaded model occupies, if anything.

        ``None`` means "this runtime cannot tell me", which the reports
        print as ``n/a``. It never means zero.
        """
        return None
