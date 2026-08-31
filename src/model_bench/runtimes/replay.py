"""The replay runtime: a recording, played back through the real harness.

This is not a mock in the testing sense. It implements the same
:class:`~model_bench.runtime.Runtime` interface the live backends do, and
the harness cannot tell the difference — which is the point, because it
means the code path that produced a live report is the code path a
replayed report is checked against.

What makes replay exact rather than approximate is
:class:`VirtualClock`. The harness never calls :func:`time.perf_counter`
itself; it times whatever ``runtime.clock`` returns. A live runtime hands
it the real clock; this one hands it a clock it moves forward to each
recorded offset. So a replay is instantaneous, deterministic on every
platform, and reproduces the original timings to the last decimal — the
property the ``dogfood`` CI job asserts against a committed fixture.
"""

from __future__ import annotations

from collections.abc import Iterator

from ..errors import RecordingError
from ..models import Chunk, MemoryReading, PromptSpec
from ..recording import RecordedStream, Recording
from ..runtime import Clock, Runtime


class VirtualClock:
    """A clock that only moves when it is told to."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance_to(self, moment: float) -> None:
        """Move forward to ``moment``; never backwards.

        Time going backwards inside a measurement would produce a
        negative duration, which the sample constructor rejects. Catching
        it here names the real cause instead.
        """
        if moment < self.now:
            raise RecordingError(
                f"replay tried to move the clock backwards to {moment} from {self.now}"
            )
        self.now = moment


class ReplayRuntime(Runtime):
    """Serves recorded streams, in the order they were recorded."""

    name = "replay"

    def __init__(self, recording: Recording) -> None:
        self.recording = recording
        self._clock = VirtualClock()
        self._pending: dict[str, list[RecordedStream]] = {}
        for stream in recording.streams:
            self._pending.setdefault(stream.prompt, []).append(stream)

    @property
    def endpoint(self) -> str:
        return self.recording.endpoint

    @property
    def clock(self) -> Clock:
        return self._clock

    @property
    def source_runtime(self) -> str:
        """The runtime that produced the recording, for the report."""
        return self.recording.runtime

    def remaining(self, prompt: str) -> int:
        """How many recorded repetitions of ``prompt`` are unplayed."""
        return len(self._pending.get(prompt, ()))

    def stream(self, model: str, prompt: PromptSpec) -> Iterator[Chunk]:
        queue = self._pending.get(prompt.name)
        if not queue:
            known = ", ".join(self.recording.prompts)
            raise RecordingError(
                f"recording has no unplayed run of prompt {prompt.name!r}; "
                f"it contains: {known}"
            )
        recorded = queue.pop(0)
        start = self._clock.now
        for chunk in recorded.chunks:
            self._clock.advance_to(start + chunk.at_s)
            yield Chunk(text=chunk.text)
        self._clock.advance_to(start + recorded.end_s)
        yield Chunk(
            text="",
            done=True,
            output_tokens=recorded.reported_output_tokens,
            prompt_tokens=recorded.reported_prompt_tokens,
        )

    def memory(self, model: str) -> MemoryReading | None:
        return self.recording.memory
