"""Errors raised by model-bench.

A benchmark that reports a number it did not measure is worse than a
benchmark that refuses to run, so every way this tool can fail to obtain
a real measurement is an explicit exception rather than a fallback
value. In particular :class:`RuntimeUnavailable` is raised — never
swallowed — when a runtime cannot be reached, and nothing downstream
invents timings to take its place.
"""

from __future__ import annotations


class ModelBenchError(Exception):
    """Base class for every error this package raises deliberately."""


class UsageError(ModelBenchError):
    """The command line, or an API call, asked for something impossible."""


class RuntimeUnavailable(ModelBenchError):
    """A runtime could not be reached, or refused the request.

    ``endpoint`` is the URL that was tried, so the message can say what
    was unreachable rather than only that something was.
    """

    def __init__(self, runtime: str, endpoint: str, reason: str) -> None:
        self.runtime = runtime
        self.endpoint = endpoint
        self.reason = reason
        super().__init__(f"runtime {runtime!r} at {endpoint} is unavailable: {reason}")


class ProtocolError(ModelBenchError):
    """A runtime answered, but not with something this tool understands."""

    def __init__(self, runtime: str, detail: str) -> None:
        self.runtime = runtime
        self.detail = detail
        super().__init__(f"runtime {runtime!r} sent an unusable response: {detail}")


class RecordingError(ModelBenchError):
    """A recording file is missing, malformed, or of an unknown schema."""
