"""A very small HTTP client over :mod:`urllib.request`.

model-bench has no runtime dependencies, so this is the whole networking
layer. It exists mostly to do two things properly.

**Stream without buffering.** Time to first token is only meaningful if
the client hands each line up the moment it arrives, so responses are
iterated line by line off the socket and never read whole.

**Turn every network failure into one exception.** A connection refused,
a DNS failure, a timeout and an HTTP 500 all mean the same thing to a
benchmark — no measurement — and all raise
:class:`~model_bench.errors.RuntimeUnavailable` with the endpoint in the
message.

The opener is injected rather than imported at the call site, so the
tests exercise these paths against fakes instead of a live server.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping
from typing import Any, Protocol

from .errors import ProtocolError, RuntimeUnavailable

DEFAULT_TIMEOUT = 120.0


class _Response(Protocol):
    """The part of :class:`http.client.HTTPResponse` this module uses."""

    def read(self) -> bytes: ...
    def __iter__(self) -> Iterator[bytes]: ...
    def __enter__(self) -> Any: ...
    def __exit__(self, *exc: Any) -> Any: ...


class Opener(Protocol):
    """``urllib.request.urlopen``, narrowed to the call we make."""

    def __call__(
        self, request: urllib.request.Request, timeout: float
    ) -> _Response: ...


def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
    """The default opener. Named so it can be substituted in tests."""
    response: _Response = urllib.request.urlopen(request, timeout=timeout)
    return response


def _open(
    opener: Opener,
    runtime: str,
    url: str,
    payload: Mapping[str, Any] | None,
    headers: Mapping[str, str],
    timeout: float,
) -> _Response:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    request.add_header("Accept", "application/json")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    for name, value in headers.items():
        request.add_header(name, value)
    try:
        return opener(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        raise RuntimeUnavailable(runtime, url, f"HTTP {error.code}") from error
    except urllib.error.URLError as error:
        raise RuntimeUnavailable(runtime, url, str(error.reason)) from error
    except TimeoutError as error:
        raise RuntimeUnavailable(runtime, url, f"timed out after {timeout}s") from error
    except OSError as error:
        raise RuntimeUnavailable(runtime, url, str(error)) from error


def request_json(
    runtime: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    opener: Opener = urlopen,
) -> Any:
    """One request, whole response, decoded as JSON."""
    response = _open(opener, runtime, url, payload, headers or {}, timeout)
    with response:
        body = response.read()
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError(runtime, f"{url} did not return JSON: {error}") from error


def stream_lines(
    runtime: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    opener: Opener = urlopen,
) -> Iterator[str]:
    """Yield decoded, non-empty lines as they arrive.

    Blank lines are dropped here rather than in each runtime: both the
    NDJSON and the server-sent-events protocols use them only as
    padding, and neither carries a token in one.
    """
    response = _open(opener, runtime, url, payload, headers or {}, timeout)
    with response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                yield line
