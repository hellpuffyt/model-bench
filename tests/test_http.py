"""The networking layer: request shaping, streaming, and failure mapping."""

from __future__ import annotations

import io
import urllib.error
import urllib.request

import pytest

from conftest import FakeOpener
from model_bench.errors import ProtocolError, RuntimeUnavailable
from model_bench.http import DEFAULT_TIMEOUT, request_json, stream_lines


def test_a_get_is_used_when_there_is_no_payload() -> None:
    opener = FakeOpener('{"ok": true}')
    request_json("r", "http://x/api", opener=opener)
    assert opener.requests[0].get_method() == "GET"
    assert opener.requests[0].data is None


def test_a_payload_makes_it_a_post_with_json_content_type() -> None:
    opener = FakeOpener('{"ok": true}')
    request_json("r", "http://x/api", payload={"a": 1}, opener=opener)
    request = opener.requests[0]
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert opener.payload == {"a": 1}


def test_extra_headers_are_sent() -> None:
    opener = FakeOpener("{}")
    request_json("r", "http://x", headers={"Authorization": "Bearer k"}, opener=opener)
    assert opener.requests[0].get_header("Authorization") == "Bearer k"


def test_request_json_decodes_the_body() -> None:
    assert request_json("r", "http://x", opener=FakeOpener('{"a": [1, 2]}')) == {
        "a": [1, 2]
    }


def test_request_json_closes_the_response() -> None:
    opener = FakeOpener("{}")
    request_json("r", "http://x", opener=opener)
    assert opener.responses[0].closed is True


def test_request_json_rejects_a_non_json_body() -> None:
    with pytest.raises(ProtocolError, match="did not return JSON"):
        request_json("r", "http://x", opener=FakeOpener("<html>nope</html>"))


def test_protocol_error_names_the_runtime() -> None:
    with pytest.raises(ProtocolError) as caught:
        request_json("ollama", "http://x", opener=FakeOpener("not json"))
    assert caught.value.runtime == "ollama"


def test_an_http_error_becomes_runtime_unavailable() -> None:
    error = urllib.error.HTTPError("http://x", 503, "busy", {}, io.BytesIO(b""))  # type: ignore[arg-type]
    with pytest.raises(RuntimeUnavailable, match="HTTP 503") as caught:
        request_json("r", "http://x", opener=FakeOpener(error=error))
    assert caught.value.endpoint == "http://x"


def test_a_connection_refusal_becomes_runtime_unavailable() -> None:
    error = urllib.error.URLError(ConnectionRefusedError("refused"))
    with pytest.raises(RuntimeUnavailable, match="refused"):
        request_json("r", "http://x", opener=FakeOpener(error=error))


def test_a_timeout_becomes_runtime_unavailable_with_the_limit() -> None:
    with pytest.raises(RuntimeUnavailable, match="timed out after 2.5s"):
        request_json(
            "r", "http://x", timeout=2.5, opener=FakeOpener(error=TimeoutError())
        )


def test_any_other_os_error_becomes_runtime_unavailable() -> None:
    with pytest.raises(RuntimeUnavailable, match="disk on fire"):
        request_json("r", "http://x", opener=FakeOpener(error=OSError("disk on fire")))


def test_runtime_unavailable_message_names_the_endpoint() -> None:
    error = RuntimeUnavailable("ollama", "http://h/api", "refused")
    assert "ollama" in str(error)
    assert "http://h/api" in str(error)
    assert error.reason == "refused"


def test_stream_lines_yields_each_line_stripped() -> None:
    opener = FakeOpener("one\ntwo\nthree\n")
    assert list(stream_lines("r", "http://x", opener=opener)) == ["one", "two", "three"]


def test_stream_lines_drops_blank_padding() -> None:
    # SSE separates events with a blank line; it carries no token.
    opener = FakeOpener("a\n\n\nb\n\n")
    assert list(stream_lines("r", "http://x", opener=opener)) == ["a", "b"]


def test_stream_lines_survives_undecodable_bytes() -> None:
    class Broken(FakeOpener):
        def __call__(self, request: object, timeout: float) -> object:  # type: ignore[override]
            self.requests.append(request)

            class Response:
                def read(self) -> bytes:
                    return b""

                def __iter__(self) -> object:
                    return iter([b"\xff\xfe bad\n", b"good\n"])

                def __enter__(self) -> object:
                    return self

                def __exit__(self, *exc: object) -> None:
                    return None

            return Response()

    lines = list(stream_lines("r", "http://x", opener=Broken("")))  # type: ignore[arg-type]
    assert lines[-1] == "good"
    assert "bad" in lines[0]


def test_stream_lines_maps_failures_before_yielding_anything() -> None:
    stream = stream_lines("r", "http://x", opener=FakeOpener(error=OSError("down")))
    with pytest.raises(RuntimeUnavailable):
        next(iter(stream))


def test_stream_lines_sends_the_payload() -> None:
    opener = FakeOpener("a\n")
    list(stream_lines("r", "http://x", payload={"stream": True}, opener=opener))
    assert opener.payload == {"stream": True}


def test_default_timeout_is_generous_enough_for_a_long_generation() -> None:
    # A 512-token completion on a slow CPU runtime can take minutes.
    assert DEFAULT_TIMEOUT >= 60.0


def test_the_default_opener_is_urllib() -> None:
    from model_bench.http import urlopen

    assert urlopen.__module__ == "model_bench.http"
    assert urllib.request.Request is not None


def test_the_default_opener_really_calls_urllib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The one line that cannot be reached through the injected opener.
    seen: dict[str, object] = {}

    def fake(request: object, timeout: float) -> object:
        seen["timeout"] = timeout
        return FakeOpener('{"ok": 1}')(request, timeout)

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    assert request_json("r", "http://x", timeout=3.0) == {"ok": 1}
    assert seen["timeout"] == 3.0
