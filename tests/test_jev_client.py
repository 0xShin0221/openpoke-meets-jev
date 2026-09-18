"""Defensive-parsing and fail-open behaviour of the Jev client wrapper."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from server.jev import client as jev_client
from server.jev import questions as q


class _Answer:
    def __init__(self, value: Any) -> None:
        self.noul = value


class _Response:
    def __init__(self, answers: Any) -> None:
        self.answers = answers


@pytest.mark.parametrize(
    "response",
    [
        None,
        _Response(None),
        _Response({}),
        _Response({"important": object()}),
        _Response({"important": _Answer("0.9")}),
        _Response({"important": _Answer(True)}),
        _Response({"important": _Answer(float("nan"))}),
        _Response({"important": _Answer(1.5)}),
        _Response({"important": _Answer(-0.1)}),
    ],
)
def test_noul_rejects_unusable_answers(response: Any) -> None:
    assert jev_client.noul(response, "important") is None


def test_noul_accepts_boundary_values() -> None:
    assert jev_client.noul(_Response({"x": _Answer(0)}), "x") == 0.0
    assert jev_client.noul(_Response({"x": _Answer(1)}), "x") == 1.0


async def test_disabled_without_api_key() -> None:
    assert jev_client.is_enabled() is False
    assert await jev_client.get_client() is None
    assert (
        await jev_client.ask(state={"a": 1}, questions=q.EMAIL_QUESTIONS, purpose="test")
        is None
    )


async def test_ask_returns_none_for_empty_questions(jev_env: None) -> None:
    assert await jev_client.ask(state={"a": 1}, questions={}, purpose="test") is None


async def test_ask_sends_pinned_model_and_all_questions(jev_env: None, jev_transport) -> None:
    calls = jev_transport({key: {"type": "noul", "noul": 0.5} for key in q.EMAIL_QUESTIONS})

    response = await jev_client.ask(
        state={"subject": "hello"}, questions=q.EMAIL_QUESTIONS, purpose="test"
    )

    assert response is not None
    assert len(calls) == 1, "every question about one state must go in a single request"
    assert calls[0]["model"] == "jev-1.13.0"
    assert set(calls[0]["questions"]) == set(q.EMAIL_QUESTIONS)


@pytest.mark.parametrize("status", [401, 422, 429, 500, 529])
async def test_ask_fails_open_on_api_errors(jev_env: None, jev_transport, status: int) -> None:
    jev_transport({}, status=status)

    result = await jev_client.ask(
        state={"subject": "hello"}, questions=q.EMAIL_QUESTIONS, purpose="test"
    )

    assert result is None


async def test_ask_fails_open_on_malformed_success_body(jev_env: None, jev_transport) -> None:
    # A 200 whose body is missing `answers` raises inside the SDK; the wrapper
    # must absorb it rather than let it reach a watcher loop.
    jev_transport({}, body={"model": "jev-1.13.0"})

    result = await jev_client.ask(
        state={"subject": "hello"}, questions=q.EMAIL_QUESTIONS, purpose="test"
    )

    assert result is None


async def test_ask_fails_open_on_transport_failure(jev_env: None, monkeypatch) -> None:
    class _Boom:
        async def system_one(self, **_: Any) -> Dict[str, Any]:
            raise TimeoutError("connection reset")

    jev_client.set_client(_Boom())

    result = await jev_client.ask(
        state={"subject": "hello"}, questions=q.EMAIL_QUESTIONS, purpose="test"
    )

    assert result is None


async def test_cancellation_is_not_swallowed(jev_env: None) -> None:
    import asyncio

    class _Cancel:
        async def system_one(self, **_: Any) -> Dict[str, Any]:
            raise asyncio.CancelledError

    jev_client.set_client(_Cancel())

    with pytest.raises(asyncio.CancelledError):
        await jev_client.ask(
            state={"subject": "hello"}, questions=q.EMAIL_QUESTIONS, purpose="test"
        )
