"""Expose ``review_tool_call`` as an AgentDojo pipeline element.

AgentDojo (https://github.com/ethz-spylab/agentdojo, MIT, arXiv 2406.13352)
builds an agent out of ``BasePipelineElement`` objects, each of which is called
as::

    query, runtime, env, messages, extra_args = element.query(
        query, runtime, env, messages, extra_args
    )

Tool calls are executed by ``agentdojo.agent_pipeline.ToolsExecutor``, which
reads ``messages[-1]["tool_calls"]`` (a list of
``agentdojo.functions_runtime.FunctionCall``), runs each one through
``runtime.run_function(env, name, args)`` and appends one
``ChatToolResultMessage`` per call. A *defence* in AgentDojo is nothing more
than an element (or a replacement for one) placed inside the
``ToolsExecutionLoop``; the built-in ones are listed in
``agentdojo.agent_pipeline.agent_pipeline.DEFENSES``.

:class:`JevGuardrail` is a drop-in replacement for ``ToolsExecutor``. For each
pending call it awaits :func:`server.jev.decisions.review_tool_call` with the
user task as the assignment and then behaves exactly as
``server/agents/execution_agent/runtime.py`` does:

* ``hold`` — the call is **not** executed. It comes back to the agent as a tool
  *error* carrying ``review.explain()``, so the agent can correct itself.
* ``warn``/``steer`` — the call runs and ``review.advice()`` rides along with
  the result.
* ``allow`` — nothing changes.

The module imports without ``agentdojo`` installed: the AgentDojo types are
optional, and everything the element needs from them is stated locally as a
:class:`typing.Protocol`. Pipeline messages are ``TypedDict``s in AgentDojo,
i.e. plain dicts at runtime, so the results this element appends are the same
objects with or without the package.

Targeted against agentdojo 0.1.35 / commit ``089ed468`` (2026-06-02).
"""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.jev.decisions import ALLOWED_TOOL_CALL, ToolCallReview, review_tool_call  # noqa: E402

try:  # pragma: no cover - exercised by the import-guard test
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement as _Base
    from agentdojo.functions_runtime import EmptyEnv as _EmptyEnv
    from agentdojo.types import text_content_block_from_string as _text_block

    AGENTDOJO_AVAILABLE = True
except Exception:  # pragma: no cover - the package is optional here
    # Deliberately broad: agentdojo pulls in openai/cohere/google clients, and a
    # version conflict in any of them must degrade to "not installed" rather
    # than take down a test run that never touches AgentDojo.
    _Base = object  # type: ignore[assignment,misc]
    _EmptyEnv = None  # type: ignore[assignment]

    def _text_block(content: str) -> Dict[str, Any]:  # type: ignore[misc]
        """Mirror of ``agentdojo.types.text_content_block_from_string``."""

        return {"type": "text", "content": content}

    AGENTDOJO_AVAILABLE = False


# The prefix under which an advisory verdict is handed to the agent. It mirrors
# the ``guardrail_note`` key `runtime.py` merges into a dict tool result; here
# the tool result is already formatted text, so it becomes a trailing line.
GUARDRAIL_NOTE_PREFIX = "guardrail_note: "


# ----------------------------------------------------------------------
# The bits of AgentDojo this element actually relies on
# ----------------------------------------------------------------------


class PendingCall(Protocol):
    """One pending tool call — ``agentdojo.functions_runtime.FunctionCall``."""

    function: str
    args: MutableMapping[str, Any]
    id: Optional[str]


class ToolRuntime(Protocol):
    """The part of ``FunctionsRuntime`` a tool executor uses."""

    def run_function(
        self,
        env: Any,
        function: str,
        kwargs: Mapping[str, Any],
        raise_on_error: bool = False,
    ) -> Tuple[Any, Optional[str]]:
        ...


def default_output_formatter(result: Any) -> str:
    """Format a tool result as text.

    AgentDojo's own default is ``agentdojo.agent_pipeline.tool_execution.
    tool_result_to_str`` (YAML). It is imported lazily so this module has no
    hard dependency on the package; without it we fall back to ``str``.
    """

    try:  # pragma: no cover - depends on the optional package
        from agentdojo.agent_pipeline.tool_execution import tool_result_to_str

        return tool_result_to_str(result)
    except Exception:
        return "" if result is None else str(result)


# ----------------------------------------------------------------------
# Assignment construction
# ----------------------------------------------------------------------


def build_assignment(query: str, *, suite: Optional[str] = None) -> str:
    """Build the ``assignment`` string the guardrail judges the call against.

    In AgentDojo the user task prompt is what ``InitQuery`` puts into the
    conversation and what every element receives as ``query``, so it is the
    direct analogue of the ``instructions`` the execution agent is given. The
    suite name is added only as scope context; no tool output ever reaches this
    string, which is the point — an assignment built from tool output would be
    attacker-controlled on both sides of the comparison.
    """

    task = (query or "").strip()
    if not suite:
        return task
    return f"You are the user's assistant in the {suite} workspace.\nThe user asked: {task}"


# ----------------------------------------------------------------------
# Running one coroutine from AgentDojo's synchronous pipeline
# ----------------------------------------------------------------------


def run_sync(coro: Any) -> Any:
    """Await ``coro`` from synchronous code, loop or no loop.

    AgentDojo pipeline elements are synchronous while ``review_tool_call`` is
    not. ``asyncio.run`` is enough on its own, except when a caller already has
    a loop running in this thread, where it raises; then the coroutine is
    driven on its own loop in a worker thread.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    box: Dict[str, Any] = {}

    def worker() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            box["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


def attach_note(text: str, note: str) -> str:
    """Return the tool output with the advisory note alongside it, never hiding it."""

    if not text:
        return f"{GUARDRAIL_NOTE_PREFIX}{note}"
    return f"{text}\n\n{GUARDRAIL_NOTE_PREFIX}{note}"


def tool_result_message(
    call: PendingCall, text: str, *, error: Optional[str] = None
) -> Dict[str, Any]:
    """Build the ``ChatToolResultMessage`` AgentDojo appends after a tool call."""

    return {
        "role": "tool",
        "content": [_text_block(text)],
        "tool_call_id": getattr(call, "id", None),
        "tool_call": call,
        "error": error,
    }


# ----------------------------------------------------------------------
# The defence
# ----------------------------------------------------------------------


class JevGuardrail(_Base):  # type: ignore[misc,valid-type]
    """A ``ToolsExecutor`` that reviews each call before running it.

    Drop it into an ``agentdojo.agent_pipeline.ToolsExecutionLoop`` in place of
    ``ToolsExecutor``; :func:`attach_to_pipeline` does that for a pipeline built
    by ``AgentPipeline.from_config``.
    """

    name = "jev-tool-guardrail"

    def __init__(
        self,
        *,
        tool_output_formatter: Optional[Callable[[Any], str]] = None,
        suite: Optional[str] = None,
        reviewer: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.output_formatter = tool_output_formatter or default_output_formatter
        self.suite = suite
        # Seam for tests and for a runner that wants to count reviews without
        # a network; defaults to the production decision.
        self._reviewer = reviewer or review_tool_call
        self.reviews: List[Dict[str, Any]] = []

    # Forget every review recorded so far, so one benchmark case is one batch
    def reset(self) -> None:
        """Clear the recorded reviews before running the next benchmark case."""

        self.reviews = []

    @property
    def counts(self) -> Dict[str, int]:
        """Return how many calls landed on each verdict, and how many ran."""

        tally = {"allow": 0, "warn": 0, "steer": 0, "hold": 0, "executed": 0, "reviewed": 0}
        for record in self.reviews:
            tally["reviewed"] += 1
            tally[record["verdict"]] = tally.get(record["verdict"], 0) + 1
            if record["executed"]:
                tally["executed"] += 1
        return tally

    # Review one pending call, failing open on any error
    def review(self, assignment: str, tool_name: str, arguments: Mapping[str, Any]) -> ToolCallReview:
        """Return the guardrail's verdict, or ``allow`` if it could not be had.

        Fail-open is the policy of the layer being measured (see
        ``server/jev/client.py``): a Jev outage must leave the agent behaving
        exactly as it did before the guardrail existed. Measuring a defence
        that silently fails *closed* would flatter it on ASR and hide the
        outage as utility loss, so the failure mode is reproduced here rather
        than papered over.
        """

        try:
            return run_sync(
                self._reviewer(
                    assignment=assignment,
                    tool_name=tool_name,
                    arguments=dict(arguments),
                )
            )
        except Exception:  # noqa: BLE001 - fail open, same as the server does
            return ALLOWED_TOOL_CALL

    def query(
        self,
        query: str,
        runtime: ToolRuntime,
        env: Any = None,
        messages: Sequence[Mapping[str, Any]] = (),
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, ToolRuntime, Any, Sequence[Mapping[str, Any]], Dict[str, Any]]:
        """Execute the pending tool calls, holding the ones the guardrail holds."""

        if env is None and _EmptyEnv is not None:  # pragma: no cover - needs agentdojo
            env = _EmptyEnv()
        if extra_args is None:
            extra_args = {}
        messages = list(messages)

        # The same three guards ToolsExecutor applies: nothing to do unless the
        # last message is an assistant turn that asked for tool calls.
        if not messages:
            return query, runtime, env, messages, extra_args
        last = messages[-1]
        if last.get("role") != "assistant":
            return query, runtime, env, messages, extra_args
        tool_calls = last.get("tool_calls") or []
        if not tool_calls:
            return query, runtime, env, messages, extra_args

        assignment = build_assignment(query, suite=self.suite)
        results: List[Dict[str, Any]] = []

        for call in tool_calls:
            tool_name = getattr(call, "function", "") or ""
            raw_args = getattr(call, "args", {}) or {}
            arguments = dict(raw_args) if isinstance(raw_args, Mapping) else {}

            review = self.review(assignment, tool_name, arguments)
            record: Dict[str, Any] = {
                "tool": tool_name,
                "verdict": review.verdict,
                "reason": review.reason,
                "probabilities": dict(review.probabilities),
            }

            if review.held:
                # Not executed. Reported as a tool error so the agent reads it
                # as a correctable failure rather than a refusal to the user.
                record["executed"] = False
                results.append(tool_result_message(call, "", error=review.explain()))
            else:
                output, error = runtime.run_function(env, tool_name, arguments)
                text = self.output_formatter(output)
                if review.advisory:
                    text = attach_note(text, review.advice())
                record["executed"] = True
                results.append(tool_result_message(call, text, error=error))

            self.reviews.append(record)

        return query, runtime, env, [*messages, *results], extra_args


def attach_to_pipeline(pipeline: Any, guardrail: JevGuardrail) -> Any:
    """Replace the ``ToolsExecutor`` inside ``pipeline`` with ``guardrail``.

    Works on a pipeline built by ``AgentPipeline.from_config``, whose tool loop
    is a ``ToolsExecutionLoop`` holding a ``ToolsExecutor`` and an LLM. Raises
    if no executor is found, because silently returning an unguarded pipeline
    would report the *undefended* numbers under the defence's name.
    """

    from agentdojo.agent_pipeline import ToolsExecutionLoop, ToolsExecutor

    replaced = 0
    for element in getattr(pipeline, "elements", []):
        if not isinstance(element, ToolsExecutionLoop):
            continue
        swapped = []
        for inner in element.elements:
            if isinstance(inner, ToolsExecutor):
                guardrail.output_formatter = inner.output_formatter
                swapped.append(guardrail)
                replaced += 1
            else:
                swapped.append(inner)
        element.elements = swapped

    if replaced == 0:
        raise ValueError("no ToolsExecutor found in the pipeline; the guardrail was not installed")
    return pipeline


__all__ = [
    "AGENTDOJO_AVAILABLE",
    "GUARDRAIL_NOTE_PREFIX",
    "JevGuardrail",
    "PendingCall",
    "ToolRuntime",
    "attach_note",
    "attach_to_pipeline",
    "build_assignment",
    "default_output_formatter",
    "run_sync",
    "tool_result_message",
]
