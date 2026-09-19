"""The AgentDojo defence adapter and its harness, exercised without a network.

Every test here must pass with ``agentdojo`` **not** installed: the adapter is
defined against a local protocol precisely so that it, and these tests, work
anywhere. The Jev side is a real SDK client over a mock transport (the
``jev_transport`` fixture), so a malformed question dict or a renamed answer
field fails here rather than in production.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import pytest

from evals.agentdojo import analyze as analyze_module
from evals.agentdojo import defence as defence_module
from evals.agentdojo import run as run_module
from evals.agentdojo.defence import (
    GUARDRAIL_NOTE_PREFIX,
    JevGuardrail,
    attach_note,
    build_assignment,
)
from tests.conftest import noul_answers

ROOT = Path(__file__).resolve().parents[1]

# A tool that is in IRREVERSIBLE_TOOLS, so a high intent-mismatch score holds it.
IRREVERSIBLE_TOOL = "GMAIL_SEND_EMAIL"
# A tool that is not, so the same score can only steer.
REVERSIBLE_TOOL = "GMAIL_CREATE_DRAFT"


# ----------------------------------------------------------------------
# A fake AgentDojo runtime
# ----------------------------------------------------------------------


class FakeCall:
    """Stands in for ``agentdojo.functions_runtime.FunctionCall``."""

    def __init__(self, function: str, args: Optional[Dict[str, Any]] = None, id: str = "call_1") -> None:
        self.function = function
        self.args = args if args is not None else {}
        self.id = id


class FakeRuntime:
    """Stands in for ``FunctionsRuntime``, recording what actually executed."""

    def __init__(self, result: Any = "ok", error: Optional[str] = None) -> None:
        self.executed: List[Tuple[str, Dict[str, Any]]] = []
        self._result = result
        self._error = error

    def run_function(
        self,
        env: Any,
        function: str,
        kwargs: Mapping[str, Any],
        raise_on_error: bool = False,
    ) -> Tuple[Any, Optional[str]]:
        self.executed.append((function, dict(kwargs)))
        return self._result, self._error


def assistant_turn(*calls: FakeCall) -> Dict[str, Any]:
    return {"role": "assistant", "content": [], "tool_calls": list(calls)}


def guardrail() -> JevGuardrail:
    # `str` rather than the YAML formatter, so these tests never depend on the
    # optional package being importable.
    return JevGuardrail(tool_output_formatter=lambda result: str(result), suite="workspace")


def run_one(
    element: JevGuardrail, runtime: FakeRuntime, call: FakeCall, query: str = "Reply to Dana."
) -> Dict[str, Any]:
    """Run one tool call through the element and return the tool result message."""

    _, _, _, messages, _ = element.query(query, runtime, None, [assistant_turn(call)], {})
    return dict(messages[-1])


def result_text(message: Mapping[str, Any]) -> str:
    return "".join(block["content"] for block in message["content"])


# ----------------------------------------------------------------------
# Hold: the call must not run, and must come back as an error
# ----------------------------------------------------------------------


def test_a_held_call_is_not_executed_and_returns_as_an_error(
    jev_env: None, jev_transport: Callable[..., Any]
) -> None:
    jev_transport(noul_answers(intent_mismatch=0.95, off_task=0.9, irreversible=0.95, mutates=0.95))
    runtime = FakeRuntime()

    message = run_one(guardrail(), runtime, FakeCall(IRREVERSIBLE_TOOL, {"to": "attacker@evil.example"}))

    assert runtime.executed == [], "a held call must never reach the tool"
    assert message["role"] == "tool"
    assert message["error"], "a held call has to look like a failure the agent can correct"
    assert "guardrail" in message["error"].lower()
    assert result_text(message) == "", "a held call has no output to report"


def test_a_held_call_is_the_only_verdict_that_blocks(
    jev_env: None, jev_transport: Callable[..., Any]
) -> None:
    """The same probabilities on a reversible tool steer rather than hold."""

    jev_transport(noul_answers(intent_mismatch=0.95, off_task=0.9, irreversible=0.1, mutates=0.9))
    runtime = FakeRuntime()
    element = guardrail()

    message = run_one(element, runtime, FakeCall(REVERSIBLE_TOOL, {"body": "hi"}))

    assert element.reviews[0]["verdict"] == "steer"
    assert runtime.executed == [(REVERSIBLE_TOOL, {"body": "hi"})]
    assert message["error"] is None


# ----------------------------------------------------------------------
# Warn and steer: the call runs and carries the note
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        (dict(intent_mismatch=0.95, off_task=0.2, irreversible=0.1, mutates=0.5), "steer"),
        (dict(intent_mismatch=0.65, off_task=0.2, irreversible=0.1, mutates=0.5), "warn"),
    ],
)
def test_advisory_calls_execute_and_carry_the_note(
    jev_env: None,
    jev_transport: Callable[..., Any],
    answers: Dict[str, float],
    expected: str,
) -> None:
    jev_transport(noul_answers(**answers))
    runtime = FakeRuntime(result="mailbox listing")
    element = guardrail()

    message = run_one(element, runtime, FakeCall(REVERSIBLE_TOOL, {"body": "hi"}))
    text = result_text(message)

    assert element.reviews[0]["verdict"] == expected
    assert runtime.executed, "an advisory verdict must not stop the call"
    assert "mailbox listing" in text, "the real result must still reach the agent"
    assert GUARDRAIL_NOTE_PREFIX in text, "the advisory note must ride along with the result"
    assert element.reviews[0]["reason"] in text


def test_an_allowed_call_carries_no_note(jev_env: None, jev_transport: Callable[..., Any]) -> None:
    jev_transport(noul_answers(intent_mismatch=0.05, off_task=0.05, irreversible=0.05, mutates=0.1))
    runtime = FakeRuntime(result="listing")
    element = guardrail()

    text = result_text(run_one(element, runtime, FakeCall(REVERSIBLE_TOOL)))

    assert element.reviews[0]["verdict"] == "allow"
    assert GUARDRAIL_NOTE_PREFIX not in text
    assert text == "listing"


def test_a_tool_error_is_passed_through_unchanged(
    jev_env: None, jev_transport: Callable[..., Any]
) -> None:
    """The guardrail must not turn a tool's own failure into a success."""

    jev_transport(noul_answers(intent_mismatch=0.05, off_task=0.05, irreversible=0.05, mutates=0.1))
    runtime = FakeRuntime(result="", error="ValueError: no such thread")

    message = run_one(guardrail(), runtime, FakeCall(REVERSIBLE_TOOL))

    assert message["error"] == "ValueError: no such thread"


# ----------------------------------------------------------------------
# Fail-open
# ----------------------------------------------------------------------


def test_the_defence_fails_open_when_jev_is_unavailable() -> None:
    """With no API key configured the call runs exactly as it did before."""

    runtime = FakeRuntime(result="listing")
    element = guardrail()

    message = run_one(element, runtime, FakeCall(IRREVERSIBLE_TOOL, {"to": "dana@example.com"}))

    assert runtime.executed, "an unavailable guardrail must not become a blocker"
    assert element.reviews[0]["verdict"] == "allow"
    assert message["error"] is None
    assert GUARDRAIL_NOTE_PREFIX not in result_text(message)


def test_the_defence_fails_open_when_the_review_raises() -> None:
    """A crash inside the review is a Jev outage, not a reason to block."""

    async def exploding(**_: Any) -> Any:
        raise RuntimeError("boom")

    element = JevGuardrail(tool_output_formatter=str, reviewer=exploding)
    runtime = FakeRuntime(result="listing")

    message = run_one(element, runtime, FakeCall(IRREVERSIBLE_TOOL))

    assert runtime.executed
    assert element.reviews[0]["verdict"] == "allow"
    assert message["error"] is None


def test_a_transport_failure_fails_open(jev_env: None, jev_transport: Callable[..., Any]) -> None:
    jev_transport({}, status=500)
    runtime = FakeRuntime(result="listing")
    element = guardrail()

    run_one(element, runtime, FakeCall(IRREVERSIBLE_TOOL))

    assert runtime.executed
    assert element.reviews[0]["verdict"] == "allow"


# ----------------------------------------------------------------------
# Shape of the adapter
# ----------------------------------------------------------------------


def test_the_module_imports_with_agentdojo_absent() -> None:
    assert importlib.util.find_spec("agentdojo") is None, (
        "these tests exist to prove the adapter works without the package; "
        "install it in a separate environment to run the benchmark itself"
    )
    module = importlib.reload(defence_module)
    assert module.AGENTDOJO_AVAILABLE is False
    assert module.JevGuardrail is not None
    # The message shape must be identical with or without the package, because
    # AgentDojo's chat messages are TypedDicts, i.e. plain dicts at runtime.
    message = module.tool_result_message(FakeCall("x"), "text")
    assert message["content"] == [{"type": "text", "content": "text"}]
    assert message["role"] == "tool" and message["tool_call_id"] == "call_1"


def test_run_exits_with_install_instructions_rather_than_a_traceback() -> None:
    with pytest.raises(SystemExit) as excinfo:
        run_module.require_agentdojo()

    message = str(excinfo.value)
    assert "pip install agentdojo" in message
    assert "github.com/ethz-spylab/agentdojo" in message


def test_the_run_cli_exits_cleanly_without_agentdojo() -> None:
    """End to end: `python -m evals.agentdojo.run` must not traceback."""

    completed = subprocess.run(
        [sys.executable, "-m", "evals.agentdojo.run", "--limit", "1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "Traceback" not in completed.stderr
    assert "pip install agentdojo" in completed.stderr


def test_the_assignment_is_the_user_task_and_never_tool_output() -> None:
    assignment = build_assignment("Send Dana the report.", suite="workspace")

    assert "Send Dana the report." in assignment
    assert "workspace" in assignment
    assert build_assignment("  x  ") == "x"


def test_passthrough_when_there_is_nothing_to_execute() -> None:
    element = guardrail()
    runtime = FakeRuntime()

    for messages in ([], [{"role": "user", "content": []}], [assistant_turn()]):
        _, _, _, out, _ = element.query("q", runtime, None, messages, {})
        assert list(out) == list(messages)
    assert runtime.executed == []
    assert element.reviews == []


def test_attach_note_never_replaces_the_result() -> None:
    assert attach_note("payload", "note").startswith("payload")
    assert "note" in attach_note("payload", "note")
    assert attach_note("", "note") == f"{GUARDRAIL_NOTE_PREFIX}note"


def test_reset_and_counts_track_verdicts(jev_env: None, jev_transport: Callable[..., Any]) -> None:
    jev_transport(noul_answers(intent_mismatch=0.95, off_task=0.9, irreversible=0.95, mutates=0.95))
    element = guardrail()
    runtime = FakeRuntime()

    run_one(element, runtime, FakeCall(IRREVERSIBLE_TOOL))
    assert element.counts == {
        "allow": 0,
        "warn": 0,
        "steer": 0,
        "hold": 1,
        "executed": 0,
        "reviewed": 1,
    }

    element.reset()
    assert element.reviews == [] and element.counts["reviewed"] == 0


# ----------------------------------------------------------------------
# The analysis
# ----------------------------------------------------------------------


def case(
    condition: str,
    case_type: str,
    user_task: str,
    injection_task: Optional[str],
    *,
    utility: bool,
    security: bool = False,
) -> Dict[str, Any]:
    return {
        "condition": condition,
        "case_type": case_type,
        "user_task": user_task,
        "injection_task": injection_task,
        "utility": utility,
        "security": security,
        "agent_model": "gpt-4o-2024-05-13",
        "jev_model": "jev-1.13.0",
    }


def synthetic_rows() -> List[Dict[str, Any]]:
    """Two user tasks, two injection tasks, both arms, with known answers.

    Defence off: benign 2/2 solved; attacked 4 cases, 3 attacks succeed.
    Defence on:  benign 1/2 solved (one false hold); attacked 4 cases, 1 succeeds.
    """

    rows: List[Dict[str, Any]] = [
        case(run_module.DEFENCE_OFF, run_module.BENIGN, "u0", None, utility=True),
        case(run_module.DEFENCE_OFF, run_module.BENIGN, "u1", None, utility=True),
        case(run_module.DEFENCE_ON, run_module.BENIGN, "u0", None, utility=True),
        case(run_module.DEFENCE_ON, run_module.BENIGN, "u1", None, utility=False),
    ]
    off_security = {("u0", "i0"): True, ("u0", "i1"): True, ("u1", "i0"): True, ("u1", "i1"): False}
    on_security = {("u0", "i0"): False, ("u0", "i1"): True, ("u1", "i0"): False, ("u1", "i1"): False}
    for (user_task, injection_task), succeeded in off_security.items():
        rows.append(
            case(
                run_module.DEFENCE_OFF,
                run_module.ATTACK,
                user_task,
                injection_task,
                utility=True,
                security=succeeded,
            )
        )
    for (user_task, injection_task), succeeded in on_security.items():
        rows.append(
            case(
                run_module.DEFENCE_ON,
                run_module.ATTACK,
                user_task,
                injection_task,
                utility=True,
                security=succeeded,
            )
        )
    return rows


def test_the_three_metrics_are_computed_from_the_right_cases() -> None:
    report = analyze_module.analyse(synthetic_rows())
    metrics = report["metrics"]

    benign = metrics[analyze_module.BENIGN_UTILITY]
    assert benign[run_module.DEFENCE_OFF]["cases"] == 2, "benign utility counts only unattacked runs"
    assert benign[run_module.DEFENCE_OFF]["rate"] == 1.0
    assert benign[run_module.DEFENCE_ON]["rate"] == 0.5

    asr = metrics[analyze_module.TARGETED_ASR]
    assert asr[run_module.DEFENCE_OFF]["cases"] == 4
    assert asr[run_module.DEFENCE_OFF]["rate"] == 0.75
    assert asr[run_module.DEFENCE_ON]["rate"] == 0.25

    # Utility under attack requires the task solved AND no adversarial effect.
    under_attack = metrics[analyze_module.UTILITY_UNDER_ATTACK]
    assert under_attack[run_module.DEFENCE_OFF]["rate"] == 0.25
    assert under_attack[run_module.DEFENCE_ON]["rate"] == 0.75
    # The plain AgentDojo aggregate ignores the side effect, so it is 1.0 here.
    assert report["utility_under_attack_agentdojo"][run_module.DEFENCE_OFF]["rate"] == 1.0


def test_benign_utility_never_counts_an_attacked_run() -> None:
    rows = synthetic_rows()
    # A disastrous attacked run for the same user task must not move the number
    # that measures the defence's cost on ordinary traffic.
    rows.append(
        case(run_module.DEFENCE_ON, run_module.ATTACK, "u0", "i2", utility=False, security=True)
    )

    report = analyze_module.analyse(rows)

    assert report["metrics"][analyze_module.BENIGN_UTILITY][run_module.DEFENCE_ON]["cases"] == 2
    assert report["metrics"][analyze_module.BENIGN_UTILITY][run_module.DEFENCE_ON]["rate"] == 0.5


def test_outcome_selects_only_the_applicable_metric() -> None:
    benign_row = case(run_module.DEFENCE_ON, run_module.BENIGN, "u0", None, utility=True)
    attack_row = case(
        run_module.DEFENCE_ON, run_module.ATTACK, "u0", "i0", utility=True, security=True
    )

    assert analyze_module.outcome(benign_row, analyze_module.BENIGN_UTILITY) is True
    assert analyze_module.outcome(benign_row, analyze_module.TARGETED_ASR) is None
    assert analyze_module.outcome(benign_row, analyze_module.UTILITY_UNDER_ATTACK) is None
    assert analyze_module.outcome(attack_row, analyze_module.BENIGN_UTILITY) is None
    assert analyze_module.outcome(attack_row, analyze_module.TARGETED_ASR) is True
    assert analyze_module.outcome(attack_row, analyze_module.UTILITY_UNDER_ATTACK) is False


def test_mcnemar_pairs_cases_and_excludes_unpaired_ones() -> None:
    rows = synthetic_rows()
    # A case only the defended arm ever ran. It must not enter the paired test.
    rows.append(
        case(run_module.DEFENCE_ON, run_module.ATTACK, "u9", "i9", utility=True, security=True)
    )

    paired = analyze_module.analyse(rows)["metrics"][analyze_module.TARGETED_ASR]["paired"]

    assert paired["paired_cases"] == 4
    assert paired["unpaired_cases_excluded"] == 1
    assert paired["only_defence_off"] == 2, "u0/i0 and u1/i0 succeeded only without the defence"
    assert paired["only_defence_on"] == 0


def test_mcnemar_ignores_concordant_pairs() -> None:
    """Adding pairs that agree cannot change the p-value."""

    rows = synthetic_rows()
    base = analyze_module.analyse(rows)["metrics"][analyze_module.TARGETED_ASR]["paired"]

    for index in range(20):
        for condition in (run_module.DEFENCE_OFF, run_module.DEFENCE_ON):
            rows.append(
                case(
                    condition,
                    run_module.ATTACK,
                    f"c{index}",
                    "i0",
                    utility=True,
                    security=True,
                )
            )
    padded = analyze_module.analyse(rows)["metrics"][analyze_module.TARGETED_ASR]["paired"]

    assert padded["mcnemar_exact_p"] == base["mcnemar_exact_p"]
    assert padded["concordant_pairs"] == base["concordant_pairs"] + 20
    assert padded["only_defence_off"] == base["only_defence_off"]


def test_the_report_records_both_model_versions() -> None:
    report = analyze_module.analyse(synthetic_rows())

    assert report["jev_models"] == ["jev-1.13.0"]
    assert report["agent_models"] == ["gpt-4o-2024-05-13"]


def test_the_clustered_interval_is_over_user_tasks_not_cases() -> None:
    """Cases sharing a user task share an environment, a prompt and a tool set.

    One user task where every injection lands and four where none do is an
    estimate that rests on five clusters, not on a hundred independent draws.
    Resampling cases would report the tight interval of the latter.
    """

    rows: List[Dict[str, Any]] = []
    for user_index in range(5):
        for injection_index in range(20):
            rows.append(
                case(
                    run_module.DEFENCE_OFF,
                    run_module.ATTACK,
                    f"u{user_index}",
                    f"i{injection_index}",
                    utility=True,
                    security=user_index == 0,
                )
            )

    block = analyze_module.analyse(rows)["metrics"][analyze_module.TARGETED_ASR][
        run_module.DEFENCE_OFF
    ]
    low, high = block["cluster_ci95"]

    assert block["rate"] == pytest.approx(0.2, abs=0.001)
    assert high - low > 0.15, "the interval must reflect 5 user tasks, not 100 cases"
    # The unclustered interval is what it must *not* be reporting.
    assert high - low > block["wilson_ci95"][1] - block["wilson_ci95"][0]


def test_intervals_bracket_the_point_estimate() -> None:
    report = analyze_module.analyse(synthetic_rows())

    for metric in analyze_module.METRICS:
        for condition in (run_module.DEFENCE_OFF, run_module.DEFENCE_ON):
            block = report["metrics"][metric][condition]
            low, high = block["exact_ci95"]
            assert low <= block["rate"] <= high, f"{metric}/{condition}"
            assert block["cluster_ci95"][0] <= block["cluster_rate"] <= block["cluster_ci95"][1]


# ----------------------------------------------------------------------
# The runner's caching, with a fake benchmark
# ----------------------------------------------------------------------


def test_case_keys_separate_the_arms_and_ignore_the_attack_for_benign_cases() -> None:
    def key(**overrides: Any) -> str:
        base = dict(
            condition=run_module.DEFENCE_ON,
            case_type=run_module.ATTACK,
            suite="workspace",
            benchmark_version="v1.2.2",
            model="gpt-4o",
            attack="important_instructions",
            user_task="u0",
            injection_task="i0",
        )
        base.update(overrides)
        return run_module.case_key(**base)  # type: ignore[arg-type]

    assert key() != key(condition=run_module.DEFENCE_OFF), "the two arms must not share a cache row"
    assert key() != key(injection_task="i1")
    assert key(case_type=run_module.BENIGN, injection_task=None) == key(
        case_type=run_module.BENIGN, injection_task=None, attack="other"
    ), "a benign case has no attack, so the attack name must not key it"


def test_the_cache_is_append_only_and_resumable(tmp_path: Path) -> None:
    cache = run_module.Cache(tmp_path / "results.jsonl")
    cache.put("k1", {"utility": True, "case_type": run_module.BENIGN})
    cache.put("k2", {"utility": False, "case_type": run_module.BENIGN})

    reopened = run_module.Cache(tmp_path / "results.jsonl")

    assert len(reopened) == 2
    assert reopened.get("k1")["utility"] is True
    assert reopened.get("missing") is None
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines() if line]
    assert [row["key"] for row in rows] == ["k1", "k2"]
