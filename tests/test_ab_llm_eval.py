"""The Jev-versus-LLM A/B harness, exercised without a network or an API key.

The comparison this harness makes is only worth anything if the two arms were
asked the same question about the same data. Most of what follows is testing
exactly that, because it is the claim a reader cannot check from the output
alone -- and because it is the claim that breaks silently: a run with diverging
inputs produces a perfectly well-formed report full of numbers that mean
nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pytest

from evals.ab_llm import analyze as analyze_module
from evals.ab_llm import run as run_module
from evals.ab_llm.run import ARM_JEV, ARM_LLM, Answer, BackendResponse, Usage
from evals.contamination.stats import mcnemar_exact


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------


class RecordingBackend(run_module.Backend):
    """A backend that answers from a table and records exactly what it was sent.

    It stores the *canonical serialisation* of the state and questions rather
    than the objects, so the identity test compares bytes and cannot be fooled
    by two dicts that merely compare equal after a mutation reorders or retypes
    a field.
    """

    def __init__(
        self,
        identifier: str,
        *,
        scores: Optional[Mapping[str, float]] = None,
        default: float = 0.2,
        probability_source: str = run_module.PROB_SOURCE_VERBALISED,
        cost_per_million: float = 0.15,
        latency_ms: int = 0,
        input_tokens: int = 100,
    ) -> None:
        self.identifier = identifier
        self.probability_source = probability_source
        self.input_cost_per_million = cost_per_million
        self._scores = dict(scores or {})
        self._default = default
        self._latency_ms = latency_ms
        self._input_tokens = input_tokens
        self.seen: List[Tuple[str, str]] = []  # (canonical state, canonical questions)
        self.question_objects: List[int] = []

    async def system_one(self, *, state: Mapping[str, Any], questions: Mapping[str, Any]) -> Any:
        self.seen.append((run_module.canonical(state), run_module.canonical(questions)))
        self.question_objects.append(id(questions))
        subject = str(state.get("subject", ""))
        score = self._scores.get(subject, self._default)
        return BackendResponse(
            answers={name: Answer(noul=score if name == "important" else 0.0) for name in questions},
            model=self.identifier,
            usage=Usage(input_tokens=self._input_tokens),
        )


class BrokenBackend(run_module.Backend):
    identifier = "broken"
    probability_source = run_module.PROB_SOURCE_VERBALISED

    async def system_one(self, **_: Any) -> Any:
        raise RuntimeError("boom")


def _backends(**overrides: Any) -> Dict[str, run_module.Backend]:
    return {
        ARM_JEV: RecordingBackend(
            "typesafe_sdk:jev-1.13.0",
            default=0.9,
            probability_source=run_module.PROB_SOURCE_NOUL,
            cost_per_million=run_module.JEV_INPUT_COST_PER_MILLION,
            **overrides.get(ARM_JEV, {}),
        ),
        ARM_LLM: RecordingBackend(
            "openrouter:test-model",
            default=0.3,
            **overrides.get(ARM_LLM, {}),
        ),
    }


async def _run(tmp_path: Path, backends: Mapping[str, run_module.Backend], **kwargs: Any) -> Dict[str, Any]:
    return await run_module.run(
        out_dir=tmp_path,
        repeats=kwargs.pop("repeats", 1),
        backends=backends,
        concurrency=kwargs.pop("concurrency", 4),
        **kwargs,
    )


def _rows(tmp_path: Path) -> List[Dict[str, Any]]:
    text = (tmp_path / "responses.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ----------------------------------------------------------------------
# The validity claim: identical inputs
# ----------------------------------------------------------------------


async def test_both_arms_receive_byte_identical_state_and_questions(tmp_path: Path) -> None:
    """The whole experiment rests on this and nothing in the output reveals it.

    If one arm is handed a reworded question or a differently-built state, every
    downstream number -- agreement, McNemar, cost-weighted loss -- is comparing
    two different experiments while looking exactly as it should.
    """

    backends = _backends()
    await _run(tmp_path, backends, repeats=2)

    jev = sorted(backends[ARM_JEV].seen)
    llm = sorted(backends[ARM_LLM].seen)

    assert jev, "the run made no calls"
    assert jev == llm, "the two arms did not see byte-identical inputs"

    questions_seen = {questions for _, questions in jev + llm}
    assert len(questions_seen) == 1, "the question set must not vary by arm or by trial"

    # Not merely equal: the same object, so no code path can rewrite one copy.
    assert len(set(backends[ARM_JEV].question_objects + backends[ARM_LLM].question_objects)) == 1


async def test_every_row_carries_the_hashes_that_prove_the_pairing(tmp_path: Path) -> None:
    """A reader must be able to verify the pairing from the output file alone."""

    backends = _backends()
    summary = await _run(tmp_path, backends, repeats=1)
    rows = _rows(tmp_path)

    assert rows
    by_key: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((row["carrier_id"], row["repeat"]), []).append(row)

    for (carrier_id, _), pair in by_key.items():
        assert len(pair) == 2, carrier_id
        assert pair[0]["state_sha256"] == pair[1]["state_sha256"], carrier_id
        assert pair[0]["questions_sha256"] == pair[1]["questions_sha256"], carrier_id
        assert len(pair[0]["state_sha256"]) == 64

    assert {row["questions_sha256"] for row in rows} == {summary["questions_sha256"]}
    for row in rows:
        assert row["backend"], "a result with no attributable backend is unusable"
        assert "body" not in row, "the cache stores answers, not a copy of the corpus"


def test_the_question_set_is_not_parameterised_by_arm() -> None:
    """A parameterless builder is what makes per-arm wording drift impossible."""

    import inspect

    signature = inspect.signature(run_module.questions_for_trial)
    assert not signature.parameters, "questions_for_trial must not take an arm"

    from server.jev import questions as server_questions

    assert run_module.questions_for_trial() == server_questions.EMAIL_QUESTIONS


# ----------------------------------------------------------------------
# Running: caching, resumability, failures
# ----------------------------------------------------------------------


async def test_a_resumed_run_spends_nothing(tmp_path: Path) -> None:
    backends = _backends()
    first = await _run(tmp_path, backends, repeats=1)
    calls_after_first = len(backends[ARM_JEV].seen) + len(backends[ARM_LLM].seen)

    second = await _run(tmp_path, backends, repeats=1)

    assert first["called"] > 0 and first["cached"] == 0
    assert second["called"] == 0, "a resumed run must spend nothing"
    assert second["cached"] == first["called"]
    assert second["estimated_cost_usd"] == 0.0
    assert len(backends[ARM_JEV].seen) + len(backends[ARM_LLM].seen) == calls_after_first


async def test_changing_the_backend_behind_an_arm_invalidates_its_cache(tmp_path: Path) -> None:
    """Otherwise a model swap silently reports the previous model's answers."""

    await _run(tmp_path, _backends(), repeats=1)

    swapped = _backends()
    swapped[ARM_LLM].identifier = "openrouter:a-different-model"
    summary = await _run(tmp_path, swapped, repeats=1)

    assert summary["per_arm"][ARM_LLM]["called"] > 0
    assert summary["per_arm"][ARM_JEV]["called"] == 0


async def test_failed_calls_are_recorded_but_never_counted_as_data(tmp_path: Path) -> None:
    """A failure is a missing measurement, not a measurement of zero."""

    backends = {ARM_JEV: RecordingBackend("jev", default=0.9), ARM_LLM: BrokenBackend()}
    summary = await _run(tmp_path, backends, repeats=1)

    expected = summary["trials"] // 2
    assert summary["per_arm"][ARM_LLM]["failed"] == expected
    assert summary["per_arm"][ARM_LLM]["called"] == 0
    assert summary["per_arm"][ARM_JEV]["failed"] == 0
    assert summary["per_arm"][ARM_JEV]["called"] == expected

    rows = _rows(tmp_path)
    assert any("error" in row for row in rows)

    loaded = analyze_module.load(tmp_path)
    assert loaded, "the healthy arm's rows must survive"
    assert all("error" not in row for row in loaded)
    assert all(row["arm"] == ARM_JEV for row in loaded), "a failed call became a data point"

    report = analyze_module.analyse(loaded)
    assert ARM_LLM not in report["arms"]
    assert report["paired"] is None, "there is nothing to pair against"


def test_load_excludes_a_failed_call_even_when_it_carries_an_answer(tmp_path: Path) -> None:
    """The ``error`` field is the exclusion, not the absence of an answer.

    A call can fail *after* something answer-shaped exists -- a timeout on a
    retry, a partial parse, a provider returning a body alongside an error. If
    the only guard were "does this row have answers", such a row would be
    averaged in at whatever value happened to be there, and a timeout recorded
    as 0.0 would read as the arm confidently saying "not important".
    """

    rows = [
        {"arm": ARM_JEV, "carrier_id": "c00", "repeat": 0, "answers": {"important": 0.9}},
        {
            "arm": ARM_JEV,
            "carrier_id": "c01",
            "repeat": 0,
            "error": "TimeoutError: boom",
            "answers": {"important": 0.0},
        },
        {"arm": ARM_JEV, "carrier_id": "c02", "repeat": 0, "answers": {"important": None}},
        {"arm": ARM_JEV, "carrier_id": "c03", "repeat": 0, "error": "RuntimeError: boom"},
    ]
    (tmp_path / "responses.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    loaded = analyze_module.load(tmp_path)

    assert [row["carrier_id"] for row in loaded] == ["c00"]


async def test_arms_are_costed_at_their_own_rate(tmp_path: Path) -> None:
    """Jev's $0.042/1M is not the LLM's price; one shared constant would lie."""

    backends = _backends()
    summary = await _run(tmp_path, backends, repeats=1)

    jev = summary["per_arm"][ARM_JEV]
    llm = summary["per_arm"][ARM_LLM]
    assert jev["input_cost_per_million"] == pytest.approx(0.042)
    assert jev["input_tokens"] == llm["input_tokens"]
    assert llm["estimated_cost_usd"] > jev["estimated_cost_usd"]
    assert jev["estimated_cost_usd"] == pytest.approx(
        jev["input_tokens"] * 0.042 / 1_000_000, abs=1e-9
    )


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------


def _row(
    arm: str,
    carrier_id: str,
    repeat: int,
    probability: float,
    *,
    source: str = run_module.PROB_SOURCE_VERBALISED,
    latency_ms: int = 10,
    tokens: int = 100,
    rate: float = 0.15,
) -> Dict[str, Any]:
    return {
        "arm": arm,
        "carrier_id": carrier_id,
        "repeat": repeat,
        "backend": f"{arm}-backend",
        "probability_source": source,
        "input_cost_per_million": rate,
        "input_tokens": tokens,
        "latency_ms": latency_ms,
        "answers": {"important": probability},
    }


def _paired_rows(
    jev_scores: Sequence[float], llm_scores: Sequence[float]
) -> List[Dict[str, Any]]:
    rows = [
        _row(ARM_JEV, f"c{i:02d}", 0, p, source=run_module.PROB_SOURCE_NOUL, rate=0.042)
        for i, p in enumerate(jev_scores)
    ]
    rows += [_row(ARM_LLM, f"c{i:02d}", 0, p) for i, p in enumerate(llm_scores)]
    return rows


def test_mcnemar_uses_paired_rows_only_and_excludes_the_unmatched(tmp_path: Path) -> None:
    """Rows one arm produced and the other did not carry no paired information.

    If they leaked in, a run where the LLM arm timed out on the hard mail would
    report the two arms as agreeing about mail only one of them ever saw.
    """

    rows = _paired_rows([0.9, 0.9, 0.1, 0.1], [0.1, 0.9, 0.9, 0.1])
    # Two carriers only the jev arm answered. Both would be discordant if the
    # pairing were a union instead of an intersection.
    rows.append(_row(ARM_JEV, "orphan-a", 0, 0.99, source=run_module.PROB_SOURCE_NOUL))
    rows.append(_row(ARM_JEV, "orphan-b", 0, 0.01, source=run_module.PROB_SOURCE_NOUL))

    paired = analyze_module.analyse(rows)["paired"]

    assert paired["paired_rows"] == 4
    assert paired["unpaired_rows_excluded"] == 2
    assert paired[f"surfaced_only_by_{ARM_JEV}"] == 1
    assert paired[f"surfaced_only_by_{ARM_LLM}"] == 1
    assert paired["agreement"] == pytest.approx(0.5)
    assert paired["mcnemar_exact_p_surfacing"] == pytest.approx(mcnemar_exact(1, 1))


def test_mcnemar_ignores_concordant_pairs() -> None:
    """Adding pairs both arms agree on must not move the test statistic."""

    base = _paired_rows([0.9, 0.1, 0.9], [0.1, 0.9, 0.9])
    extra = [
        _row(ARM_JEV, f"agree{i}", 0, 0.95, source=run_module.PROB_SOURCE_NOUL) for i in range(20)
    ] + [_row(ARM_LLM, f"agree{i}", 0, 0.95) for i in range(20)]

    first = analyze_module.analyse(base)["paired"]["mcnemar_exact_p_surfacing"]
    second = analyze_module.analyse(base + extra)["paired"]["mcnemar_exact_p_surfacing"]

    assert first == second


def test_without_labels_no_accuracy_is_reported_and_the_gap_is_named() -> None:
    """Agreement is not accuracy. The report has to say so rather than imply it."""

    report = analyze_module.analyse(_paired_rows([0.9, 0.1], [0.9, 0.1]))

    assert report["labels_available"] is False
    assert any("Agreement is not accuracy" in w for w in report["warnings"])
    for arm in (ARM_JEV, ARM_LLM):
        assert report["arms"][arm]["accuracy"] is None
        assert report["arms"][arm]["brier"] is None
        assert report["arms"][arm]["cost_weighted_optimum"] is None
        # The shape is still reported, because it costs no label to report it.
        assert report["arms"][arm]["reliability"]
        assert all(entry["empirical_rate"] is None for entry in report["arms"][arm]["reliability"])
    assert report["paired"]["mcnemar_exact_p_accuracy"] is None
    assert report["paired"]["cost_weighted"] is None


def test_with_labels_accuracy_precision_recall_and_brier_are_reported() -> None:
    labels = {"c00": True, "c01": True, "c02": False, "c03": False}
    rows = _paired_rows([0.9, 0.1, 0.9, 0.1], [0.9, 0.9, 0.1, 0.1])

    report = analyze_module.analyse(rows, labels=labels)

    jev = report["arms"][ARM_JEV]["accuracy"]
    assert (jev["true_positives"], jev["false_negatives"]) == (1, 1)
    assert (jev["false_positives"], jev["true_negatives"]) == (1, 1)
    assert jev["accuracy"] == pytest.approx(0.5)
    assert jev["precision"] == pytest.approx(0.5)
    assert jev["recall"] == pytest.approx(0.5)

    llm = report["arms"][ARM_LLM]["accuracy"]
    assert llm["accuracy"] == pytest.approx(1.0)
    assert report["arms"][ARM_LLM]["brier"] < report["arms"][ARM_JEV]["brier"]

    paired = report["paired"]
    assert paired[f"correct_only_{ARM_LLM}"] == 2
    assert paired[f"correct_only_{ARM_JEV}"] == 0
    assert paired["mcnemar_exact_p_accuracy"] == pytest.approx(mcnemar_exact(0, 2))


def test_labels_file_ignores_values_it_does_not_understand(tmp_path: Path) -> None:
    """A "borderline" annotation is not a label and must not become one."""

    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps({"c00": "important", "c01": "not_important", "c02": "borderline"}),
        encoding="utf-8",
    )

    labels = analyze_module.load_labels(path)

    assert labels == {"c00": True, "c01": False}
    assert analyze_module.load_labels(tmp_path / "missing.json") is None


# ----------------------------------------------------------------------
# Cost-weighted comparison
# ----------------------------------------------------------------------


def test_each_arm_is_scored_at_its_own_optimal_threshold() -> None:
    """A shared cut point measures the cut point as much as it measures the arm.

    Here the LLM arm separates the classes perfectly but does it around 0.4,
    below the server's 0.75 bar. At the shared bar it looks useless; at its own
    optimum it is perfect. Handicapping it to the other arm's operating point
    would invert the conclusion.
    """

    labels = {f"c{i:02d}": i < 2 for i in range(4)}
    jev_scores = [0.9, 0.7, 0.8, 0.1]        # the classes overlap: 0.7 true, 0.8 false
    llm_scores = [0.45, 0.42, 0.20, 0.10]    # perfect, but never reaches 0.75

    report = analyze_module.analyse(_paired_rows(jev_scores, llm_scores), labels=labels, cost_ratio=5.0)

    shared_bar = report["arms"][ARM_LLM]["accuracy"]
    assert shared_bar["true_positives"] == 0, "at the shared bar the LLM arm surfaces nothing"

    cost = report["paired"]["cost_weighted"]
    assert cost[ARM_LLM]["loss"] == pytest.approx(0.0), "its own optimum separates the classes"
    assert cost[ARM_LLM]["threshold"] <= 0.45
    assert cost[ARM_JEV]["loss"] > 0.0
    assert cost["better"] == ARM_LLM


def test_the_cost_ratio_moves_the_optimal_threshold_down() -> None:
    """A costlier false negative has to buy a more permissive threshold."""

    # Deliberately overlapping: 0.55 is important and 0.6 is not, so no
    # threshold is free and the ratio has to decide which error to buy.
    pairs = [(0.9, True), (0.55, True), (0.6, False), (0.2, False)]

    cheap = analyze_module.optimal_cost_weighted_threshold(pairs, cost_ratio=0.2)
    dear = analyze_module.optimal_cost_weighted_threshold(pairs, cost_ratio=20.0)

    assert cheap["threshold"] > dear["threshold"]
    assert dear["false_negatives"] == 0, "at 20:1 no true positive is worth dropping"
    assert cheap["false_positives"] == 0, "at 1:5 no false positive is worth accepting"


def test_optimal_loss_is_never_beaten_by_any_other_threshold() -> None:
    """The sweep must actually find the minimum, not the first decent value."""

    pairs = [(0.05, False), (0.3, True), (0.5, False), (0.7, True), (0.95, True)]
    best = analyze_module.optimal_cost_weighted_threshold(pairs, cost_ratio=3.0)

    for candidate in [0.0, 0.05, 0.2, 0.3, 0.5, 0.6, 0.7, 0.9, 0.95, 1.0, 2.0]:
        fp = sum(1 for p, y in pairs if p >= candidate and not y)
        fn = sum(1 for p, y in pairs if p < candidate and y)
        assert (3.0 * fn + fp) / len(pairs) >= best["loss"] - 1e-9, candidate


# ----------------------------------------------------------------------
# Calibration honesty
# ----------------------------------------------------------------------


def test_the_calibration_warning_is_emitted_whenever_two_arms_are_rendered() -> None:
    """The LLM arm's number is a verbalised confidence, not a logprob.

    Reliability diagrams of the two arms side by side look like a calibration
    comparison and are not one. The warning is the only thing standing between
    that layout and a false claim, so it is a tested behaviour, not a comment.
    """

    report = analyze_module.analyse(_paired_rows([0.9, 0.1, 0.8], [0.9, 0.9, 0.2]))

    assert "calibration_warning" in report
    assert any("verbalised" in w for w in report["warnings"])
    assert "verbalised" in analyze_module.format_report(report)
    assert "logprob" in report["calibration_warning"]


def test_a_single_arm_gets_no_cross_arm_calibration_warning() -> None:
    """Nothing is being compared, so the warning would be noise."""

    rows = [_row(ARM_JEV, f"c{i}", 0, 0.5, source=run_module.PROB_SOURCE_NOUL) for i in range(4)]

    report = analyze_module.analyse(rows)

    assert "calibration_warning" not in report


def test_a_logprob_arm_is_the_documented_way_out_of_the_warning() -> None:
    """The hook has to work, or it is decoration.

    A future arm that reads token logprobs sets ``probability_source`` and the
    verbalised-confidence caveat stops applying to it.
    """

    rows = [
        _row(ARM_JEV, f"c{i}", 0, 0.5 + i / 20, source=run_module.PROB_SOURCE_NOUL) for i in range(4)
    ] + [
        _row(ARM_LLM, f"c{i}", 0, 0.4 + i / 20, source=run_module.PROB_SOURCE_LOGPROB)
        for i in range(4)
    ]

    report = analyze_module.analyse(rows)

    assert report["arms"][ARM_LLM]["probability_source"] == run_module.PROB_SOURCE_LOGPROB
    assert "calibration_warning" not in report


def test_reliability_bins_are_quantile_bins_with_intervals_when_labelled() -> None:
    """Fixed-width bins would leave a verbalised arm in one bucket."""

    pairs: List[Tuple[float, Optional[bool]]] = [(0.9, True)] * 8 + [(0.1, False)] * 2
    table = analyze_module.reliability_table(pairs, bins=2)

    assert [entry["n"] for entry in table] == [5, 5], "bins must be equal-count, not equal-width"
    for entry in table:
        assert entry["wilson_ci95"] is not None
        low, high = entry["wilson_ci95"]
        assert low <= entry["empirical_rate"] <= high


# ----------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------


def test_the_adapter_is_not_a_hard_dependency_at_import_time() -> None:
    """The adapter repo is one commit old and zero stars. It may not be there.

    Importing it at module scope would make the entire eval -- including the
    OpenRouter arm, which needs nothing extra -- fail to import when it isn't.
    """

    source = Path(run_module.__file__).read_text(encoding="utf-8")
    module_level = [
        line for line in source.splitlines()
        if line.startswith("import ") or line.startswith("from ")
    ]
    assert not any("system_one_adapter" in line for line in module_level)
    assert "pin a commit" in run_module.ADAPTER_IMPORT_HINT.lower()


def test_the_openrouter_tool_schema_is_derived_from_the_question_dicts() -> None:
    """A second hand-written copy of the wording is a second thing to drift."""

    questions = run_module.questions_for_trial()
    schema = run_module.tool_schema(questions)

    parameters = schema[0]["function"]["parameters"]
    assert set(parameters["properties"]) == set(questions)
    assert set(parameters["required"]) == set(questions)
    for name, question in questions.items():
        assert parameters["properties"][name]["description"] == question["instructions"]


def test_the_openrouter_backend_reads_a_probability_out_of_the_tool_call() -> None:
    questions = run_module.questions_for_trial()
    payload = {
        "model": "test/model",
        "usage": {"prompt_tokens": 321, "completion_tokens": 7},
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "answer_questions",
                                "arguments": json.dumps(
                                    {name: 0.8 for name in questions} | {"important": 1.7}
                                ),
                            }
                        }
                    ]
                }
            }
        ],
    }

    parsed = run_module.OpenRouterBackend._parse(payload, questions)

    assert parsed.usage.input_tokens == 321
    assert parsed.answers["important"].noul == 1.0, "a probability must be clamped to [0, 1]"
    assert parsed.answers["automated_bulk"].noul == pytest.approx(0.8)

    with pytest.raises(ValueError):
        run_module.OpenRouterBackend._parse(
            {"choices": [{"message": {"content": "not json"}}]}, questions
        )


def test_backends_declare_where_their_probabilities_come_from() -> None:
    assert run_module.JevBackend.probability_source == run_module.PROB_SOURCE_NOUL
    assert run_module.AdapterBackend.probability_source == run_module.PROB_SOURCE_VERBALISED
    assert run_module.OpenRouterBackend.probability_source == run_module.PROB_SOURCE_VERBALISED


# ----------------------------------------------------------------------
# End to end
# ----------------------------------------------------------------------


async def test_a_full_offline_run_analyses_into_a_paired_report(tmp_path: Path) -> None:
    backends = _backends()
    await _run(tmp_path, backends, repeats=2)

    report = analyze_module.analyse(analyze_module.load(tmp_path))

    assert set(report["arms"]) == {ARM_JEV, ARM_LLM}
    assert list(report["arms"])[0] == ARM_JEV, "jev reads first: it is the incumbent"
    assert report["paired"]["paired_rows"] == report["arms"][ARM_JEV]["n"]
    assert report["paired"]["unpaired_rows_excluded"] == 0
    assert report["arms"][ARM_JEV]["estimated_cost_usd"] < report["arms"][ARM_LLM]["estimated_cost_usd"]
    assert report["arms"][ARM_JEV]["backends"] == ["typesafe_sdk:jev-1.13.0"]
