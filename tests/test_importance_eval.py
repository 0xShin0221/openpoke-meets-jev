"""The importance labelling and threshold sweep, without a network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from evals.importance import annotate as annotate_module
from evals.importance import corpus as corpus_module
from evals.importance import sweep as sweep_module
from evals.importance.metrics import (
    best_threshold,
    brier,
    brier_baseline,
    cohen_kappa,
    cost_weighted_loss,
    expected_calibration_error,
    krippendorff_alpha_binary,
    precision_recall,
    reliability,
    repeated_stratified_kfold,
    theoretical_optimum,
)


# ----------------------------------------------------------------------
# The corpus
# ----------------------------------------------------------------------


def test_corpus_is_stratified_with_the_hard_cases_at_half() -> None:
    """A set that is 80% obvious teaches nothing.

    The two hard strata are the ones a keyword classifier gets wrong: urgent
    wording on unimportant mail, and quiet wording on critical mail.
    """

    rows = corpus_module.load()
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["stratum"]] = counts.get(row["stratum"], 0) + 1

    assert set(counts) == set(corpus_module.STRATA)
    hard = counts["urgent_unimportant"] + counts["quiet_critical"] + counts["borderline"]
    assert hard >= len(rows) / 2, "the informative half of the set must not shrink"


def test_corpus_ids_are_unique_across_sources(tmp_path: Path) -> None:
    private = tmp_path / "private.jsonl"
    private.write_text(
        json.dumps({"id": "priv-1", "sender": "a", "subject": "b", "body": "c"}) + "\n",
        encoding="utf-8",
    )

    rows = corpus_module.load(private)

    assert len({row["id"] for row in rows}) == len(rows)
    assert {row["source"] for row in rows} == {"synthetic", "private"}


def test_a_colliding_private_id_is_refused(tmp_path: Path) -> None:
    # Silently overwriting a synthetic row with a private one would put
    # unpublishable content behind a redistributable id.
    private = tmp_path / "private.jsonl"
    first = corpus_module.load()[0]["id"]
    private.write_text(json.dumps({"id": first, "body": "x"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        corpus_module.load(private)


# ----------------------------------------------------------------------
# Annotation
# ----------------------------------------------------------------------


def test_the_annotator_is_never_shown_the_stratum_or_a_score() -> None:
    """Blind by construction: seeing either one anchors the label."""

    row = {
        "id": "x",
        "stratum": "quiet_critical",
        "sender": "a@b.example",
        "subject": "s",
        "body": "body",
        "expected": 1,
    }

    rendered = annotate_module.render(row, 0, 1)

    assert "quiet_critical" in json.dumps(row) and "quiet_critical" not in rendered
    assert "expected" not in rendered
    assert "0." not in rendered, "no probability may reach the annotator"


def test_annotation_loop_records_skips_and_quits(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"id": f"r{i}", "sender": "", "subject": "", "body": ""} for i in range(4)]
    answers = iter(["1", "s", "0", "q"])

    labels = annotate_module.annotate(
        rows, {}, reader=lambda _prompt: next(answers), writer=lambda _text: None
    )

    assert labels == {"r0": 1, "r2": 0}, "a skip must not become a label"


def test_annotation_resumes_without_re_asking(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"id": f"r{i}", "sender": "", "subject": "", "body": ""} for i in range(3)]
    asked: List[str] = []

    def reader(prompt: str) -> str:
        asked.append(prompt)
        return "1"

    labels = annotate_module.annotate(
        rows, {"r0": 0}, reader=reader, writer=lambda _text: None
    )

    assert len(asked) == 2
    assert labels["r0"] == 0, "an existing label must not be overwritten"


def test_agreement_reports_kappa_over_shared_items(tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    annotate_module.save_labels(a, {"x": 1, "y": 0, "z": 1}, {})
    annotate_module.save_labels(b, {"x": 1, "y": 0, "z": 0}, {})

    report = annotate_module.agreement([a, b])

    assert report["items_in_common"] == 3
    assert 0.0 < report["cohen_kappa"] < 1.0
    assert "krippendorff_alpha" in report


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------


def test_kappa_and_alpha_behave_at_the_extremes() -> None:
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0
    assert krippendorff_alpha_binary([[1, 1], [0, 0], [1, 1], [0, 0]]) == pytest.approx(1.0)


def test_alpha_tolerates_missing_ratings() -> None:
    """A real annotation pass is incomplete; kappa cannot handle that, alpha can."""

    value = krippendorff_alpha_binary([[1, 1], [0, None], [1, 1], [None, 0], [0, 0]])

    assert value == value  # not NaN
    assert -1.0 <= value <= 1.0


def test_cost_weighted_loss_prices_a_miss_higher_than_an_interruption() -> None:
    labels = [1, 0]
    # One false negative at a high bar, one false positive at a low bar.
    miss = cost_weighted_loss(labels, [0.1, 0.0], 0.5, cost_ratio=20)
    interrupt = cost_weighted_loss(labels, [1.0, 0.9], 0.5, cost_ratio=20)

    assert miss > interrupt
    assert theoretical_optimum(20) == pytest.approx(0.0476, abs=0.001)


def test_the_optimal_cut_is_far_below_the_shipped_threshold_at_20_to_1() -> None:
    """Worth knowing before defending 0.75 as conservative."""

    from server.jev import thresholds as t

    assert theoretical_optimum(20) < t.EMAIL_IMPORTANT_HIGH / 5


def test_best_threshold_reports_a_plateau_not_a_false_precision() -> None:
    labels = [1] * 10 + [0] * 10
    scores = [0.9] * 10 + [0.1] * 10

    threshold, loss, plateau = best_threshold(labels, scores, cost_ratio=5)

    assert loss == 0.0
    assert plateau[0] < plateau[1], "many cuts separate these perfectly; say so"


def test_cross_validation_is_not_the_in_sample_optimum() -> None:
    """Tuning and reporting on the same items overstates performance."""

    labels = [1] * 20 + [0] * 60
    scores = [0.8 - 0.01 * i for i in range(20)] + [0.4 - 0.004 * i for i in range(60)]

    _, in_sample, _ = best_threshold(labels, scores, cost_ratio=10)
    cv = repeated_stratified_kfold(labels, scores, cost_ratio=10, repeats=3)

    assert cv["folds_evaluated"] > 0
    assert cv["cv_loss"] >= in_sample - 1e-9


def test_cross_validation_refuses_when_a_class_is_too_small() -> None:
    with pytest.raises(ValueError, match="at least"):
        repeated_stratified_kfold([1, 0, 0, 0, 0, 0], [0.9, 0.1, 0.1, 0.1, 0.1, 0.1], cost_ratio=5)


def test_reliability_uses_quantile_bins_with_intervals() -> None:
    # Noul outputs pile up at the ends; equal-width bins would leave middles
    # with two or three points and error bars wider than the plot.
    labels = [1] * 25 + [0] * 25
    scores = [0.95] * 20 + [0.6] * 5 + [0.05] * 25

    table = reliability(labels, scores, bins=5)

    assert len(table) == 5
    counts = [row["n"] for row in table]
    assert max(counts) - min(counts) <= 1, "bins must hold equal counts, not equal widths"
    for row in table:
        assert row["observed_ci95"][0] <= row["observed_rate"] <= row["observed_ci95"][1]


def test_brier_is_reported_against_a_baseline() -> None:
    labels = [1] * 10 + [0] * 90
    good = [0.9] * 10 + [0.05] * 90
    useless = [0.1] * 100

    assert brier(labels, good) < brier_baseline(labels)
    assert brier(labels, useless) >= brier_baseline(labels) - 1e-9


def test_precision_and_recall_are_zero_not_undefined_on_empty_cells() -> None:
    assert precision_recall([0, 0], [0.1, 0.2], 0.9) == (0.0, 0.0)


def test_ece_is_computed_over_the_same_quantile_bins() -> None:
    labels = [1] * 10 + [0] * 10
    perfect = [1.0] * 10 + [0.0] * 10

    assert expected_calibration_error(labels, perfect, bins=2) == pytest.approx(0.0, abs=1e-9)


# ----------------------------------------------------------------------
# The sweep
# ----------------------------------------------------------------------


class _FakeClient:
    """Scores by stratum, with a little per-call drift."""

    def __init__(self) -> None:
        self.calls = 0

    async def system_one(self, *, state: Dict[str, Any], questions: Dict[str, Any]) -> Any:
        self.calls += 1
        subject = str(state.get("subject", ""))
        body = str(state.get("body", ""))
        important = 0.9 if ("deploy" in subject.lower() or "board" in subject.lower()) else 0.2
        if "FINAL NOTICE" in subject or "ACT NOW" in body:
            important = 0.55
        important += 0.01 * (self.calls % 3)
        answers = {
            name: type("A", (), {"noul": important if name == "important" else 0.0})()
            for name in questions
        }
        return type(
            "R",
            (),
            {
                "answers": answers,
                "model": "jev-1.13.0",
                "usage": type("U", (), {"input_tokens": 80, "output_tokens": 0})(),
            },
        )()

    async def aclose(self) -> None:
        return None


async def test_scoring_is_cached_and_resumable(tmp_path: Path) -> None:
    rows = corpus_module.load()[:6]
    client = _FakeClient()
    path = tmp_path / "scores.jsonl"

    first = await sweep_module.score(
        rows=rows, out_path=path, model="jev-1.13.0", repeats=2, concurrency=4, client=client
    )
    calls_after_first = client.calls
    second = await sweep_module.score(
        rows=rows, out_path=path, model="jev-1.13.0", repeats=2, concurrency=4, client=client
    )

    assert first["called"] == 12 and first["cached"] == 0
    assert second["called"] == 0 and second["cached"] == 12
    assert client.calls == calls_after_first, "a resumed run must spend nothing"


async def test_scored_rows_carry_the_model_and_the_state_hash(tmp_path: Path) -> None:
    rows = corpus_module.load()[:3]
    path = tmp_path / "scores.jsonl"

    await sweep_module.score(
        rows=rows, out_path=path, model="jev-1.13.0", repeats=1, concurrency=2,
        client=_FakeClient(),
    )

    records = [json.loads(l) for l in path.read_text().splitlines() if l]
    for record in records:
        assert record["model"] == "jev-1.13.0"
        assert len(record["state_sha256"]) == 64
        assert "body" not in record


async def test_failed_scores_are_recorded_and_excluded(tmp_path: Path) -> None:
    class _Broken:
        async def system_one(self, **_: Any) -> Any:
            raise RuntimeError("boom")

        async def aclose(self) -> None:
            return None

    path = tmp_path / "scores.jsonl"
    summary = await sweep_module.score(
        rows=corpus_module.load()[:3], out_path=path, model="jev-1.13.0", repeats=1,
        concurrency=2, client=_Broken(),
    )

    assert summary["failed"] == 3
    assert sweep_module.load_scores(path) == {}


async def test_analysis_reports_the_shipped_threshold_and_a_cross_validated_one(
    tmp_path: Path,
) -> None:
    rows = corpus_module.load()
    path = tmp_path / "scores.jsonl"
    await sweep_module.score(
        rows=rows, out_path=path, model="jev-1.13.0", repeats=3, concurrency=8,
        client=_FakeClient(),
    )
    labels = {
        row["id"]: (1 if row["stratum"] in ("obvious_important", "quiet_critical") else 0)
        for row in rows
    }

    report = sweep_module.analyse(
        grouped=sweep_module.load_scores(path), labels=labels, rows=rows, cost_ratio=20
    )

    assert report["n"] == len(rows)
    assert report["shipped_threshold"]["value"] == pytest.approx(0.75)
    assert "cost_weighted_loss" in report["shipped_threshold"]
    assert "note" in report["in_sample_best"], "an in-sample optimum must be labelled as such"
    assert "cross_validated" in report
    assert report["limits"], "the limits are part of the result, not a footnote"


async def test_analysis_reports_repeatability_straddle(tmp_path: Path) -> None:
    """A gate whose answer depends on which call you made is not a gate."""

    rows = corpus_module.load()[:4]
    path = tmp_path / "scores.jsonl"

    class _Straddler:
        def __init__(self) -> None:
            self.n = 0

        async def system_one(self, *, state: Dict[str, Any], questions: Dict[str, Any]) -> Any:
            self.n += 1
            value = 0.74 if self.n % 2 else 0.76
            answers = {
                name: type("A", (), {"noul": value if name == "important" else 0.0})()
                for name in questions
            }
            return type("R", (), {"answers": answers, "model": "jev-1.13.0",
                                  "usage": type("U", (), {"input_tokens": 1})()})()

        async def aclose(self) -> None:
            return None

    await sweep_module.score(
        rows=rows, out_path=path, model="jev-1.13.0", repeats=4, concurrency=1,
        client=_Straddler(),
    )
    report = sweep_module.analyse(
        grouped=sweep_module.load_scores(path),
        labels={row["id"]: 1 for row in rows},
        rows=rows,
        cost_ratio=20,
    )

    assert report["repeatability"]["straddling_the_threshold"] == len(rows)


def test_analysis_refuses_without_labels() -> None:
    report = sweep_module.analyse(grouped={}, labels={}, rows=corpus_module.load(), cost_ratio=20)

    assert "error" in report, "an unlabelled sweep must not invent a number"


def test_cross_validation_picks_the_threshold_on_the_training_folds() -> None:
    """On pure noise nothing generalises, so held-out loss must exceed in-sample.

    If the threshold were picked on the test fold the two would be close, which
    is exactly the leak that makes a tuned number look better than it is.
    """

    import random

    rng = random.Random(5)
    labels = [1] * 30 + [0] * 70
    scores = [rng.random() for _ in labels]

    _, in_sample, _ = best_threshold(labels, scores, cost_ratio=10)
    cv = repeated_stratified_kfold(labels, scores, cost_ratio=10, repeats=5)

    assert cv["cv_loss"] > in_sample * 1.1, (
        f"held-out loss {cv['cv_loss']:.4f} is suspiciously close to the "
        f"in-sample optimum {in_sample:.4f}; the threshold may be leaking"
    )


def test_reliability_bins_are_ordered_by_score_whatever_the_input_order() -> None:
    """Bins are quantiles of the score, not slices of the input sequence."""

    import random

    rng = random.Random(11)
    pairs = [(0.95, 1)] * 20 + [(0.55, 1)] * 5 + [(0.05, 0)] * 25
    rng.shuffle(pairs)
    scores = [score for score, _ in pairs]
    labels = [label for _, label in pairs]

    table = reliability(labels, scores, bins=5)

    lows = [row["bin_low"] for row in table]
    assert lows == sorted(lows), "bins must be ordered by score"
    for earlier, later in zip(table, table[1:]):
        assert earlier["bin_high"] <= later["bin_low"] + 1e-9, "bins must not overlap"


def test_a_partially_parsed_score_is_not_a_confident_zero(tmp_path: Path) -> None:
    """A row carrying answers whose `important` is null is a failure, not data.

    Treating it as data would read as the model confidently saying "not
    important", which is the worst possible way to lose a call.
    """

    path = tmp_path / "scores.jsonl"
    path.write_text(
        json.dumps(
            {
                "key": "k",
                "id": "imp-01",
                "repeat": 0,
                "answers": {"important": None, "security_code": 0.1},
                "model": "jev-1.13.0",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert sweep_module.load_scores(path) == {}


async def test_changing_the_model_invalidates_the_cache(tmp_path: Path) -> None:
    """Thresholds are calibrated against one model, so scores are too."""

    rows = corpus_module.load()[:3]
    path = tmp_path / "scores.jsonl"
    client = _FakeClient()

    await sweep_module.score(
        rows=rows, out_path=path, model="jev-1.13.0", repeats=1, concurrency=2, client=client
    )
    after_first = client.calls
    summary = await sweep_module.score(
        rows=rows, out_path=path, model="jev-1.14.0", repeats=1, concurrency=2, client=client
    )

    assert summary["cached"] == 0, "a different model must not reuse the old scores"
    assert client.calls == after_first + 3
