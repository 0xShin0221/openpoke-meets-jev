"""Metrics for a binary gate whose two error types cost different amounts.

The primary metric here is **cost-weighted expected loss**, because it is the
only one that answers "what threshold should we ship". Everything else is there
to say how much to trust it.

Nothing in this module is prescribed by TypeSafe's documentation — the words
precision, recall, Brier and calibration appear nowhere in it. The one piece of
official guidance is "test thresholds by plotting confidence against accuracy on
your data" (https://docs.typesafe.ai/concepts/how-to-build-with-system-one),
which is a reliability diagram in plain language, and that is what
:func:`reliability` builds.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.contamination.stats import wilson  # noqa: E402


# ----------------------------------------------------------------------
# Agreement between annotators
# ----------------------------------------------------------------------


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Return Cohen's kappa for two annotators over the same items.

    Our own label noise bounds every accuracy figure downstream, so this is
    reported before any of them. The only published importance corpus (ACL 2015,
    Enron, 3-level) reached 0.64-0.80 by annotator tier; that is the realistic
    target and a fair bar to be judged against.
    """

    if len(a) != len(b) or not a:
        raise ValueError("annotator sequences must be the same non-zero length")

    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    labels = set(a) | set(b)
    expected = sum(
        (sum(1 for x in a if x == label) / n) * (sum(1 for y in b if y == label) / n)
        for label in labels
    )
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def krippendorff_alpha_binary(ratings: Sequence[Sequence[Optional[int]]]) -> float:
    """Return Krippendorff's alpha for binary, possibly incomplete, ratings.

    ``ratings`` is one row per item, one column per annotator, ``None`` where an
    annotator did not see the item. Handles the missing data that a real
    annotation pass produces, which Cohen's kappa cannot.
    """

    pairable: List[List[int]] = [
        [value for value in row if value is not None] for row in ratings
    ]
    usable = [row for row in pairable if len(row) >= 2]
    if not usable:
        return float("nan")

    total_pairs = 0.0
    disagreements = 0.0
    counts: Dict[int, int] = {}
    for row in usable:
        m = len(row)
        for i in range(m):
            counts[row[i]] = counts.get(row[i], 0) + 1
            for j in range(m):
                if i == j:
                    continue
                total_pairs += 1 / (m - 1)
                if row[i] != row[j]:
                    disagreements += 1 / (m - 1)

    if total_pairs == 0:
        return float("nan")
    observed = disagreements / total_pairs

    n = sum(counts.values())
    if n <= 1:
        return float("nan")
    expected = 1 - sum(count * (count - 1) for count in counts.values()) / (n * (n - 1))
    if expected == 0:
        return 1.0
    return 1 - observed / expected


# ----------------------------------------------------------------------
# Scoring a threshold
# ----------------------------------------------------------------------


def confusion(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> Dict[str, int]:
    """Return tp/fp/tn/fn at ``threshold`` (predict positive when score >= t)."""

    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores):
        predicted = score >= threshold
        if label == 1 and predicted:
            tp += 1
        elif label == 0 and predicted:
            fp += 1
        elif label == 0:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision_recall(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> Tuple[float, float]:
    """Return (precision, recall) at ``threshold``."""

    matrix = confusion(labels, scores, threshold)
    predicted_positive = matrix["tp"] + matrix["fp"]
    actual_positive = matrix["tp"] + matrix["fn"]
    precision = matrix["tp"] / predicted_positive if predicted_positive else 0.0
    recall = matrix["tp"] / actual_positive if actual_positive else 0.0
    return precision, recall


def cost_weighted_loss(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
    *,
    cost_ratio: float,
) -> float:
    """Return mean cost per item, with a missed positive costing ``cost_ratio``.

    ``cost_ratio`` is an exchange rate, not money: "I would accept 20 extra
    interruptions a week to catch one more important email" is 20. On a
    *calibrated* probability the theoretically optimal cut is
    ``1 / (1 + cost_ratio)`` — for 20:1 that is about 0.048, far below the 0.75
    this repo currently ships, which is worth knowing before defending 0.75. The
    formula is only as good as the calibration, which is why calibration is
    measured rather than assumed.
    """

    if not labels:
        return 0.0
    matrix = confusion(labels, scores, threshold)
    return (matrix["fp"] * 1.0 + matrix["fn"] * cost_ratio) / len(labels)


def theoretical_optimum(cost_ratio: float) -> float:
    """Return the cut point a perfectly calibrated probability would imply."""

    return 1.0 / (1.0 + cost_ratio)


def brier(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Return the Brier score (lower is better)."""

    if not labels:
        return float("nan")
    return sum((score - label) ** 2 for label, score in zip(labels, scores)) / len(labels)


def brier_baseline(labels: Sequence[int]) -> float:
    """Return the Brier score of always predicting the base rate.

    Reported alongside :func:`brier`, because the absolute value of a Brier
    score is uninterpretable without it.
    """

    if not labels:
        return float("nan")
    rate = sum(labels) / len(labels)
    return sum((rate - label) ** 2 for label in labels) / len(labels)


# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------


def reliability(
    labels: Sequence[int], scores: Sequence[float], *, bins: int = 5
) -> List[Dict[str, float]]:
    """Return a quantile-binned reliability table with per-bin intervals.

    Quantile bins, not equal-width ones: noul outputs pile up near 0 and 1, so
    equal-width binning leaves middle bins with two or three points and error
    bars wider than the plot. Every bin carries its count and a Wilson interval,
    because a reliability point without one is the standard way small-N
    calibration analysis misleads.
    """

    paired = sorted(zip(scores, labels))
    if not paired:
        return []
    bins = max(1, min(bins, len(paired)))
    size = math.ceil(len(paired) / bins)

    table: List[Dict[str, float]] = []
    for start in range(0, len(paired), size):
        chunk = paired[start : start + size]
        if not chunk:
            continue
        observed = sum(label for _, label in chunk)
        low, high = wilson(observed, len(chunk))
        table.append(
            {
                "bin_low": round(chunk[0][0], 4),
                "bin_high": round(chunk[-1][0], 4),
                "n": len(chunk),
                "mean_predicted": round(sum(score for score, _ in chunk) / len(chunk), 4),
                "observed_rate": round(observed / len(chunk), 4),
                "observed_ci95": [round(low, 4), round(high, 4)],
            }
        )
    return table


def expected_calibration_error(
    labels: Sequence[int], scores: Sequence[float], *, bins: int = 5
) -> float:
    """Return ECE over quantile bins.

    Reported with its binning stated and never as a headline: at a couple of
    hundred items ECE is dominated by bias and binning artefacts. Brier is the
    better summary.
    """

    table = reliability(labels, scores, bins=bins)
    if not table:
        return float("nan")
    total = sum(row["n"] for row in table)
    return sum(
        row["n"] / total * abs(row["mean_predicted"] - row["observed_rate"]) for row in table
    )


# ----------------------------------------------------------------------
# Choosing a threshold honestly
# ----------------------------------------------------------------------


def candidate_thresholds(scores: Sequence[float]) -> List[float]:
    """Return the cut points worth evaluating.

    The observed scores plus the midpoints between consecutive distinct ones. A
    cut placed exactly on an observed score is arbitrary — the interval between
    two observations is where the boundary actually lives, and including the
    midpoints is what lets a plateau appear instead of a single spuriously
    precise number.
    """

    observed = sorted({round(score, 4) for score in scores})
    midpoints = [
        round((low + high) / 2, 4) for low, high in zip(observed, observed[1:]) if high > low
    ]
    return sorted(set(observed) | set(midpoints) | {0.0, 1.0})


def best_threshold(
    labels: Sequence[int], scores: Sequence[float], *, cost_ratio: float
) -> Tuple[float, float, Tuple[float, float]]:
    """Return (threshold, loss, plateau) minimising cost-weighted loss.

    The plateau is the range of thresholds within 1% of the minimum. With a few
    hundred items you cannot tell 0.62 from 0.68, so reporting a single tuned
    number to two decimals overstates what the data supports.
    """

    candidates = candidate_thresholds(scores)
    losses = [
        (threshold, cost_weighted_loss(labels, scores, threshold, cost_ratio=cost_ratio))
        for threshold in candidates
    ]
    best = min(losses, key=lambda pair: pair[1])
    tolerance = best[1] * 1.01 + 1e-9
    near = [threshold for threshold, loss in losses if loss <= tolerance]
    return best[0], best[1], (min(near), max(near))


def repeated_stratified_kfold(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    cost_ratio: float,
    folds: int = 5,
    repeats: int = 10,
    seed: int = 20260919,
) -> Dict[str, float]:
    """Pick the threshold on training folds and score it on held-out folds.

    With a few hundred items there is no room for a clean train/test split, and
    tuning and reporting on the same data overstates performance by several
    points. This is the honest alternative, and the number to publish is the
    cross-validated loss, not the in-sample optimum.
    """

    rng = random.Random(seed)
    positives = [i for i, label in enumerate(labels) if label == 1]
    negatives = [i for i, label in enumerate(labels) if label == 0]
    if len(positives) < folds or len(negatives) < folds:
        raise ValueError(
            f"need at least {folds} of each class; have {len(positives)} positive "
            f"and {len(negatives)} negative"
        )

    held_out_losses: List[float] = []
    chosen: List[float] = []

    for _ in range(repeats):
        rng.shuffle(positives)
        rng.shuffle(negatives)
        fold_members: List[List[int]] = [[] for _ in range(folds)]
        for offset, index in enumerate(positives):
            fold_members[offset % folds].append(index)
        for offset, index in enumerate(negatives):
            fold_members[offset % folds].append(index)

        for fold in range(folds):
            test_idx = fold_members[fold]
            train_idx = [i for f in range(folds) if f != fold for i in fold_members[f]]
            if not test_idx or not train_idx:
                continue
            threshold, _, _ = best_threshold(
                [labels[i] for i in train_idx],
                [scores[i] for i in train_idx],
                cost_ratio=cost_ratio,
            )
            chosen.append(threshold)
            held_out_losses.append(
                cost_weighted_loss(
                    [labels[i] for i in test_idx],
                    [scores[i] for i in test_idx],
                    threshold,
                    cost_ratio=cost_ratio,
                )
            )

    if not held_out_losses:
        return {"cv_loss": float("nan"), "threshold_mean": float("nan")}

    mean_threshold = sum(chosen) / len(chosen)
    return {
        "cv_loss": sum(held_out_losses) / len(held_out_losses),
        "cv_loss_sd": (
            sum((loss - sum(held_out_losses) / len(held_out_losses)) ** 2 for loss in held_out_losses)
            / max(1, len(held_out_losses) - 1)
        )
        ** 0.5,
        "threshold_mean": mean_threshold,
        "threshold_min": min(chosen),
        "threshold_max": max(chosen),
        "folds_evaluated": len(held_out_losses),
    }


__all__ = [
    "best_threshold",
    "brier",
    "brier_baseline",
    "candidate_thresholds",
    "cohen_kappa",
    "confusion",
    "cost_weighted_loss",
    "expected_calibration_error",
    "krippendorff_alpha_binary",
    "precision_recall",
    "reliability",
    "repeated_stratified_kfold",
    "theoretical_optimum",
]
