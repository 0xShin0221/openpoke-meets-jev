"""Analyse a Jev-versus-LLM A/B run.

    python -m evals.ab_llm.analyze --results evals/ab_llm/results --cost-ratio 5

What it reports, per arm: accuracy / precision / recall (only when a labels file
exists), latency mean and p50/p95, input tokens and estimated cost, a reliability
table over quantile bins with Wilson intervals, and Brier score. Across arms:
agreement, an **exact McNemar** on the paired disagreements, and cost-weighted
loss with each arm free to pick its own optimal threshold, so neither arm is
handicapped by the other's operating point.

Read the calibration warning this prints before comparing the two reliability
tables. It is not a caveat, it is the reason the comparison is not fair:

    Jev's noul is a probability the model was trained to emit. The LLM arm's
    "probability" is a number the model was asked to say out loud inside a
    structured output -- a *verbalised* confidence. Verbalised confidences are
    known to be poorly calibrated and to pile up on round numbers (0.9, 0.95,
    0.8). A reliability diagram of one against the other is comparing two
    different quantities.

The honest fix is a logprob-based arm. The hook is already in place: a
``Backend`` sets ``probability_source = PROB_SOURCE_LOGPROB`` and its rows are
excluded from the warning. Nothing else in this file needs to change.

All statistics come from ``evals.contamination.stats``; none are reimplemented.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.jev import thresholds as t  # noqa: E402

from evals.contamination.stats import mcnemar_exact, wilson  # noqa: E402

from .run import (  # noqa: E402
    ARM_JEV,
    ARM_LLM,
    PROB_SOURCE_LOGPROB,
    PROB_SOURCE_VERBALISED,
)

# The bar the server actually surfaces on. Used for agreement and for the
# labelled accuracy table; the cost-weighted comparison ignores it on purpose.
SURFACE_BAR = t.EMAIL_IMPORTANT_HIGH

#: The question this A/B is about. The other three ride along in the same
#: request and are recorded, but only this one replaced an LLM decision.
TARGET_QUESTION = "important"

CALIBRATION_WARNING = (
    "CALIBRATION WARNING: the reliability tables below are NOT comparable. "
    "The LLM arm's probabilities are verbalised confidences read out of a "
    "structured output, not logprobs; they are known to be poorly calibrated "
    "and quantised to round numbers. Jev's noul is a different quantity. Read "
    "each arm's reliability table on its own, and do not report one arm as "
    "'better calibrated' than the other from these numbers. A logprob-based "
    "arm would make this comparison meaningful; see Backend.probability_source."
)

NO_LABELS_NOTE = (
    "No labels file was supplied, so no accuracy, precision, recall, Brier or "
    "cost-weighted comparison is reported. What is reported is agreement "
    "between the arms, latency, cost, and the SHAPE of each arm's probability "
    "distribution. Agreement is not accuracy: two arms can agree and both be "
    "wrong. Labels were not invented to fill these fields."
)


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------


def load(results_dir: Path) -> List[Dict[str, Any]]:
    """Read the cached responses, dropping failed calls.

    A row with an ``error`` is a call that did not happen. It is kept on disk so
    the run stays resumable and the failure is countable, but it never becomes a
    data point here.
    """

    path = results_dir / "responses.jsonl"
    if not path.is_file():
        raise SystemExit(f"no responses at {path}; run the experiment first")
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "error" in record:
            continue
        if not isinstance(record.get("answers"), dict):
            continue
        if record["answers"].get(TARGET_QUESTION) is None:
            continue
        rows.append(record)
    return rows


def load_labels(path: Optional[Path]) -> Optional[Dict[str, bool]]:
    """Read an optional carrier-id -> "important"/"not_important" mapping.

    Returns ``None`` when the file is absent, which is different from an empty
    mapping: absent means "report no accuracy", not "nothing is important".
    Carriers whose label is neither value (e.g. a "borderline" annotation) are
    dropped rather than guessed at.
    """

    if path is None or not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    labels: Dict[str, bool] = {}
    for carrier_id, value in raw.items():
        text = str(value).strip().lower()
        if text == "important":
            labels[carrier_id] = True
        elif text == "not_important":
            labels[carrier_id] = False
    return labels


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    """Nearest-rank percentile. ``fraction`` is in [0, 1]."""

    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def reliability_table(
    pairs: Sequence[Tuple[float, Optional[bool]]], *, bins: int
) -> List[Dict[str, Any]]:
    """Quantile-binned reliability, with Wilson intervals when labels exist.

    Quantile bins rather than fixed-width ones because a verbalised confidence
    piles up on a handful of round numbers: fixed-width bins would leave most of
    the range empty and put almost every row in one bucket.
    """

    if not pairs:
        return []
    ordered = sorted(pairs, key=lambda item: item[0])
    total = len(ordered)
    table: List[Dict[str, Any]] = []
    for index in range(bins):
        start = index * total // bins
        stop = (index + 1) * total // bins
        chunk = ordered[start:stop]
        if not chunk:
            continue
        probabilities = [p for p, _ in chunk]
        labelled = [y for _, y in chunk if y is not None]
        entry: Dict[str, Any] = {
            "bin": index,
            "n": len(chunk),
            "predicted_range": [round(min(probabilities), 4), round(max(probabilities), 4)],
            "mean_predicted": round(sum(probabilities) / len(probabilities), 4),
            "empirical_rate": None,
            "wilson_ci95": None,
        }
        if labelled:
            successes = sum(1 for y in labelled if y)
            entry["empirical_rate"] = round(successes / len(labelled), 4)
            entry["wilson_ci95"] = [round(v, 4) for v in wilson(successes, len(labelled))]
            entry["labelled_n"] = len(labelled)
        table.append(entry)
    return table


def optimal_cost_weighted_threshold(
    pairs: Sequence[Tuple[float, bool]], *, cost_ratio: float
) -> Optional[Dict[str, Any]]:
    """Sweep every threshold and return the one minimising cost-weighted loss.

    ``cost_ratio`` is the cost of a false negative (missing mail the recipient
    needed) relative to a false positive (an unnecessary interrupt), which is
    the asymmetry the server's threshold actually encodes. Each arm gets its own
    optimum: comparing two arms at one shared cut point measures the cut point
    as much as it measures the arms.
    """

    if not pairs:
        return None
    candidates = sorted({p for p, _ in pairs})
    # One threshold above every observation, i.e. "surface nothing".
    candidates.append(max(candidates) + 1.0)
    best: Optional[Dict[str, Any]] = None
    for threshold in candidates:
        false_positives = sum(1 for p, y in pairs if p >= threshold and not y)
        false_negatives = sum(1 for p, y in pairs if p < threshold and y)
        loss = (cost_ratio * false_negatives + false_positives) / len(pairs)
        if best is None or loss < best["loss"]:
            best = {
                "threshold": round(threshold, 4),
                "loss": loss,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
            }
    assert best is not None
    best["loss"] = round(best["loss"], 5)
    return best


def _confusion(pairs: Sequence[Tuple[float, bool]], *, threshold: float) -> Dict[str, Any]:
    tp = sum(1 for p, y in pairs if p >= threshold and y)
    fp = sum(1 for p, y in pairs if p >= threshold and not y)
    fn = sum(1 for p, y in pairs if p < threshold and y)
    tn = sum(1 for p, y in pairs if p < threshold and not y)
    total = tp + fp + fn + tn
    return {
        "threshold": threshold,
        "n": total,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "accuracy": round((tp + tn) / total, 4) if total else None,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
    }


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------


def analyse(
    rows: Sequence[Mapping[str, Any]],
    *,
    labels: Optional[Mapping[str, bool]] = None,
    bins: int = 5,
    cost_ratio: float = 5.0,
) -> Dict[str, Any]:
    """Return the full A/B report."""

    by_arm: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[str(row["arm"])].append(row)

    # jev first when present, so the report reads "jev vs the thing it replaced".
    arms = sorted(by_arm, key=lambda arm: (arm != ARM_JEV, arm))
    has_labels = labels is not None

    report: Dict[str, Any] = {
        "target_question": TARGET_QUESTION,
        "surface_bar": SURFACE_BAR,
        "cost_ratio_fn_over_fp": cost_ratio,
        "labels_available": has_labels,
        "rows_analysed": len(rows),
        "warnings": [],
        "arms": {},
    }
    if not has_labels:
        report["warnings"].append(NO_LABELS_NOTE)

    probability_sources: Dict[str, str] = {}

    for arm in arms:
        arm_rows = by_arm[arm]
        probabilities = [float(r["answers"][TARGET_QUESTION]) for r in arm_rows]
        latencies = [float(r.get("latency_ms") or 0.0) for r in arm_rows]
        tokens = sum(int(r.get("input_tokens") or 0) for r in arm_rows)
        cost = sum(
            int(r.get("input_tokens") or 0) * float(r.get("input_cost_per_million") or 0.0)
            for r in arm_rows
        ) / 1_000_000
        source = str(arm_rows[0].get("probability_source") or PROB_SOURCE_VERBALISED)
        probability_sources[arm] = source

        labelled_pairs: List[Tuple[float, bool]] = []
        if labels is not None:
            labelled_pairs = [
                (float(r["answers"][TARGET_QUESTION]), bool(labels[r["carrier_id"]]))
                for r in arm_rows
                if r["carrier_id"] in labels
            ]

        entry: Dict[str, Any] = {
            "backends": sorted({str(r.get("backend") or "") for r in arm_rows}),
            "probability_source": source,
            "n": len(arm_rows),
            "latency_ms": {
                "mean": round(sum(latencies) / len(latencies), 1) if latencies else None,
                "p50": percentile(latencies, 0.50),
                "p95": percentile(latencies, 0.95),
            },
            "input_tokens": tokens,
            "estimated_cost_usd": round(cost, 8),
            "surfaced_at_bar": sum(1 for p in probabilities if p >= SURFACE_BAR),
            "mean_probability": round(sum(probabilities) / len(probabilities), 4),
            "distinct_probabilities": len(set(probabilities)),
            "reliability": reliability_table(
                [
                    (float(r["answers"][TARGET_QUESTION]),
                     labels.get(r["carrier_id"]) if labels is not None else None)
                    for r in arm_rows
                ],
                bins=bins,
            ),
            "accuracy": None,
            "brier": None,
            "cost_weighted_optimum": None,
        }
        if labelled_pairs:
            entry["labelled_n"] = len(labelled_pairs)
            entry["accuracy"] = _confusion(labelled_pairs, threshold=SURFACE_BAR)
            entry["brier"] = round(
                sum((p - (1.0 if y else 0.0)) ** 2 for p, y in labelled_pairs)
                / len(labelled_pairs),
                5,
            )
            entry["cost_weighted_optimum"] = optimal_cost_weighted_threshold(
                labelled_pairs, cost_ratio=cost_ratio
            )
        report["arms"][arm] = entry

    # A calibration comparison is only rendered when two or more arms have a
    # reliability table. Warn exactly then.
    comparable = [arm for arm in arms if report["arms"][arm]["reliability"]]
    if len(comparable) >= 2:
        sources = {probability_sources[arm] for arm in comparable}
        # The caveat is specifically about verbalised confidence. An arm reading
        # logprobs (PROB_SOURCE_LOGPROB) does not attract it -- that is the
        # documented way out, and the reason the field exists.
        if PROB_SOURCE_VERBALISED in sources:
            report["warnings"].append(CALIBRATION_WARNING)
            report["calibration_warning"] = CALIBRATION_WARNING

    report["paired"] = _paired(by_arm, arms, labels=labels, cost_ratio=cost_ratio, report=report)
    return report


def _paired(
    by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    arms: Sequence[str],
    *,
    labels: Optional[Mapping[str, bool]],
    cost_ratio: float,
    report: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Compare two arms on the rows they both produced.

    A row present in one arm and missing from the other (a failed call, a
    partial run, a cache from a different backend) carries no paired
    information and is excluded by construction, not by filtering afterwards.
    """

    if len(arms) < 2:
        return None
    left, right = arms[0], arms[1]

    def index(arm: str) -> Dict[Tuple[str, int], float]:
        out: Dict[Tuple[str, int], float] = {}
        for row in by_arm[arm]:
            out[(str(row["carrier_id"]), int(row["repeat"]))] = float(
                row["answers"][TARGET_QUESTION]
            )
        return out

    left_index, right_index = index(left), index(right)
    keys = sorted(set(left_index) & set(right_index))
    dropped = len(set(left_index) ^ set(right_index))

    left_surface = {k: left_index[k] >= SURFACE_BAR for k in keys}
    right_surface = {k: right_index[k] >= SURFACE_BAR for k in keys}

    agree = sum(1 for k in keys if left_surface[k] == right_surface[k])
    only_left = sum(1 for k in keys if left_surface[k] and not right_surface[k])
    only_right = sum(1 for k in keys if right_surface[k] and not left_surface[k])

    paired: Dict[str, Any] = {
        "arms": [left, right],
        "paired_rows": len(keys),
        "unpaired_rows_excluded": dropped,
        "agreement": round(agree / len(keys), 4) if keys else None,
        "agreement_wilson_ci95": [round(v, 4) for v in wilson(agree, len(keys))] if keys else None,
        f"surfaced_only_by_{left}": only_left,
        f"surfaced_only_by_{right}": only_right,
        "mcnemar_exact_p_surfacing": round(mcnemar_exact(only_left, only_right), 5),
        "reading": (
            "The surfacing McNemar asks whether one arm surfaces more mail than "
            "the other at the same bar. It is computed on discordant PAIRS only; "
            "concordant pairs and rows missing from either arm carry no "
            "information. A significant result is a difference in operating "
            "point, not in quality -- without labels neither arm is known to be "
            "right."
        ),
        "mcnemar_exact_p_accuracy": None,
        "cost_weighted": None,
    }

    if labels is None:
        return paired

    labelled_keys = [k for k in keys if k[0] in labels]
    if not labelled_keys:
        return paired

    left_correct = {
        k: (left_index[k] >= SURFACE_BAR) == bool(labels[k[0]]) for k in labelled_keys
    }
    right_correct = {
        k: (right_index[k] >= SURFACE_BAR) == bool(labels[k[0]]) for k in labelled_keys
    }
    left_only_correct = sum(1 for k in labelled_keys if left_correct[k] and not right_correct[k])
    right_only_correct = sum(1 for k in labelled_keys if right_correct[k] and not left_correct[k])
    paired["labelled_paired_rows"] = len(labelled_keys)
    paired[f"correct_only_{left}"] = left_only_correct
    paired[f"correct_only_{right}"] = right_only_correct
    paired["mcnemar_exact_p_accuracy"] = round(
        mcnemar_exact(left_only_correct, right_only_correct), 5
    )

    # Cost-weighted loss with each arm at its OWN optimum, on the paired rows.
    def own_optimum(index_map: Mapping[Tuple[str, int], float]) -> Optional[Dict[str, Any]]:
        return optimal_cost_weighted_threshold(
            [(index_map[k], bool(labels[k[0]])) for k in labelled_keys], cost_ratio=cost_ratio
        )

    left_best = own_optimum(left_index)
    right_best = own_optimum(right_index)
    if left_best and right_best:
        paired["cost_weighted"] = {
            "cost_ratio_fn_over_fp": cost_ratio,
            left: left_best,
            right: right_best,
            "loss_delta": round(left_best["loss"] - right_best["loss"], 5),
            "better": left if left_best["loss"] < right_best["loss"] else (
                right if right_best["loss"] < left_best["loss"] else None
            ),
            "reading": (
                "Each arm picks the threshold that minimises its own "
                "cost-weighted loss, so neither is handicapped by the other's "
                "operating point. These optima are fitted on the same rows they "
                "are scored on; with a set this small treat the delta as a "
                "direction, not an estimate."
            ),
        }
    return paired


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------


def format_report(report: Mapping[str, Any]) -> str:
    """Render the report as text, warnings first."""

    lines: List[str] = []
    for warning in report.get("warnings", []):
        lines.append("!! " + warning)
        lines.append("")
    lines.append(json.dumps(report, indent=2))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("evals/ab_llm/results"))
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="JSON mapping carrier id -> important/not_important. Optional.",
    )
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument(
        "--cost-ratio",
        type=float,
        default=5.0,
        help="Cost of a false negative relative to a false positive.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    labels_path = args.labels
    if labels_path is None:
        default_labels = args.results / "labels.json"
        labels_path = default_labels if default_labels.is_file() else None

    report = analyse(
        load(args.results),
        labels=load_labels(labels_path),
        bins=args.bins,
        cost_ratio=args.cost_ratio,
    )
    text = format_report(report)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(text)
    for warning in report.get("warnings", []):
        print(warning, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_JEV",
    "ARM_LLM",
    "CALIBRATION_WARNING",
    "NO_LABELS_NOTE",
    "SURFACE_BAR",
    "TARGET_QUESTION",
    "analyse",
    "format_report",
    "load",
    "load_labels",
    "optimal_cost_weighted_threshold",
    "reliability_table",
]
