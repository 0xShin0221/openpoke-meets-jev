"""Score the corpus with Jev, then sweep the importance threshold.

    TYPESAFE_API_KEY=... python -m evals.importance.sweep score
    python -m evals.importance.sweep analyze --cost-ratio 20

Two steps on purpose. Scoring costs money and is cached; sweeping is free and
can be re-run at any cost ratio, with any threshold, forever. That separation is
the whole argument for storing probabilities rather than verdicts.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.contamination.persona import persona_brief  # noqa: E402
from evals.contamination.stats import clopper_pearson, wilson  # noqa: E402
from server.jev import questions as q  # noqa: E402
from server.jev import state as jev_state  # noqa: E402
from server.jev import thresholds as t  # noqa: E402

from . import corpus as corpus_module  # noqa: E402
from .annotate import load_labels  # noqa: E402
from .metrics import (  # noqa: E402
    best_threshold,
    brier,
    brier_baseline,
    cohen_kappa,
    confusion,
    cost_weighted_loss,
    expected_calibration_error,
    precision_recall,
    reliability,
    repeated_stratified_kfold,
    theoretical_optimum,
)

DEFAULT_MODEL = "jev-1.13.0"
DEFAULT_COST_RATIO = 20.0


def build_state(row: Dict[str, Any]) -> Dict[str, Any]:
    """Build the request state, with the persona as explicit context."""

    state = jev_state.email_state(
        sender=str(row.get("sender", "")),
        recipient="Rio Tanaka <rio@northgate.example>",
        subject=str(row.get("subject", "")),
        body=str(row.get("body", "")),
    )
    state["recipient_context"] = persona_brief()
    return state


async def score(
    *,
    rows: Sequence[Dict[str, Any]],
    out_path: Path,
    model: str,
    repeats: int,
    concurrency: int,
    client: Any = None,
) -> Dict[str, Any]:
    """Score every row ``repeats`` times, caching each response."""

    from typesafe_sdk import AsyncTypeSafeClient

    cached: Dict[str, Dict[str, Any]] = {}
    if out_path.is_file():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                cached[record["key"]] = record

    owns_client = client is None
    if owns_client:
        api_key = os.environ.get("TYPESAFE_API_KEY")
        if not api_key:
            raise SystemExit("TYPESAFE_API_KEY is not set")
        client = AsyncTypeSafeClient(api_key=api_key, model=model, timeout=20.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    counters = {"cached": 0, "called": 0, "failed": 0}
    tokens = 0
    started = time.time()

    async def one(row: Dict[str, Any], repeat: int) -> None:
        nonlocal tokens
        key = f"{row['id']}:{repeat}:{model}"
        if key in cached:
            counters["cached"] += 1
            return
        state = build_state(row)
        async with semaphore:
            call_started = time.time()
            try:
                response = await client.system_one(state=state, questions=q.EMAIL_QUESTIONS)
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                counters["failed"] += 1
                record = {"key": key, "id": row["id"], "repeat": repeat,
                          "error": f"{type(exc).__name__}: {exc}"}
                with out_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                return
            latency_ms = int((time.time() - call_started) * 1000)

        answers = {}
        for name in q.EMAIL_QUESTIONS:
            answer = getattr(response, "answers", {}).get(name)
            value = getattr(answer, "noul", None)
            answers[name] = float(value) if isinstance(value, (int, float)) else None

        usage = getattr(response, "usage", None)
        row_tokens = getattr(usage, "input_tokens", None) or 0
        tokens += row_tokens
        counters["called"] += 1
        record = {
            "key": key,
            "id": row["id"],
            "stratum": row.get("stratum"),
            "source": row.get("source"),
            "repeat": repeat,
            "answers": answers,
            "model": getattr(response, "model", model),
            "input_tokens": row_tokens,
            "latency_ms": latency_ms,
            "state_sha256": hashlib.sha256(
                json.dumps(state, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
        }
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(row, repeat) for row in rows for repeat in range(repeats)))
    if owns_client:
        await client.aclose()

    return {
        **counters,
        "input_tokens": tokens,
        "estimated_cost_usd": round(tokens * 0.042 / 1_000_000, 4),
        "wall_seconds": round(time.time() - started, 1),
        "model": model,
        "repeats": repeats,
    }


def load_scores(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Return successful score records grouped by corpus id."""

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    if not path.is_file():
        return grouped
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "answers" not in record:
            continue
        if record["answers"].get("important") is None:
            continue
        grouped.setdefault(record["id"], []).append(record)
    return grouped


def repeatability(grouped: Dict[str, List[Dict[str, Any]]], threshold: float) -> Dict[str, Any]:
    """Report how often repeated calls on identical state straddle the bar.

    TypeSafe's own consistency cookbook concedes this failure: an estimate
    ranging 0.43-0.53 across identical inputs, straddling the decision line.
    Items that flip are the ones to hand-inspect; a gate whose answer depends on
    which call you made is not a gate.
    """

    straddled: List[str] = []
    spreads: List[float] = []
    for item_id, records in grouped.items():
        values = [r["answers"]["important"] for r in records]
        if len(values) < 2:
            continue
        spreads.append(max(values) - min(values))
        if min(values) < threshold <= max(values):
            straddled.append(item_id)
    return {
        "items_with_repeats": len(spreads),
        "straddling_the_threshold": len(straddled),
        "straddling_ids": sorted(straddled),
        "max_spread": round(max(spreads), 4) if spreads else None,
        "median_spread": round(statistics.median(spreads), 4) if spreads else None,
    }


def analyse(
    *,
    grouped: Dict[str, List[Dict[str, Any]]],
    labels: Dict[str, int],
    rows: Sequence[Dict[str, Any]],
    cost_ratio: float,
) -> Dict[str, Any]:
    """Sweep the threshold over stored scores and report honestly."""

    by_id = corpus_module.by_id(rows)
    ids = sorted(set(grouped) & set(labels))
    if not ids:
        return {"error": "no scored and labelled items in common; score and label first"}

    scores = [statistics.mean(r["answers"]["important"] for r in grouped[i]) for i in ids]
    ys = [labels[i] for i in ids]
    positives = sum(ys)

    shipped = t.EMAIL_IMPORTANT_HIGH
    chosen, chosen_loss, plateau = best_threshold(ys, scores, cost_ratio=cost_ratio)
    precision_at_shipped, recall_at_shipped = precision_recall(ys, scores, shipped)
    precision_at_best, recall_at_best = precision_recall(ys, scores, chosen)

    report: Dict[str, Any] = {
        "n": len(ids),
        "positives": positives,
        "base_rate": round(positives / len(ids), 4),
        "cost_ratio": cost_ratio,
        "model": grouped[ids[0]][0].get("model"),
        "shipped_threshold": {
            "value": shipped,
            "confusion": confusion(ys, scores, shipped),
            "precision": round(precision_at_shipped, 4),
            "recall": round(recall_at_shipped, 4),
            "cost_weighted_loss": round(
                cost_weighted_loss(ys, scores, shipped, cost_ratio=cost_ratio), 4
            ),
        },
        "in_sample_best": {
            "threshold": chosen,
            "plateau": [plateau[0], plateau[1]],
            "precision": round(precision_at_best, 4),
            "recall": round(recall_at_best, 4),
            "cost_weighted_loss": round(chosen_loss, 4),
            "note": (
                "In-sample. Tuning and reporting on the same items overstates "
                "performance; the number to quote is the cross-validated loss below."
            ),
        },
        "theoretical_optimum_if_calibrated": round(theoretical_optimum(cost_ratio), 4),
        "calibration": {
            "brier": round(brier(ys, scores), 4),
            "brier_base_rate_baseline": round(brier_baseline(ys), 4),
            "ece_quantile_bins_5": round(expected_calibration_error(ys, scores, bins=5), 4),
            "reliability": reliability(ys, scores, bins=5),
            "note": (
                "Quantile bins with per-bin Wilson intervals. ECE at this N is "
                "dominated by binning; Brier against the base-rate baseline is "
                "the summary to read."
            ),
        },
        "recall_interval_95": [
            round(v, 4) for v in clopper_pearson(int(round(recall_at_shipped * positives)), positives)
        ]
        if positives
        else None,
        "by_stratum": {},
        "repeatability": repeatability(grouped, shipped),
    }

    try:
        report["cross_validated"] = {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in repeated_stratified_kfold(ys, scores, cost_ratio=cost_ratio).items()
        }
    except ValueError as exc:
        report["cross_validated"] = {"unavailable": str(exc)}

    for stratum in corpus_module.STRATA:
        stratum_ids = [i for i in ids if by_id.get(i, {}).get("stratum") == stratum]
        if not stratum_ids:
            continue
        stratum_scores = [scores[ids.index(i)] for i in stratum_ids]
        stratum_ys = [labels[i] for i in stratum_ids]
        surfaced = sum(1 for s in stratum_scores if s >= shipped)
        low, high = wilson(surfaced, len(stratum_ids))
        report["by_stratum"][stratum] = {
            "n": len(stratum_ids),
            "labelled_positive": sum(stratum_ys),
            "surfaced_at_shipped_threshold": surfaced,
            "surfaced_ci95": [round(low, 4), round(high, 4)],
            "mean_score": round(statistics.mean(stratum_scores), 4),
        }

    report["limits"] = [
        f"n={len(ids)} with {positives} positives. Recall on the rare class is the "
        "binding constraint and its interval is wide; see recall_interval_95.",
        "Per-stratum cells are small. Read them as direction, not as measurement.",
        "Synthetic carriers measure a mechanism, not a mailbox. Numbers from the "
        "private source are reported separately and do not transfer.",
    ]
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("score", "analyze"))
    parser.add_argument("--scores", type=Path, default=Path("evals/importance/scores.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("evals/importance/labels.json"))
    parser.add_argument("--private", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Pinned, never an alias.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--cost-ratio", type=float, default=DEFAULT_COST_RATIO)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = corpus_module.load(args.private)

    if args.command == "score":
        summary = asyncio.run(
            score(
                rows=rows,
                out_path=args.scores,
                model=args.model,
                repeats=args.repeats,
                concurrency=args.concurrency,
            )
        )
        print(json.dumps(summary, indent=2))
        return 0

    report = analyse(
        grouped=load_scores(args.scores),
        labels=load_labels(args.labels),
        rows=rows,
        cost_ratio=args.cost_ratio,
    )
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
