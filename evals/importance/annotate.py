"""Label the corpus, blind to the model's scores.

    python -m evals.importance.annotate --out evals/importance/labels.json
    python -m evals.importance.annotate --out labels-b.json --shuffle-seed 7
    python -m evals.importance.annotate --agreement labels.json labels-b.json

Blind by construction: this tool never reads the scores file and has no way to
display a probability. An annotator who has seen a score for an email should
hand that email to someone else.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.contamination.persona import persona_brief  # noqa: E402

from . import corpus as corpus_module  # noqa: E402
from .metrics import cohen_kappa, krippendorff_alpha_binary  # noqa: E402

PROMPT = "[1] interrupt  [0] don't  [s] skip  [q] save and quit > "


def load_labels(path: Path) -> Dict[str, int]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {key: int(value) for key, value in data.get("labels", {}).items()}


def save_labels(path: Path, labels: Dict[str, int], meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"meta": meta, "labels": labels}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def render(row: Dict[str, Any], index: int, total: int) -> str:
    """Render one email for the annotator. Deliberately shows no stratum."""

    return (
        f"\n--- {index + 1}/{total} ---\n"
        f"From:    {row.get('sender', '')}\n"
        f"Subject: {row.get('subject', '')}\n\n"
        f"{row.get('body', '')}\n"
    )


def annotate(
    rows: Sequence[Dict[str, Any]],
    existing: Dict[str, int],
    *,
    reader,
    writer,
    shuffle_seed: Optional[int] = None,
) -> Dict[str, int]:
    """Run the labelling loop. ``reader`` returns one keystroke per call."""

    pending = [row for row in rows if row["id"] not in existing]
    if shuffle_seed is not None:
        # Order effects are real: a run of obvious items anchors the next
        # borderline one. Shuffling per annotator breaks that correlation.
        random.Random(shuffle_seed).shuffle(pending)

    labels = dict(existing)
    writer(f"Recipient: {persona_brief()}\n")
    for index, row in enumerate(pending):
        writer(render(row, index, len(pending)))
        while True:
            answer = (reader(PROMPT) or "").strip().lower()
            if answer in ("1", "0"):
                labels[row["id"]] = int(answer)
                break
            if answer == "s":
                break
            if answer == "q":
                return labels
            writer("  please answer 1, 0, s or q\n")
    return labels


def agreement(paths: Sequence[Path]) -> Dict[str, Any]:
    """Report inter-annotator agreement across two or more label files."""

    label_sets = [load_labels(path) for path in paths]
    shared = set(label_sets[0])
    for labels in label_sets[1:]:
        shared &= set(labels)
    shared_ids = sorted(shared)

    report: Dict[str, Any] = {
        "annotators": len(label_sets),
        "items_per_annotator": [len(labels) for labels in label_sets],
        "items_in_common": len(shared_ids),
    }
    if len(shared_ids) < 2:
        report["note"] = "not enough shared items to measure agreement"
        return report

    if len(label_sets) == 2:
        report["cohen_kappa"] = round(
            cohen_kappa(
                [label_sets[0][i] for i in shared_ids],
                [label_sets[1][i] for i in shared_ids],
            ),
            4,
        )
    every_id = sorted(set().union(*[set(labels) for labels in label_sets]))
    report["krippendorff_alpha"] = round(
        krippendorff_alpha_binary(
            [[labels.get(item) for labels in label_sets] for item in every_id]
        ),
        4,
    )
    report["reading"] = (
        "Below about 0.7 the rubric is the problem, not the annotator. The only "
        "published email-importance corpus reached 0.64-0.80 by annotator tier."
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("evals/importance/labels.json"))
    parser.add_argument("--private", type=Path, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--annotator", default="unnamed")
    parser.add_argument("--agreement", nargs="+", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.agreement:
        print(json.dumps(agreement(args.agreement), indent=2))
        return 0

    rows = corpus_module.load(args.private)
    labels = annotate(
        rows,
        load_labels(args.out),
        reader=input,
        writer=lambda text: print(text, end=""),
        shuffle_seed=args.shuffle_seed,
    )
    save_labels(
        args.out,
        labels,
        {
            "annotator": args.annotator,
            "corpus_items": len(rows),
            "labelled": len(labels),
            "rubric": "evals/importance/rubric.md",
        },
    )
    print(f"\nsaved {len(labels)} labels to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
