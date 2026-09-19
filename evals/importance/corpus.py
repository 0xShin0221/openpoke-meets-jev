"""Loading the importance corpus.

Two sources, deliberately separate:

* ``corpus.jsonl`` — 40 synthetic emails written against the published persona,
  stratified so that the hard cases are half the set: urgent-sounding but
  unimportant, and quiet but critical. Redistributable, because we wrote it.
* A private file of your own mail, which is not redistributable and is never
  committed. ``--private`` points at it and the analysis reports the two sources
  separately, because a number from one does not transfer to the other.

The synthetic set is a *mechanism* check, not a mailbox. 40 items supports a
coarse reliability diagram and nothing finer; see README.md for what that rules
out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

HERE = Path(__file__).resolve().parent
SYNTHETIC_PATH = HERE / "corpus.jsonl"

STRATA = (
    "obvious_important",
    "obvious_unimportant",
    "urgent_unimportant",
    "quiet_critical",
    "borderline",
)

# The strata whose name already states the answer. Used only to sanity-check a
# labelling pass, never as a label: if an annotator disagrees with the stratum
# on an "obvious" item, that is a signal about the rubric or the item.
EXPECTED_BY_STRATUM = {
    "obvious_important": 1,
    "obvious_unimportant": 0,
    "urgent_unimportant": 0,
    "quiet_critical": 1,
    "borderline": None,
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSON-lines corpus file."""

    if not path.is_file():
        raise FileNotFoundError(path)
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load(private: Optional[Path] = None, *, include_synthetic: bool = True) -> List[Dict[str, Any]]:
    """Return the corpus, tagging every row with its source."""

    rows: List[Dict[str, Any]] = []
    if include_synthetic:
        for row in load_jsonl(SYNTHETIC_PATH):
            rows.append({**row, "source": "synthetic"})
    if private is not None:
        for index, row in enumerate(load_jsonl(Path(private))):
            row.setdefault("id", f"private-{index:04d}")
            row.setdefault("stratum", "unknown")
            rows.append({**row, "source": "private"})

    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError(f"duplicate corpus id: {row['id']}")
        seen.add(row["id"])
    return rows


def by_id(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {row["id"]: row for row in rows}


def base_rate(labels: Iterable[int]) -> float:
    values = list(labels)
    return sum(values) / len(values) if values else 0.0


__all__ = [
    "EXPECTED_BY_STRATUM",
    "STRATA",
    "SYNTHETIC_PATH",
    "base_rate",
    "by_id",
    "load",
    "load_jsonl",
]
