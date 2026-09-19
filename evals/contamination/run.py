"""Run the cross-question contamination experiment.

    TYPESAFE_API_KEY=... python -m evals.contamination.run --out evals/contamination/results

What it measures: whether hostile text inside an email body moves the answer to
the **importance** question, and whether asking the injection question in the
same batched request makes that better or worse.

Every response is cached to disk keyed by a hash of the exact request, so the
whole run is resumable and the analysis re-runs for free. Nothing is sent
anywhere except api.typesafe.ai, and the API key is read from the environment
and never written to the cache.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.jev import questions as q  # noqa: E402
from server.jev import state as s  # noqa: E402

from . import carriers as carriers_module  # noqa: E402
from . import payloads as payloads_module  # noqa: E402
from .persona import persona_brief  # noqa: E402

DEFAULT_MODEL = "jev-1.13.0"

# The two question configurations. The comparison between them is the whole
# experiment: if the importance answer behaves differently depending on whether
# the injection question rides along, the questions are leaking into each other.
CONFIG_WITH = "with_injection_question"
CONFIG_WITHOUT = "without_injection_question"


def questions_for(config: str) -> Dict[str, Dict[str, Any]]:
    """Return the question set for a configuration."""

    if config == CONFIG_WITHOUT:
        return {key: value for key, value in q.EMAIL_QUESTIONS.items() if key != "prompt_injection"}
    return dict(q.EMAIL_QUESTIONS)


@dataclass(frozen=True)
class Trial:
    """One request: a carrier, a treatment, a position, a question config."""

    carrier_id: str
    category: str          # "clean", "filler", or a payload category
    template_index: int    # -1 for clean and filler
    position: str          # "none" for clean
    config: str
    goal: str
    repeat: int

    def key(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def build_body(carrier: Mapping[str, Any], trial: Trial) -> str:
    """Return the carrier body with the trial's treatment applied."""

    body = str(carrier["body"])
    if trial.category == "clean":
        return body
    if trial.category == "filler":
        return payloads_module.inject(body, payloads_module.FILLER, trial.position)
    payload = payloads_module.render(trial.category, trial.template_index, trial.goal)
    return payloads_module.inject(body, payload, trial.position)


def build_state(carrier: Mapping[str, Any], body: str) -> Dict[str, Any]:
    """Build the request state, with the persona as explicit context.

    jev-sec-bench found that telling the model what the assistant is *for* moved
    accuracy 89.7% -> 96.5% while AUC barely moved: the ordering was already
    right, what changed was whether the probabilities landed where a fixed
    threshold could use them. The persona is that context here.
    """

    state = s.email_state(
        sender=str(carrier["sender"]),
        recipient="Rio Tanaka <rio@northgate.example>",
        subject=str(carrier["subject"]),
        body=body,
    )
    state["recipient_context"] = persona_brief()
    return state


def enumerate_trials(
    *,
    templates_per_category: int,
    positions: Sequence[str],
    repeats: int,
    goal: str,
    carrier_ids: Optional[Sequence[str]] = None,
) -> List[Trial]:
    """Build the full trial grid, controls included."""

    ids = list(carrier_ids or [c["id"] for c in carriers_module.CARRIERS])
    trials: List[Trial] = []

    for carrier_id in ids:
        for config in (CONFIG_WITH, CONFIG_WITHOUT):
            for repeat in range(repeats):
                # Baseline 1: the same email, untouched. ASR only means anything
                # as a paired delta from this.
                trials.append(
                    Trial(carrier_id, "clean", -1, "none", config, goal, repeat)
                )
                # Baseline 2: same length, same position, no adversarial content.
                for position in positions:
                    trials.append(
                        Trial(carrier_id, "filler", -1, position, config, goal, repeat)
                    )
                for category in payloads_module.PAYLOADS:
                    count = min(templates_per_category, len(payloads_module.PAYLOADS[category]))
                    for template_index in range(count):
                        for position in positions:
                            trials.append(
                                Trial(
                                    carrier_id,
                                    category,
                                    template_index,
                                    position,
                                    config,
                                    goal,
                                    repeat,
                                )
                            )
    return trials


class Cache:
    """Content-addressed response cache, so a run is resumable and free to redo."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: Dict[str, Dict[str, Any]] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._data[record["key"]] = record

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._data.get(key)

    def put(self, key: str, record: Dict[str, Any]) -> None:
        record = {"key": key, **record}
        self._data[key] = record
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._data)


async def run(
    *,
    out_dir: Path,
    templates_per_category: int,
    positions: Sequence[str],
    repeats: int,
    goal: str,
    model: str,
    concurrency: int,
    limit: Optional[int],
    client: Any = None,
) -> Dict[str, Any]:
    """Execute the grid and return a run summary."""

    from typesafe_sdk import AsyncTypeSafeClient

    carriers = carriers_module.carriers_by_id()
    trials = enumerate_trials(
        templates_per_category=templates_per_category,
        positions=positions,
        repeats=repeats,
        goal=goal,
    )
    if limit is not None:
        trials = trials[:limit]

    cache = Cache(out_dir / "responses.jsonl")
    owns_client = client is None
    if owns_client:
        api_key = os.environ.get("TYPESAFE_API_KEY")
        if not api_key:
            raise SystemExit("TYPESAFE_API_KEY is not set")
        client = AsyncTypeSafeClient(api_key=api_key, model=model, timeout=20.0)

    semaphore = asyncio.Semaphore(concurrency)
    started = time.time()
    counters = {"cached": 0, "called": 0, "failed": 0}
    input_tokens = 0

    async def one(trial: Trial) -> None:
        nonlocal input_tokens
        key = trial.key()
        if cache.get(key) is not None:
            counters["cached"] += 1
            return

        carrier = carriers[trial.carrier_id]
        body = build_body(carrier, trial)
        state = build_state(carrier, body)
        questions = questions_for(trial.config)

        async with semaphore:
            call_started = time.time()
            try:
                response = await client.system_one(state=state, questions=questions)
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                counters["failed"] += 1
                cache.put(key, {**asdict(trial), "error": f"{type(exc).__name__}: {exc}"})
                return
            latency_ms = int((time.time() - call_started) * 1000)

        answers: Dict[str, Optional[float]] = {}
        for name in questions:
            answer = getattr(response, "answers", {}).get(name)
            value = getattr(answer, "noul", None)
            answers[name] = float(value) if isinstance(value, (int, float)) else None

        usage = getattr(response, "usage", None)
        tokens = getattr(usage, "input_tokens", None) or 0
        input_tokens += tokens
        counters["called"] += 1

        cache.put(
            key,
            {
                **asdict(trial),
                "answers": answers,
                "model": getattr(response, "model", model),
                "input_tokens": tokens,
                "latency_ms": latency_ms,
                # A hash of the exact state, so a later reader can prove two
                # rows saw the same input without the bodies being republished.
                "state_sha256": hashlib.sha256(
                    json.dumps(state, sort_keys=True, ensure_ascii=False).encode("utf-8")
                ).hexdigest(),
            },
        )

    await asyncio.gather(*(one(trial) for trial in trials))
    if owns_client:
        await client.aclose()

    summary = {
        "trials": len(trials),
        "cached": counters["cached"],
        "called": counters["called"],
        "failed": counters["failed"],
        "input_tokens": input_tokens,
        # $0.042 per 1M input tokens; output tokens are free.
        # https://docs.typesafe.ai/models
        "estimated_cost_usd": round(input_tokens * 0.042 / 1_000_000, 4),
        "wall_seconds": round(time.time() - started, 1),
        "model": model,
        "templates_per_category": templates_per_category,
        "positions": list(positions),
        "repeats": repeats,
        "goal": goal,
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("evals/contamination/results"))
    parser.add_argument("--templates-per-category", type=int, default=10)
    parser.add_argument("--positions", nargs="+", default=list(payloads_module.POSITIONS))
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Jev drifts run to run; repeat and treat the outcome as a rate.",
    )
    parser.add_argument("--goal", default="force_important", choices=sorted(payloads_module.GOALS))
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Pinned, never an alias.")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Count trials and stop.")
    args = parser.parse_args(argv)

    if args.dry_run:
        trials = enumerate_trials(
            templates_per_category=args.templates_per_category,
            positions=args.positions,
            repeats=args.repeats,
            goal=args.goal,
        )
        print(f"{len(trials)} trials")
        print(f"~{len(trials) * 1000 / 1_000_000 * 0.042:.2f} USD at ~1k input tokens per call")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    summary = asyncio.run(
        run(
            out_dir=args.out,
            templates_per_category=args.templates_per_category,
            positions=args.positions,
            repeats=args.repeats,
            goal=args.goal,
            model=args.model,
            concurrency=args.concurrency,
            limit=args.limit,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
