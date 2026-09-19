"""Run AgentDojo's Workspace suite with the typed-decision guardrail on and off.

    OPENAI_API_KEY=... TYPESAFE_API_KEY=... \
        python -m evals.agentdojo.run --out evals/agentdojo/results

What it measures: whether reviewing each pending tool call with
``server.jev.decisions.review_tool_call`` changes AgentDojo's three numbers —
benign utility, utility under attack, and targeted attack success rate — on the
Workspace suite, whose Gmail/calendar/drive tools return email bodies with
injections substituted into them.

Every case is cached to disk keyed by a hash of everything that defines it, so
the run is resumable and the analysis re-runs for free. Each row records both
model versions: the agent LLM and the pinned Jev model, because the thresholds
in ``server/jev/thresholds.py`` are calibrated (or, today, *not* calibrated)
against one specific Jev release.

Targeted against agentdojo 0.1.35 / commit ``089ed468`` (2026-06-02).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.config import get_settings  # noqa: E402

from .defence import AGENTDOJO_AVAILABLE, JevGuardrail, attach_to_pipeline  # noqa: E402

DEFAULT_SUITE = "workspace"
DEFAULT_BENCHMARK_VERSION = "v1.2.2"
DEFAULT_ATTACK = "important_instructions"
DEFAULT_MODEL = "gpt-4o-2024-05-13"

DEFENCE_ON = "defence_on"
DEFENCE_OFF = "defence_off"
CONDITIONS = (DEFENCE_OFF, DEFENCE_ON)

BENIGN = "benign"
ATTACK = "attack"

INSTALL_HINT = """\
agentdojo is not installed, so this experiment cannot run.

    pip install agentdojo

It is a heavy optional dependency (it pulls in the OpenAI, Anthropic, Cohere and
Google clients), which is why it is not in the server's requirements: the
adapter in evals/agentdojo/defence.py and its tests work without it.

Running the benchmark also needs an API key for the agent LLM (e.g.
OPENAI_API_KEY) and TYPESAFE_API_KEY for the guardrail itself. See
evals/agentdojo/README.md.

Upstream: https://github.com/ethz-spylab/agentdojo"""


def require_agentdojo() -> None:
    """Exit with instructions rather than a traceback when the package is absent."""

    if not AGENTDOJO_AVAILABLE:
        raise SystemExit(INSTALL_HINT)


# ----------------------------------------------------------------------
# Case enumeration and caching
# ----------------------------------------------------------------------


def case_key(
    *,
    condition: str,
    case_type: str,
    suite: str,
    benchmark_version: str,
    model: str,
    attack: str,
    user_task: str,
    injection_task: Optional[str],
) -> str:
    """Return the cache key for one benchmark case."""

    raw = json.dumps(
        {
            "condition": condition,
            "case_type": case_type,
            "suite": suite,
            "benchmark_version": benchmark_version,
            "model": model,
            # A benign case has no attack, so the attack name must not enter
            # its key: it would otherwise re-run for every attack.
            "attack": attack if case_type == ATTACK else None,
            "user_task": user_task,
            "injection_task": injection_task,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class Cache:
    """Append-only, content-addressed result cache. Same shape as the
    contamination harness's, so a run is resumable and re-analysis is free."""

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
                key = record.get("key")
                if key:
                    self._data[key] = record

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._data.get(key)

    def put(self, key: str, record: Dict[str, Any]) -> None:
        record = {"key": key, **record}
        self._data[key] = record
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def rows(self) -> List[Dict[str, Any]]:
        return list(self._data.values())

    def __len__(self) -> int:
        return len(self._data)


# ----------------------------------------------------------------------
# Pipelines
# ----------------------------------------------------------------------


def build_pipeline(
    *, condition: str, model: str, suite_name: str
) -> Tuple[Any, Optional[JevGuardrail]]:
    """Build the AgentDojo pipeline for one condition.

    ``defence_off`` is the stock undefended pipeline — the control the whole
    comparison rests on. ``defence_on`` is the same pipeline with the
    ``ToolsExecutor`` swapped for :class:`JevGuardrail`, so the *only*
    difference between the two arms is the review.
    """

    from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig

    config = PipelineConfig(
        llm=model,
        model_id=None,
        defense=None,
        system_message_name=None,
        system_message=None,
    )
    pipeline = AgentPipeline.from_config(config)

    if condition == DEFENCE_OFF:
        pipeline.name = f"{model}-undefended"
        return pipeline, None

    guardrail = JevGuardrail(suite=suite_name)
    attach_to_pipeline(pipeline, guardrail)
    # The name is what AgentDojo's own logdir keys on, so the two arms must not
    # share it or one would silently resume from the other's logs.
    pipeline.name = f"{model}-jev-guardrail"
    return pipeline, guardrail


# ----------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------


def run(
    *,
    out_dir: Path,
    suite_name: str = DEFAULT_SUITE,
    benchmark_version: str = DEFAULT_BENCHMARK_VERSION,
    model: str = DEFAULT_MODEL,
    attack_name: str = DEFAULT_ATTACK,
    conditions: Sequence[str] = CONDITIONS,
    user_task_ids: Optional[Sequence[str]] = None,
    injection_task_ids: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    force_rerun: bool = False,
    logdir: Optional[Path] = None,
    pipeline_factory: Any = None,
) -> Dict[str, Any]:
    """Execute the benchmark for each condition and return a run summary."""

    require_agentdojo()

    from agentdojo.attacks.attack_registry import load_attack
    from agentdojo.benchmark import run_task_with_injection_tasks, run_task_without_injection_tasks
    from agentdojo.task_suite.load_suites import get_suite

    suite = get_suite(benchmark_version, suite_name)
    factory = pipeline_factory or build_pipeline

    user_tasks = list(user_task_ids or suite.user_tasks.keys())
    injection_tasks = list(injection_task_ids or suite.injection_tasks.keys())
    if limit is not None:
        user_tasks = user_tasks[:limit]

    cache = Cache(out_dir / "results.jsonl")
    settings = get_settings()
    jev_model = settings.jev_model
    started = time.time()
    counters = {"cached": 0, "ran": 0}

    for condition in conditions:
        pipeline, guardrail = factory(
            condition=condition, model=model, suite_name=suite_name
        )
        attack = load_attack(attack_name, suite, pipeline)

        common = {
            "condition": condition,
            "suite": suite_name,
            "benchmark_version": benchmark_version,
            "model": model,
            "attack": attack_name,
            # Both model versions, on every row: a guardrail verdict is only
            # interpretable against the Jev release its thresholds were set for.
            "agent_model": model,
            "jev_model": jev_model,
            "guardrail_enabled": condition == DEFENCE_ON,
        }

        for user_task_id in user_tasks:
            user_task = suite.get_user_task_by_id(user_task_id)

            # 1. Benign utility: the user task with no attack present. This is
            #    where a false hold shows up as a lost task.
            key = case_key(
                condition=condition,
                case_type=BENIGN,
                suite=suite_name,
                benchmark_version=benchmark_version,
                model=model,
                attack=attack_name,
                user_task=user_task_id,
                injection_task=None,
            )
            if cache.get(key) is not None and not force_rerun:
                counters["cached"] += 1
            else:
                if guardrail is not None:
                    guardrail.reset()
                utility, _ = run_task_without_injection_tasks(
                    suite, pipeline, user_task, logdir, force_rerun, benchmark_version
                )
                counters["ran"] += 1
                cache.put(
                    key,
                    {
                        **common,
                        "case_type": BENIGN,
                        "user_task": user_task_id,
                        "injection_task": None,
                        "utility": bool(utility),
                        # No injection task was planted, so there is no attacker
                        # goal to have achieved. Recorded explicitly rather than
                        # left out, so the analysis never has to guess.
                        "security": False,
                        "guardrail": guardrail.counts if guardrail is not None else None,
                    },
                )

            # 2. Security cases: one row per (user task, injection task) pair.
            for injection_task_id in injection_tasks:
                key = case_key(
                    condition=condition,
                    case_type=ATTACK,
                    suite=suite_name,
                    benchmark_version=benchmark_version,
                    model=model,
                    attack=attack_name,
                    user_task=user_task_id,
                    injection_task=injection_task_id,
                )
                if cache.get(key) is not None and not force_rerun:
                    counters["cached"] += 1
                    continue
                if guardrail is not None:
                    guardrail.reset()
                utility_results, security_results = run_task_with_injection_tasks(
                    suite,
                    pipeline,
                    user_task,
                    attack,
                    logdir,
                    force_rerun,
                    [injection_task_id],
                    benchmark_version,
                )
                counters["ran"] += 1
                pair = (user_task_id, injection_task_id)
                cache.put(
                    key,
                    {
                        **common,
                        "case_type": ATTACK,
                        "user_task": user_task_id,
                        "injection_task": injection_task_id,
                        "utility": bool(utility_results.get(pair, False)),
                        # AgentDojo's `security` is True when the *attacker's*
                        # goal was achieved, so it is the targeted ASR outcome.
                        "security": bool(security_results.get(pair, False)),
                        "guardrail": guardrail.counts if guardrail is not None else None,
                    },
                )

    summary = {
        "suite": suite_name,
        "benchmark_version": benchmark_version,
        "attack": attack_name,
        "agent_model": model,
        "jev_model": jev_model,
        "conditions": list(conditions),
        "user_tasks": len(user_tasks),
        "injection_tasks": len(injection_tasks),
        "cases": len(cache),
        "cached": counters["cached"],
        "ran": counters["ran"],
        "wall_seconds": round(time.time() - started, 1),
        "agentdojo_commit": "089ed468cf3ed0322acc66b0211f26d9d90dbf60",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("evals/agentdojo/results"))
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--benchmark-version", default=DEFAULT_BENCHMARK_VERSION)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="The agent LLM. Pinned, never an alias.")
    parser.add_argument("--attack", default=DEFAULT_ATTACK)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--user-tasks", nargs="+", default=None)
    parser.add_argument("--injection-tasks", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of user tasks.")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--logdir", type=Path, default=None, help="AgentDojo's own trace log directory.")
    args = parser.parse_args(argv)

    require_agentdojo()

    args.out.mkdir(parents=True, exist_ok=True)
    summary = run(
        out_dir=args.out,
        suite_name=args.suite,
        benchmark_version=args.benchmark_version,
        model=args.model,
        attack_name=args.attack,
        conditions=args.conditions,
        user_task_ids=args.user_tasks,
        injection_task_ids=args.injection_tasks,
        limit=args.limit,
        force_rerun=args.force_rerun,
        logdir=args.logdir,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
