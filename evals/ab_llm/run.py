"""Run the Jev-versus-LLM A/B experiment.

    TYPESAFE_API_KEY=... OPENROUTER_API_KEY=... python -m evals.ab_llm.run

The server replaced an LLM classification call with a typed Jev question. That
swap has never been measured: we know what Jev answers, we do not know what the
LLM it replaced would have answered on the same mail. This harness asks both
arms the *same* question about the *same* state and records both answers.

Two arms:

* ``jev``  -- ``typesafe_sdk.AsyncTypeSafeClient``, model pinned ``jev-1.13.0``.
* ``llm``  -- the identical ``state`` and the identical ``questions`` dicts from
  :mod:`server.jev.questions`, routed through an LLM backend.

The single validity claim this file exists to support is that the two arms see
byte-identical inputs. Everything below is arranged so that claim is structural
rather than aspirational: ``build_state`` and ``questions_for_trial`` take no arm
argument, both arms are handed the *same object*, and the SHA-256 of each is
written into every output row so a reader can verify the pairing afterwards
without trusting this docstring.

Caching, resumability and the run-summary shape follow
``evals/contamination/run.py``; the statistics live in
``evals.contamination.stats`` and are not reimplemented here.
"""

from __future__ import annotations

import abc
import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.jev import questions as q  # noqa: E402
from server.jev import state as s  # noqa: E402

from evals.contamination import carriers as carriers_module  # noqa: E402
from evals.contamination.persona import persona_brief  # noqa: E402

ARM_JEV = "jev"
ARM_LLM = "llm"
ARMS = (ARM_JEV, ARM_LLM)

DEFAULT_JEV_MODEL = "jev-1.13.0"
DEFAULT_LLM_MODEL = "openai/gpt-4o-mini"

# $0.042 per 1M input tokens, output free. https://docs.typesafe.ai/models
JEV_INPUT_COST_PER_MILLION = 0.042
# OpenRouter prices vary per model and change without notice, so the LLM arm's
# rate is a flag rather than a constant. The default is a placeholder and is
# recorded in every row so a reader can see which number produced a cost.
DEFAULT_LLM_COST_PER_MILLION = 0.150

# Where a backend's probabilities come from. This is the honesty switch that
# ``analyze`` keys its calibration warning off, and the documented hook for a
# future logprob-based arm: a backend that reads token logprobs sets
# ``PROB_SOURCE_LOGPROB`` and the warning stops applying to it.
PROB_SOURCE_NOUL = "noul"
PROB_SOURCE_VERBALISED = "verbalised_structured_output"
PROB_SOURCE_LOGPROB = "logprob"

ADAPTER_IMPORT_HINT = (
    "The optional adapter package is not importable. Install it with\n"
    "    pip install 'system-one-adapter @ "
    "git+https://github.com/typesafe-ai/system-one-adapter-python@<commit-sha>'\n"
    "and pin a commit: at the time of writing that repository has one commit and "
    "zero stars, so its API can change under you without a release.\n"
    "Or run the LLM arm with --llm-backend openrouter, which needs nothing extra."
)


# ----------------------------------------------------------------------
# The shared inputs. Neither function takes an arm.
# ----------------------------------------------------------------------


def questions_for_trial() -> Dict[str, Dict[str, Any]]:
    """Return the question set both arms are asked.

    Deliberately parameterless. The moment this grows an ``arm`` argument the
    experiment stops being a comparison of two answerers and becomes a
    comparison of two questions.
    """

    return dict(q.EMAIL_QUESTIONS)


def build_state(carrier: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the request state for one carrier email, persona included."""

    state = s.email_state(
        sender=str(carrier["sender"]),
        recipient="Rio Tanaka <rio@northgate.example>",
        subject=str(carrier["subject"]),
        body=str(carrier["body"]),
    )
    state["recipient_context"] = persona_brief()
    return state


def canonical(payload: Any) -> str:
    """Canonical JSON: the exact bytes whose hash is written into a row."""

    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def sha256_of(payload: Any) -> str:
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# Answer shape, shared by every backend
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Answer:
    """One probability, in the shape ``typesafe_sdk`` returns."""

    noul: float


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class BackendResponse:
    """What every backend returns, whatever it wraps."""

    answers: Dict[str, Answer] = field(default_factory=dict)
    model: str = ""
    usage: Usage = field(default_factory=Usage)


class Backend(abc.ABC):
    """One answerer.

    Subclasses differ only in how a probability is produced. They must not touch
    the ``state`` or ``questions`` they are handed: both arms are given the same
    objects, and mutating them would silently destroy the pairing.
    """

    #: Recorded in every output row, so a result is always attributable.
    identifier: str = "backend"
    #: See PROB_SOURCE_* above. Drives the calibration warning in ``analyze``.
    probability_source: str = PROB_SOURCE_VERBALISED
    #: USD per 1M input tokens for this backend.
    input_cost_per_million: float = 0.0

    @abc.abstractmethod
    async def system_one(
        self,
        *,
        state: Mapping[str, Any],
        questions: Mapping[str, Mapping[str, Any]],
    ) -> Any:
        """Answer every question in ``questions`` about ``state``."""

    async def aclose(self) -> None:  # pragma: no cover - trivial default
        return None


class JevBackend(Backend):
    """The production path: a real ``AsyncTypeSafeClient``."""

    probability_source = PROB_SOURCE_NOUL
    input_cost_per_million = JEV_INPUT_COST_PER_MILLION

    def __init__(self, *, api_key: str, model: str = DEFAULT_JEV_MODEL, timeout: float = 20.0) -> None:
        from typesafe_sdk import AsyncTypeSafeClient

        self._client = AsyncTypeSafeClient(api_key=api_key, model=model, timeout=timeout)
        self.identifier = f"typesafe_sdk:{model}"

    async def system_one(self, *, state: Mapping[str, Any], questions: Mapping[str, Any]) -> Any:
        return await self._client.system_one(state=state, questions=questions)

    async def aclose(self) -> None:
        await self._client.aclose()


class AdapterBackend(Backend):
    """``system_one`` served by an OpenAI/Anthropic-compatible provider.

    Wraps https://github.com/typesafe-ai/system-one-adapter-python, which is a
    drop-in replacement for the SDK's ``system_one`` API. Two things a reader
    should know before trusting a number produced through it:

    * It is a **1-commit, 0-star repository**. Pin a commit; do not track a
      branch. Nothing here depends on it at import time -- the import happens in
      this constructor, so the module loads and the OpenRouter arm runs fine
      when the package is absent.
    * It derives probabilities from **structured output, not logprobs**, so what
      it returns is a *verbalised* confidence. That is a different quantity from
      Jev's noul and is known to be poorly calibrated and quantised to round
      numbers. See the README.
    """

    probability_source = PROB_SOURCE_VERBALISED

    def __init__(
        self,
        *,
        model: str = DEFAULT_LLM_MODEL,
        api_key: Optional[str] = None,
        cost_per_million: float = DEFAULT_LLM_COST_PER_MILLION,
        provider: str = "openai",
    ) -> None:
        try:  # imported here, never at module import time
            import system_one_adapter  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised by hand
            raise SystemExit(f"{exc}\n\n{ADAPTER_IMPORT_HINT}") from exc

        factory = getattr(system_one_adapter, "AsyncSystemOneClient", None)
        if factory is None:  # pragma: no cover - depends on an unpinned package
            raise SystemExit(
                "system_one_adapter is installed but exposes no AsyncSystemOneClient; "
                "this is what an unpinned 1-commit dependency looks like.\n"
                + ADAPTER_IMPORT_HINT
            )
        self._client = factory(model=model, api_key=api_key, provider=provider)
        version = getattr(system_one_adapter, "__version__", "unpinned")
        self.identifier = f"system_one_adapter[{provider}]:{model}@{version}"
        self.input_cost_per_million = cost_per_million

    async def system_one(self, *, state: Mapping[str, Any], questions: Mapping[str, Any]) -> Any:
        return await self._client.system_one(state=state, questions=questions)

    async def aclose(self) -> None:  # pragma: no cover - depends on the package
        close = getattr(self._client, "aclose", None)
        if close is not None:
            await close()


def tool_schema(questions: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Mirror the question dicts as a single tool call.

    The schema is *derived* from the questions rather than written out again, so
    there is no second copy of the wording to drift from the first.
    """

    properties: Dict[str, Any] = {}
    for name, question in questions.items():
        properties[name] = {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": str(question.get("instructions", "")),
        }
    return [
        {
            "type": "function",
            "function": {
                "name": "answer_questions",
                "description": "Answer every question with a probability in [0, 1].",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(questions),
                    "additionalProperties": False,
                },
            },
        }
    ]


OPENROUTER_SYSTEM = (
    "You answer typed yes/no questions about a piece of data. For each question "
    "return the probability, between 0 and 1, that the statement in its "
    "`instructions` is true of the data in `state`. Use the `criteria` object as "
    "written. Answer every question independently. Call the `answer_questions` "
    "tool exactly once and say nothing else."
)


class OpenRouterBackend(Backend):
    """A plain OpenRouter arm, reusing the server's own client.

    Needs no third-party adapter, which is the point: the comparison still runs
    when the 1-commit adapter repository is unavailable.
    """

    probability_source = PROB_SOURCE_VERBALISED

    def __init__(
        self,
        *,
        model: str = DEFAULT_LLM_MODEL,
        api_key: Optional[str] = None,
        cost_per_million: float = DEFAULT_LLM_COST_PER_MILLION,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self.identifier = f"openrouter:{model}"
        self.input_cost_per_million = cost_per_million

    async def system_one(self, *, state: Mapping[str, Any], questions: Mapping[str, Any]) -> Any:
        from server.openrouter_client import request_chat_completion

        # The model is shown the same question objects the SDK would be sent,
        # serialised canonically. No paraphrase, no per-arm rewording.
        content = (
            "state:\n"
            + canonical(state)
            + "\n\nquestions:\n"
            + canonical(questions)
        )
        payload = await request_chat_completion(
            model=self._model,
            messages=[{"role": "user", "content": content}],
            system=OPENROUTER_SYSTEM,
            api_key=self._api_key,
            tools=tool_schema(questions),
        )
        return self._parse(payload, questions)

    @staticmethod
    def _parse(payload: Mapping[str, Any], questions: Mapping[str, Any]) -> BackendResponse:
        choices = payload.get("choices") or []
        arguments = "{}"
        if choices:
            message = choices[0].get("message") or {}
            calls = message.get("tool_calls") or []
            if calls:
                arguments = (calls[0].get("function") or {}).get("arguments") or "{}"
            elif message.get("content"):
                arguments = str(message["content"])
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM arm returned unparseable arguments: {arguments[:200]}") from exc

        answers: Dict[str, Answer] = {}
        for name in questions:
            value = parsed.get(name)
            if isinstance(value, Mapping):
                value = value.get("probability")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            answers[name] = Answer(noul=max(0.0, min(1.0, float(value))))

        usage = payload.get("usage") or {}
        return BackendResponse(
            answers=answers,
            model=str(payload.get("model") or ""),
            usage=Usage(
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
            ),
        )


# ----------------------------------------------------------------------
# Trials and cache
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Trial:
    """One arm answering one carrier email once."""

    carrier_id: str
    arm: str
    repeat: int

    def key(self, backend_id: str) -> str:
        """Cache key. The backend identifier is part of it, so swapping the
        model behind an arm does not silently reuse the previous model's rows."""

        raw = canonical({**asdict(self), "backend": backend_id})
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def enumerate_trials(
    *,
    repeats: int,
    arms: Sequence[str] = ARMS,
    carrier_ids: Optional[Sequence[str]] = None,
) -> List[Trial]:
    """Build the trial grid: every arm answers every carrier the same number of times."""

    ids = list(carrier_ids or [c["id"] for c in carriers_module.CARRIERS])
    return [
        Trial(carrier_id, arm, repeat)
        for carrier_id in ids
        for arm in arms
        for repeat in range(repeats)
    ]


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


def build_backends(
    *,
    jev_model: str,
    llm_backend: str,
    llm_model: str,
    llm_cost_per_million: float,
) -> Dict[str, Backend]:
    """Construct the real backends. Tests inject fakes instead of calling this."""

    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        raise SystemExit("TYPESAFE_API_KEY is not set")

    llm: Backend
    if llm_backend == "adapter":
        llm = AdapterBackend(model=llm_model, cost_per_million=llm_cost_per_million)
    elif llm_backend == "openrouter":
        llm = OpenRouterBackend(model=llm_model, cost_per_million=llm_cost_per_million)
    else:  # pragma: no cover - argparse constrains this
        raise SystemExit(f"unknown llm backend {llm_backend!r}")

    return {ARM_JEV: JevBackend(api_key=api_key, model=jev_model), ARM_LLM: llm}


async def run(
    *,
    out_dir: Path,
    repeats: int,
    backends: Mapping[str, Backend],
    concurrency: int = 4,
    limit: Optional[int] = None,
    carrier_ids: Optional[Sequence[str]] = None,
    close_backends: bool = False,
) -> Dict[str, Any]:
    """Execute the grid and return a run summary.

    ``backends`` maps arm name -> :class:`Backend`. It is a required argument
    rather than something this function builds, so the tests can run the whole
    harness with no network and no keys.
    """

    carriers = carriers_module.carriers_by_id()
    arms = tuple(backends)
    trials = enumerate_trials(repeats=repeats, arms=arms, carrier_ids=carrier_ids)
    if limit is not None:
        trials = trials[:limit]

    cache = Cache(out_dir / "responses.jsonl")
    semaphore = asyncio.Semaphore(concurrency)
    started = time.time()
    counters: Dict[str, Dict[str, int]] = {
        arm: {"cached": 0, "called": 0, "failed": 0, "input_tokens": 0} for arm in arms
    }

    # Built once, outside the loop, and handed to both arms unchanged. This is
    # the pairing: there is only ever one questions object per process.
    questions = questions_for_trial()
    questions_hash = sha256_of(questions)

    async def one(trial: Trial) -> None:
        backend = backends[trial.arm]
        key = trial.key(backend.identifier)
        if cache.get(key) is not None:
            counters[trial.arm]["cached"] += 1
            return

        state = build_state(carriers[trial.carrier_id])

        async with semaphore:
            call_started = time.time()
            try:
                response = await backend.system_one(state=state, questions=questions)
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                counters[trial.arm]["failed"] += 1
                cache.put(
                    key,
                    {
                        **asdict(trial),
                        "backend": backend.identifier,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                return
            latency_ms = int((time.time() - call_started) * 1000)

        answers: Dict[str, Optional[float]] = {}
        for name in questions:
            answer = getattr(response, "answers", {}).get(name)
            value = getattr(answer, "noul", None)
            answers[name] = float(value) if isinstance(value, (int, float)) else None

        usage = getattr(response, "usage", None)
        tokens = int(getattr(usage, "input_tokens", None) or 0)
        counters[trial.arm]["input_tokens"] += tokens
        counters[trial.arm]["called"] += 1

        cache.put(
            key,
            {
                **asdict(trial),
                "answers": answers,
                "backend": backend.identifier,
                "probability_source": backend.probability_source,
                "input_cost_per_million": backend.input_cost_per_million,
                "model": getattr(response, "model", "") or backend.identifier,
                "input_tokens": tokens,
                "latency_ms": latency_ms,
                # Hashes of the exact inputs, so a later reader can prove the
                # two arms answered the same question about the same state
                # without the bodies being republished.
                "state_sha256": sha256_of(state),
                "questions_sha256": questions_hash,
            },
        )

    await asyncio.gather(*(one(trial) for trial in trials))
    if close_backends:
        for backend in backends.values():
            await backend.aclose()

    per_arm = {
        arm: {
            **counters[arm],
            "backend": backends[arm].identifier,
            "probability_source": backends[arm].probability_source,
            "input_cost_per_million": backends[arm].input_cost_per_million,
            "estimated_cost_usd": round(
                counters[arm]["input_tokens"] * backends[arm].input_cost_per_million / 1_000_000, 8
            ),
        }
        for arm in arms
    }
    summary = {
        "trials": len(trials),
        "repeats": repeats,
        "arms": list(arms),
        "questions_sha256": questions_hash,
        "question_keys": list(questions),
        "per_arm": per_arm,
        "cached": sum(per_arm[arm]["cached"] for arm in arms),
        "called": sum(per_arm[arm]["called"] for arm in arms),
        "failed": sum(per_arm[arm]["failed"] for arm in arms),
        "input_tokens": sum(per_arm[arm]["input_tokens"] for arm in arms),
        "estimated_cost_usd": round(sum(per_arm[arm]["estimated_cost_usd"] for arm in arms), 8),
        "wall_seconds": round(time.time() - started, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("evals/ab_llm/results"))
    parser.add_argument("--repeats", type=int, default=3, help="Both arms drift; repeat.")
    parser.add_argument("--jev-model", default=DEFAULT_JEV_MODEL, help="Pinned, never an alias.")
    parser.add_argument("--llm-backend", default="openrouter", choices=("openrouter", "adapter"))
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-cost-per-million", type=float, default=DEFAULT_LLM_COST_PER_MILLION)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Count trials and stop.")
    args = parser.parse_args(argv)

    if args.dry_run:
        trials = enumerate_trials(repeats=args.repeats)
        print(f"{len(trials)} trials ({len(ARMS)} arms)")
        jev = len(trials) / len(ARMS) * 1000 / 1_000_000 * JEV_INPUT_COST_PER_MILLION
        llm = len(trials) / len(ARMS) * 1000 / 1_000_000 * args.llm_cost_per_million
        print(f"~${jev:.4f} jev + ~${llm:.4f} llm at ~1k input tokens per call")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    backends = build_backends(
        jev_model=args.jev_model,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        llm_cost_per_million=args.llm_cost_per_million,
    )
    summary = asyncio.run(
        run(
            out_dir=args.out,
            repeats=args.repeats,
            backends=backends,
            concurrency=args.concurrency,
            limit=args.limit,
            close_backends=True,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
