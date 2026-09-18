"""The typed decisions OpenPoke makes with Jev.

Each function here turns one batched Jev call into a small frozen result that
call sites can branch on. Every one of them has an "I don't know" state, and
reaching it must leave the caller behaving exactly as it did before this layer
existed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..config import get_settings
from ..logging_config import logger
from . import questions as q
from . import state as s
from . import thresholds as t
from .client import ask, is_enabled, noul

# ----------------------------------------------------------------------
# Email screening
# ----------------------------------------------------------------------

SURFACE = "surface"
SKIP = "skip"
UNDECIDED = "undecided"


@dataclass(frozen=True)
class EmailScreening:
    """Outcome of screening one incoming email."""

    verdict: str
    reason: str
    probabilities: Dict[str, float] = field(default_factory=dict)

    @property
    def decided(self) -> bool:
        """Return ``True`` when the caller may skip the LLM decision entirely."""

        return self.verdict in (SURFACE, SKIP)


UNDECIDED_EMAIL = EmailScreening(verdict=UNDECIDED, reason="jev_unavailable")


# Screen one email with a single batched Jev call
async def screen_email(
    *,
    sender: str,
    recipient: str,
    subject: str,
    body: str,
    labels: Sequence[str] = (),
    has_attachments: bool = False,
    attachment_filenames: Sequence[str] = (),
) -> EmailScreening:
    """Decide whether an email should be surfaced, dropped, or sent to an LLM."""

    settings = get_settings()
    if not settings.jev_email_screening_enabled or not is_enabled():
        return UNDECIDED_EMAIL

    response = await ask(
        state=s.email_state(
            sender=sender,
            recipient=recipient,
            subject=subject,
            body=body,
            labels=labels,
            has_attachments=has_attachments,
            attachment_filenames=attachment_filenames,
        ),
        questions=q.EMAIL_QUESTIONS,
        purpose="email_screening",
    )
    if response is None:
        return UNDECIDED_EMAIL

    probabilities: Dict[str, float] = {}
    for key in q.EMAIL_QUESTIONS:
        value = noul(response, key)
        if value is not None:
            probabilities[key] = value

    # The security gate is checked first and on its own: it must not depend on
    # an unrelated answer having arrived.
    injection = probabilities.get("prompt_injection")
    if injection is not None and injection >= t.EMAIL_PROMPT_INJECTION:
        # Never hand attacker-authored instructions to the interaction agent.
        logger.warning(
            "Suppressing email that reads as a prompt-injection attempt",
            extra={"prompt_injection": injection},
        )
        return EmailScreening(verdict=SKIP, reason="prompt_injection", probabilities=probabilities)

    important = probabilities.get("important")
    if important is None:
        # The question the rest of the decision turns on came back unusable.
        return EmailScreening(verdict=UNDECIDED, reason="missing_answer", probabilities=probabilities)

    security = probabilities.get("security_code")
    if security is not None and security >= t.EMAIL_SECURITY_CODE:
        return EmailScreening(verdict=SURFACE, reason="security_code", probabilities=probabilities)

    if important >= t.EMAIL_IMPORTANT_HIGH:
        return EmailScreening(verdict=SURFACE, reason="important", probabilities=probabilities)

    if important <= t.EMAIL_IMPORTANT_LOW:
        return EmailScreening(verdict=SKIP, reason="not_important", probabilities=probabilities)

    bulk = probabilities.get("automated_bulk")
    if bulk is not None and bulk >= t.EMAIL_AUTOMATED_BULK:
        return EmailScreening(verdict=SKIP, reason="automated_bulk", probabilities=probabilities)

    # Between the bands: noul probabilities near 0.5 are the uncertain region,
    # so this is exactly where the expensive model earns its cost.
    return EmailScreening(verdict=UNDECIDED, reason="uncertain", probabilities=probabilities)


# ----------------------------------------------------------------------
# Execution-agent tool guardrail
# ----------------------------------------------------------------------

ALLOW = "allow"
HOLD = "hold"


@dataclass(frozen=True)
class ToolCallReview:
    """Advisory verdict on a pending tool call."""

    verdict: str
    reason: str
    probabilities: Dict[str, float] = field(default_factory=dict)

    @property
    def held(self) -> bool:
        """Return ``True`` when the call should be returned to the agent unrun."""

        return self.verdict == HOLD

    def explain(self) -> str:
        """Return the message handed back to the agent when a call is held."""

        detail = ", ".join(f"{key}={value:.2f}" for key, value in sorted(self.probabilities.items()))
        return (
            f"Blocked by the typed-decision guardrail ({self.reason}). "
            "Re-read your assignment and either correct the arguments or "
            f"explain what you intend to do. [{detail}]"
        )


ALLOWED_TOOL_CALL = ToolCallReview(verdict=ALLOW, reason="jev_unavailable")


# Review a pending tool call with a single batched Jev call
async def review_tool_call(
    *,
    assignment: str,
    tool_name: str,
    arguments: Mapping[str, Any],
) -> ToolCallReview:
    """Judge whether a tool call matches the assignment before it runs.

    The verdict is advisory and fails open. A held call is reported back to the
    agent as a tool error so it can correct itself; it is never surfaced to the
    user as a refusal.
    """

    settings = get_settings()
    if not settings.jev_tool_guardrail_enabled or not is_enabled():
        return ALLOWED_TOOL_CALL
    if not assignment or not tool_name:
        return ALLOWED_TOOL_CALL
    if not isinstance(arguments, Mapping):
        # Tool arguments come from `json.loads` on model output and are not
        # guaranteed to be an object. Let the tool itself reject the shape, as
        # it did before this guardrail existed.
        return ALLOWED_TOOL_CALL

    response = await ask(
        state=s.tool_call_state(assignment=assignment, tool_name=tool_name, arguments=arguments),
        questions=q.TOOL_QUESTIONS,
        purpose="tool_guardrail",
        deadline=settings.jev_guardrail_deadline_seconds,
    )
    if response is None:
        return ALLOWED_TOOL_CALL

    probabilities: Dict[str, float] = {}
    for key in q.TOOL_QUESTIONS:
        value = noul(response, key)
        if value is not None:
            probabilities[key] = value

    mismatch = probabilities.get("intent_mismatch")
    if mismatch is not None and mismatch >= t.TOOL_INTENT_MISMATCH_HOLD:
        return ToolCallReview(verdict=HOLD, reason="intent_mismatch", probabilities=probabilities)

    irreversible_answer = probabilities.get("irreversible")
    irreversible = tool_name in t.IRREVERSIBLE_TOOLS or (
        irreversible_answer is not None and irreversible_answer >= t.TOOL_IRREVERSIBLE_HOLD
    )

    # Off-task work is only worth stopping when it cannot be undone; gating
    # every read on it would make the agent useless.
    if irreversible:
        off_task = probabilities.get("off_task")
        if off_task is not None and off_task >= t.TOOL_OFF_TASK_HOLD:
            return ToolCallReview(verdict=HOLD, reason="off_task_irreversible", probabilities=probabilities)

    return ToolCallReview(verdict=ALLOW, reason="passed", probabilities=probabilities)


# ----------------------------------------------------------------------
# Email search relevance filter
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SearchFilter:
    """Outcome of verifying an LLM's search-result selection."""

    kept: List[str]
    dropped: List[str] = field(default_factory=list)
    applied: bool = False


# Verify an LLM's chosen search results against the original request
async def filter_search_results(
    *,
    request: str,
    selected_ids: Sequence[str],
    candidates: Mapping[str, Mapping[str, Any]],
) -> SearchFilter:
    """Drop selected results Jev is confident do not answer the request.

    ``candidates`` maps message id to a mapping with ``from``/``subject``/``body``
    (or ``sender``/``clean_text``) keys. Ids missing from it are always kept —
    this filter only ever removes results it could actually read.
    """

    settings = get_settings()
    ordered = [message_id for message_id in selected_ids if message_id in candidates]
    # Same cap the state builder applies, so question keys and candidate
    # indexes cannot drift apart.
    ordered = ordered[: s.search_candidate_limit()]
    if not settings.jev_search_filter_enabled or not is_enabled() or len(ordered) < 2:
        # With a single result there is nothing to discriminate between, and
        # dropping it would turn a hit into an empty answer.
        return SearchFilter(kept=list(selected_ids))

    response = await ask(
        state=s.search_state(
            request=request,
            candidates=[candidates[message_id] for message_id in ordered],
        ),
        questions=q.search_relevance_questions(len(ordered)),
        purpose="search_relevance",
    )
    if response is None:
        return SearchFilter(kept=list(selected_ids))

    dropped: List[str] = []
    for index, message_id in enumerate(ordered):
        value = noul(response, q.relevance_key(index))
        if value is not None and value <= t.SEARCH_RELEVANCE_DROP:
            dropped.append(message_id)

    if not dropped:
        return SearchFilter(kept=list(selected_ids), applied=True)

    kept = [message_id for message_id in selected_ids if message_id not in dropped]
    # Counted among the ids that were actually scored: an id the filter could
    # not read (an LLM-hallucinated message id, say) is not evidence that the
    # filter left a usable result behind.
    scored_survivors = [message_id for message_id in ordered if message_id not in dropped]
    if len(scored_survivors) < t.SEARCH_MIN_SURVIVORS:
        # Everything scored low. That is a signal about the query, not about
        # any one result, so the LLM's selection stands.
        logger.info(
            "Skipping search relevance filter; every result scored low",
            extra={"candidates": len(ordered)},
        )
        return SearchFilter(kept=list(selected_ids))

    logger.info(
        "Search relevance filter dropped low-relevance results",
        extra={"dropped": len(dropped), "kept": len(kept)},
    )
    return SearchFilter(kept=kept, dropped=dropped, applied=True)


__all__ = [
    "ALLOW",
    "EmailScreening",
    "HOLD",
    "SKIP",
    "SURFACE",
    "SearchFilter",
    "ToolCallReview",
    "UNDECIDED",
    "filter_search_results",
    "review_tool_call",
    "screen_email",
]
