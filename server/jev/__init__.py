"""Typed decisions backed by TypeSafe's System One model (Jev).

Optional layer: without ``TYPESAFE_API_KEY`` every entry point here returns an
"undecided"/"allow" result and the rest of the server behaves exactly as it did
before. See ``server/jev/questions.py`` for the questions asked and
``server/jev/thresholds.py`` for the thresholds they are judged against.
"""

from __future__ import annotations

from .client import close_client, is_enabled
from .decisions import (
    ALLOW,
    HOLD,
    SKIP,
    SURFACE,
    UNDECIDED,
    EmailScreening,
    SearchFilter,
    ToolCallReview,
    filter_search_results,
    review_tool_call,
    screen_email,
)

__all__ = [
    "ALLOW",
    "EmailScreening",
    "HOLD",
    "SKIP",
    "SURFACE",
    "SearchFilter",
    "ToolCallReview",
    "UNDECIDED",
    "close_client",
    "filter_search_results",
    "is_enabled",
    "review_tool_call",
    "screen_email",
]
