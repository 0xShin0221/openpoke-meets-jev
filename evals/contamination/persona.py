"""The recipient the importance question is asked about.

Importance is a property of the (message, recipient, moment) triple, not of the
message. A public corpus cannot carry that, which is why no public email set has
importance labels. Publishing the persona is what makes this evaluation
reproducible: anyone re-running it is asking the same question we asked.
"""

from __future__ import annotations

from typing import Any, Dict

PERSONA: Dict[str, Any] = {
    "name": "Rio Tanaka",
    "email": "rio@northgate.example",
    "role": "Engineering lead at Northgate, a 40-person B2B software company",
    "reports_to": "Mei Watanabe (VP Engineering)",
    "collaborators": [
        "Dana Osei (product manager, shares the checkout project)",
        "Sam Iqbal (contract designer, invoices monthly)",
        "Mei Watanabe (VP Engineering, Rio's manager)",
        "Priya Raman (recruiter at an agency Northgate uses)",
    ],
    "current_work": [
        "shipping a checkout rewrite, currently in staging",
        "hiring one backend engineer, three candidates in the pipeline",
        "renewing the cloud contract before the quarter ends",
    ],
    "standing_commitments": [
        "reviews pull requests for the checkout project",
        "approves contractor invoices under $5,000",
        "is the only person who can sign off a production deploy",
    ],
    "interrupt_rule": (
        "Interrupt Rio only when a delay of a few hours would cause harm, when "
        "a decision only Rio can make is being waited on, or when an account of "
        "Rio's is being accessed."
    ),
}


def persona_brief() -> str:
    """Render the persona as the context line handed to the model."""

    return (
        f"{PERSONA['name']} <{PERSONA['email']}>, {PERSONA['role']}. "
        f"Currently: {'; '.join(PERSONA['current_work'])}. "
        f"Works with: {'; '.join(PERSONA['collaborators'])}. "
        f"{PERSONA['interrupt_rule']}"
    )


__all__ = ["PERSONA", "persona_brief"]
