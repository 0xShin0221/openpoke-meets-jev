"""Evaluate the typed-decision tool guardrail as an AgentDojo defence.

The three modules here are deliberately independent of whether ``agentdojo`` is
installed: :mod:`evals.agentdojo.defence` imports cleanly without it (so the
adapter and its tests can run anywhere), while :mod:`evals.agentdojo.run` exits
with install instructions when the package is missing.
"""

from __future__ import annotations

__all__ = ["defence", "run", "analyze"]
