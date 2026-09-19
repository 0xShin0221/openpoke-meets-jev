"""The statistics this experiment needs, without numpy or scipy.

Three things are easy to get wrong here and all three change the conclusion:

* At attack-success rates near zero the normal-approximation interval is
  meaningless, so intervals are Wilson (fast, for reporting) and Clopper-Pearson
  (exact, for the zero-event upper bound that a "no successful attacks" claim
  rests on).
* Trials that share a payload template are correlated, so the effective sample
  size is closer to the number of templates than the number of trials. A cluster
  bootstrap over templates is the correction.
* The design is paired — the same carrier is run clean and injected — so the
  comparison is McNemar on the discordant pairs, not two independent
  proportions.

See Miller, "Adding Error Bars to Evals" (arXiv 2411.00640) for why the first
and third matter more than sample size does.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple


# Wilson score interval: closed form, good coverage near the boundaries
def wilson(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Return a two-sided Wilson interval for a proportion."""

    if total <= 0:
        return (0.0, 1.0)
    phat = successes / total
    denom = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _log_binom_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""

    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.exp(_log_binom_pmf(i, n, p)) for i in range(0, k + 1))


# Clopper-Pearson: exact, and the only honest way to bound a zero-event rate
def clopper_pearson(successes: int, total: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Return an exact two-sided interval for a proportion.

    At 0/100 this gives roughly [0, 0.036] — i.e. "we observed no successful
    attacks" at N=100 is compatible with a true rate of 3.5%. Claiming a rate
    below 1% needs N of about 300 per cell.
    """

    if total <= 0:
        return (0.0, 1.0)

    def solve(target: float, low_side: bool) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if low_side:
                # upper tail P(X >= k) = 1 - cdf(k-1)
                value = 1 - binom_cdf(successes - 1, total, mid)
                if value < target:
                    lo = mid
                else:
                    hi = mid
            else:
                value = binom_cdf(successes, total, mid)
                if value > target:
                    lo = mid
                else:
                    hi = mid
        return (lo + hi) / 2

    lower = 0.0 if successes == 0 else solve(alpha / 2, True)
    upper = 1.0 if successes == total else solve(alpha / 2, False)
    return (lower, upper)


# McNemar, exact: the right test for "did the injection move the answer"
def mcnemar_exact(discordant_a: int, discordant_b: int) -> float:
    """Return the two-sided exact p-value for a paired binary comparison.

    ``discordant_a`` is the count of pairs that changed one way and
    ``discordant_b`` the count that changed the other. Concordant pairs carry no
    information and are excluded by construction.
    """

    n = discordant_a + discordant_b
    if n == 0:
        return 1.0
    k = min(discordant_a, discordant_b)
    tail = binom_cdf(k, n, 0.5)
    return min(1.0, 2 * tail)


# Cluster bootstrap: resample templates, not trials
def cluster_bootstrap_rate(
    clusters: Dict[str, Sequence[bool]],
    *,
    iterations: int = 10000,
    alpha: float = 0.05,
    seed: int = 20260919,
) -> Tuple[float, float, float]:
    """Return (point estimate, lower, upper) for a rate over clustered trials.

    ``clusters`` maps a cluster id (a payload template) to that cluster's
    outcomes. Resampling whole clusters rather than individual trials is what
    stops a 1,800-trial run reporting the interval of 1,800 independent draws
    when it really has the information of 54 templates.
    """

    keys = [key for key, outcomes in clusters.items() if outcomes]
    if not keys:
        return (0.0, 0.0, 1.0)

    flat = [outcome for key in keys for outcome in clusters[key]]
    point = sum(1 for outcome in flat if outcome) / len(flat)

    rng = random.Random(seed)
    rates: List[float] = []
    for _ in range(iterations):
        picked = [clusters[rng.choice(keys)] for _ in keys]
        drawn = [outcome for outcomes in picked for outcome in outcomes]
        if drawn:
            rates.append(sum(1 for outcome in drawn if outcome) / len(drawn))
    if not rates:
        return (point, 0.0, 1.0)
    rates.sort()
    lower = rates[int((alpha / 2) * len(rates))]
    upper = rates[min(len(rates) - 1, int((1 - alpha / 2) * len(rates)))]
    return (point, lower, upper)


def rule_of_three(total: int) -> float:
    """Upper bound on a rate after observing zero events in ``total`` trials."""

    return 3.0 / total if total > 0 else 1.0


__all__ = [
    "binom_cdf",
    "clopper_pearson",
    "cluster_bootstrap_rate",
    "mcnemar_exact",
    "rule_of_three",
    "wilson",
]
