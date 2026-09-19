# Importance: labelling, and choosing a threshold honestly

`server/jev/thresholds.py` ships `EMAIL_IMPORTANT_HIGH = 0.75`. I picked that
number by eye. This is the machinery for replacing it with one that was
measured, and for saying out loud how much the measurement is worth.

As far as I can find, **nobody has published a Jev threshold derived by sweeping
a labelled set.** Email spam and phishing are well covered
([jev-spam-eval](https://github.com/bitnovus/jev-spam-eval) is excellent), and
everyone — TypeSafe's docs, its official skill, every project in the awesome
lists — says "tune against your own data". Then nobody shows the tuning. That is
the gap this fills.

## Why there is no public corpus to borrow

Importance is a property of the **(message, recipient, moment)** triple, not of
the message. A stranger cannot annotate it from outside, which is why every
public email set stops at spam and phishing. The one research artifact that
exists — [ACL 2015](https://aclanthology.org/P15-2107.pdf), 2,250 Enron emails,
three importance levels, five annotators each, Kappa 0.64–0.80 by tier — has no
release URL that I could find.

So the recipient is published instead: `evals/contamination/persona.py`. Labels
are meaningless without it, and publishing it is what makes them reproducible.

## Running it

```bash
# 1. Label, blind to any score. Read rubric.md first.
python -m evals.importance.annotate --out evals/importance/labels.json --annotator you

# 2. Double-label a subset with a second person, then check agreement.
python -m evals.importance.annotate --out labels-b.json --annotator them --shuffle-seed 7
python -m evals.importance.annotate --agreement labels.json labels-b.json

# 3. Score with Jev. Costs money; cached and resumable.
TYPESAFE_API_KEY=... python -m evals.importance.sweep score --repeats 3

# 4. Sweep. Free, and re-runnable at any cost ratio forever.
python -m evals.importance.sweep analyze --cost-ratio 20
```

Steps 3 and 4 are separate on purpose. Scoring the 40-item corpus three times is
120 calls, well under a cent. Once the probabilities are stored, every later
question — a different threshold, a different cost ratio, a different metric —
is answered offline for nothing. That separation is the argument for storing
probabilities rather than verdicts, in the evaluation and in the server alike.

## The metric that actually decides the threshold

**Cost-weighted expected loss.** The two error types do not cost the same: a
missed important email is silent and can cost hours, a false interruption costs
seconds of attention and a little trust. `--cost-ratio` is the exchange rate —
"I would accept 20 extra interruptions a week to catch one more important
email" is 20. Pre-register it before looking at any score, because it decides
the answer.

On a *calibrated* probability the optimal cut is `1 / (1 + ratio)`, which at 20:1
is about **0.048** — a long way below the 0.75 this repo ships. That gap is
either evidence the shipped threshold is far too tight, or evidence the
probabilities are not calibrated for this task. Which one it is is exactly what
the calibration section measures, and it is why calibration is not an academic
addendum here.

Everything else is reported to say how much to trust that number: precision and
recall at both thresholds, a reliability diagram in quantile bins with per-bin
Wilson intervals, Brier against a base-rate baseline, ECE with its binning
stated (never as a headline), per-stratum surfacing rates, and a repeatability
check.

## Repeatability, which is TypeSafe's own flagship metric

Two of the eight official cookbooks are about consistency, and the noul one
concedes the failure case honestly: an estimate ranging 0.43–0.53 across
identical inputs, straddling the decision line. `--repeats 3` exists for that.
The analysis reports how many items **straddle the threshold across repeated
calls on identical state**, and names them. A gate whose answer depends on which
call you happened to make is not a gate, and those items are the ones to inspect
by hand.

## Limits of the evidence

Stated here rather than left for a reader to discover.

- **40 synthetic items is a mechanism check, not a mailbox.** It supports a
  coarse five-bin reliability diagram and a direction. It does not support a
  precision figure anyone should quote. Point `--private` at your own labelled
  mail for that; the analysis reports the two sources separately because a
  number from one does not transfer to the other, and the private file is never
  committed.
- **Recall on the rare class is the binding constraint.** At a 15% base rate,
  200 items gives ~30 positives and recall carries an interval of roughly ±11
  points; at 5% it is unmeasurable. Stratify and over-sample the positive class,
  then weight back. The analysis prints the exact interval on recall so this
  cannot be glossed over.
- **Tuning and reporting on the same items overstates performance by several
  points.** The in-sample optimum is reported *labelled as in-sample*; the
  number to quote is the repeated stratified k-fold cross-validated loss.
- **A single tuned threshold to two decimals is false precision.** A plateau of
  near-optimal cuts is reported instead.
- **Your own label noise bounds every figure here.** Report Cohen's kappa before
  reporting accuracy. Below about 0.7 the rubric is the problem; fix it and
  re-label rather than pressing on.
- **Synthetic mail is stylistically too clean** and will flatter the classifier.
  Half the corpus is deliberately the hard cases — urgent wording on unimportant
  mail, quiet wording on critical mail — which helps, but does not fix it.
- **Pinned to `jev-1.13.0`.** Thresholds are calibrated against one model; an
  alias moves when a release ships and every number here would need re-sweeping.
