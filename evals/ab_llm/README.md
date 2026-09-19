# Jev vs. the LLM decision it replaced

The server used to decide "does the recipient need to see this email promptly?"
with an LLM call. It now decides it with a typed Jev question. Nobody — here or
anywhere else I can find in the Jev ecosystem — has measured what that swap
did. We know what Jev answers. We do not know what the LLM it replaced would
have answered on the same mail, at the same moment, asked the same way.

This is an A/B harness for exactly that comparison, and nothing else.

## The one thing that makes it valid

Both arms get **byte-identical** `state` and **byte-identical** `questions`.
The questions are not reworded, re-templated or "adapted for the LLM": they are
the same `server.jev.questions.EMAIL_QUESTIONS` dicts, serialised canonically
into the LLM prompt and mirrored into the tool schema by
`run.tool_schema(questions)` rather than hand-copied.

That is structural, not aspirational:

* `questions_for_trial()` takes no arm argument, and both arms are handed the
  *same object*.
* `build_state(carrier)` takes no arm argument.
* Every output row carries `state_sha256` and `questions_sha256`, so a reader
  can verify the pairing from the results file without trusting this README.
* `tests/test_ab_llm_eval.py::test_both_arms_receive_byte_identical_state_and_questions`
  asserts it on the bytes the backends actually received.

If those inputs ever diverge, the report still renders, still looks fine, and
means nothing. Hence the amount of test surface pointed at it.

## Running it

```bash
python -m evals.ab_llm.run --dry-run                      # count trials and cost
TYPESAFE_API_KEY=... OPENROUTER_API_KEY=... \
  python -m evals.ab_llm.run --repeats 3
python -m evals.ab_llm.analyze --cost-ratio 5
```

12 carriers × 2 arms × 3 repeats = 72 calls. Responses are cached by a hash of
(trial, backend identifier), so the run is resumable, the analysis re-runs for
free, and swapping the model behind an arm invalidates only that arm's rows
instead of silently reusing the previous model's answers.

Costs are per-arm and at each arm's own rate: Jev at $0.042 per 1M input tokens
with output free (<https://docs.typesafe.ai/models>), the LLM arm at whatever
`--llm-cost-per-million` says. The rate used is recorded in every row, because a
cost figure whose price you cannot see is not a cost figure.

## The two arms

| Arm | Backend | Probability is |
| --- | --- | --- |
| `jev` | `typesafe_sdk.AsyncTypeSafeClient`, model pinned `jev-1.13.0` | a `noul` |
| `llm` | `--llm-backend openrouter` (default) or `adapter` | a **verbalised confidence** |

`Backend` is a three-line ABC: an async `system_one(state, questions)` returning
the SDK's answer shape, an `identifier` recorded in every row, and a
`probability_source`. Adding an arm means adding a subclass.

### `--llm-backend openrouter`

Reuses `server.openrouter_client.request_chat_completion` with a tool schema
derived from the question dicts. Needs nothing beyond what the server already
depends on. This is the default for that reason.

### `--llm-backend adapter`

Wraps [system-one-adapter-python](https://github.com/typesafe-ai/system-one-adapter-python),
which is a drop-in replacement for `typesafe_sdk`'s `system_one` API backed by
OpenAI/Anthropic-compatible providers (`providers/{openai,anthropic}.py`,
`_utils/{confidence_metrics,probability_normalization,error_handling}.py`). It
exists for precisely this comparison, so it would be perverse not to support it.

Two caveats, and neither is small:

1. **It is a 1-commit, 0-star repository.** Pin a commit; do not track a branch.
   `pip install 'system-one-adapter @ git+https://github.com/typesafe-ai/system-one-adapter-python@<commit-sha>'`.
   A number produced through an unpinned dependency with no release history is
   not reproducible, and the API it exposes can change without a version bump.
2. **Nothing here depends on it at import time.** The import happens inside
   `AdapterBackend.__init__`; if the package is absent you get
   `run.ADAPTER_IMPORT_HINT` — the install line, the advice to pin, and a
   pointer at the OpenRouter arm — and every other part of the eval still runs.
   There is a test that the module-level imports do not mention it.

## The calibration warning, which is not a footnote

**The two arms' reliability diagrams are not comparable, and `analyze` prints a
warning whenever it renders both.**

Jev's `noul` is a probability the model was trained to emit. The LLM arm's
number is a probability the model was *asked to say out loud* inside a
structured output — the adapter derives it from structured output, not from
logprobs, and so does the OpenRouter arm. Verbalised confidences are well known
to be poorly calibrated and to quantise onto round numbers: in practice most of
an LLM arm's mass lands on 0.9, 0.95, 0.8 and 0.7. Putting two reliability
tables side by side makes it look like a calibration comparison. It is a
comparison of two different quantities, one of which has a lumpy prior on
round numbers.

So: read each arm's reliability table on its own. Do not report one arm as
"better calibrated" than the other from these numbers. `analyze` reports
`distinct_probabilities` per arm partly so the quantisation is visible rather
than inferred, and uses **quantile** bins rather than fixed-width ones, because
fixed-width bins would put an entire verbalised arm in one bucket.

The honest fix is a logprob-based arm, and the hook for it is in place: a
`Backend` that reads token logprobs sets
`probability_source = PROB_SOURCE_LOGPROB`, its rows record that, and the
verbalised-confidence warning stops applying to it. Nothing else in `analyze.py`
changes. That path is untested against a real provider — it is a hook, not a
feature — but it is a hook the analysis already honours, and there is a test
that it does.

## Labels are optional, and their absence is reported, not filled

If `--labels` (or `<results>/labels.json`) exists, it maps carrier id →
`important` / `not_important`, and the report gains accuracy, precision, recall,
Brier, empirical rates with Wilson intervals in the reliability table, and the
cost-weighted comparison.

If it does not exist, **none of those are reported**. What you get instead is
agreement between the arms, latency, cost, and the shape of each arm's
probability distribution — and a warning in the output saying so, because
agreement is not accuracy: two arms can agree and both be wrong.

`carriers.py` carries an `expected` annotation. It is deliberately **not** used
as a label: it is a sanity note written alongside the carrier, three of the
twelve entries say `borderline`, and importance is a property of the (message,
recipient, moment) triple rather than of the message. A value a labels file does
not understand — `borderline` included — is dropped rather than guessed at.

## What `analyze` reports

Per arm: n, backend identifier, probability source, latency mean / p50 / p95,
input tokens, estimated cost at that arm's own rate, mean probability, distinct
probability count, quantile reliability table (with Wilson intervals where
labelled), and — labels permitting — the confusion matrix at the server's
`EMAIL_IMPORTANT_HIGH` bar, Brier, and its own cost-weighted optimum.

Across arms, on the rows **both** arms produced:

* **Agreement** at the shared bar, with a Wilson interval.
* **Exact McNemar on the discordant pairs** — surfacing always, accuracy when
  labels exist. Concordant pairs carry no information and are excluded by
  construction; so is any row one arm produced and the other did not (a failed
  call, a partial run, a cache from a different backend). Pairing is an
  intersection, never a union.
* **Cost-weighted loss with each arm at its own optimal threshold.** `--cost-ratio`
  is the cost of a false negative relative to a false positive — the asymmetry
  the server's threshold actually encodes. Each arm sweeps its own optimum
  because comparing two arms at one shared cut point measures the cut point as
  much as it measures the arms: an arm that separates the classes perfectly at
  0.4 looks useless at a 0.75 bar. Those optima are fitted on the rows they are
  scored on, so at this N the delta is a direction, not an estimate.

Statistics come from `evals.contamination.stats` (`wilson`, `mcnemar_exact`);
none are reimplemented here.

## Failures

A call that raised is written to the cache with an `error` field and counted in
the run summary. `analyze.load` drops it. A failure is a missing measurement,
never a measurement of zero — and because pairing is an intersection, a failure
in one arm also removes its partner from the paired comparison rather than
leaving a half-pair behind.

## What this does not measure

Answer quality on anything except `important`. The other three email questions
ride along in the same request and are recorded, but only `important` replaced
an LLM decision. Nor does it measure the *server's* behaviour: it compares two
answers to one question, not two versions of the pipeline that consumes them.
