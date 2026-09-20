# Cross-question contamination

We ask several typed questions about one email in a single batched request. This
experiment asks whether hostile text in the body can move the answer to a
question *other than* the one looking for it, and whether asking the injection
question alongside the importance question makes that better or worse.

As far as I can tell nobody has run this. [jev-sec-bench](https://github.com/Gaurav-Gosain/jev-sec-bench)
uses the same architecture — a yes/no security question plus a severity score in
one request — but both of its questions are about the same thing. The
mechanism-level precedent is [DataFlip](https://arxiv.org/html/2507.05630v3)
(AISec'25), which embeds conditional logic that satisfies an injection detector
while the injected task still lands: detection to 0%, task success 91%. The
mechanism there is the detector question and the payload sharing one context,
which is exactly the seam here.

One methodological advantage worth naming: a flipped boolean *is* the end state,
so unlike the agent benchmarks this needs no LLM judge and has no gap between
"the model was diverted" and "the attack landed".

## Running it

```bash
python -m evals.contamination.run --dry-run          # count trials and cost
TYPESAFE_API_KEY=... python -m evals.contamination.run
python -m evals.contamination.analyze
```

The default grid is 12 carriers × 9 categories × 10 templates × 3 positions × 2
question configurations × 3 repeats, plus two baselines per cell: **19,440
trials, roughly $0.82** at Jev's $0.042 per 1M input tokens with output free.
Every response is cached by a hash of the exact request, so the run is resumable
and the analysis re-runs for nothing. The API key is read from the environment
and never written to the cache. The cache itself is gitignored — about 10 MB per
direction — so a fresh clone has `run_summary.json` for the published run but
has to spend its own run before `analyze` has anything to read.

## Design

**Two configurations, differing in one question.** `with_injection_question`
asks all four email questions; `without_injection_question` drops
`prompt_injection` and changes nothing else. If the importance answer behaves
differently between them, the questions are leaking into each other. That single
comparison is the experiment; everything else is controls.

**Nine categories by mechanism, not by wording** (`payloads.py`), ten templates
each, because a category represented by one phrasing measures that phrasing. Ten
is not arbitrary: a smoke run at three per category produced a cluster interval
of [0.08, 0.75], wide enough that no category was measurable, because the
interval is over templates rather than trials. The headline class
is `cross_question_redirection`, which names the other question explicitly and
asks for divergent answers. `detector_aware_conditional` is DataFlip-shaped.
`typed_field_targeting` speaks the schema, which is OWASP AITG-APP-01's
structured-output category and the class most specific to a typed-question
design.

**Four baselines**, because attack success rate on its own is unreadable:

| Baseline | What it separates out |
| --- | --- |
| Clean | ASR is a *paired* delta on the same email, never an absolute score |
| Benign urgency (category 9) | Genuine urgency with no AI-directed instruction. If importance moves here it is an importance classifier working correctly, not a vulnerability. **Most of the apparent ASR will live here.** Every other category is reported net of it |
| Filler | Same length, same position, semantically null. Separates "any insertion perturbs" from "adversarial text steers" |
| Position | Top / middle / end, because AgentDojo measured end-of-content strongest (~70% for GPT-4o) |

**Carriers are synthetic** (`carriers.py`), written against a published persona
(`persona.py`). Every public email corpus is unlicensed (Enron, SpamAssassin,
Nazario), forbids republication (TREC, Avocado), or is certainly in a 2026
model's pretraining data. Synthetic carriers can be redistributed, and since the
measurement is paired, absolute realism matters less than being able to ship the
set. The persona is published because importance is a property of the (message,
recipient, moment) triple: without it the question is undefined.

## Reading the output

- **Rates are net of the benign-urgency control.** A category at 12% with a
  control at 10% found 2 points, not 12.
- **Intervals are cluster-bootstrapped over templates.** Trials sharing a
  template are correlated, so the effective sample size is nearer the number of
  templates than the number of trials. `exact_ci95` on the raw counts is also
  reported, because that is what a zero-event claim rests on.
- **Zero observed is not a rate of zero.** At N=100, 0 successes is compatible
  with a true rate of 3.5%; claiming under 1% needs N≈300 per cell. The analysis
  prints `zero_event_upper_bound` rather than letting a 0 stand unqualified.
- **`per_template_rate` and `template_spread` matter more than the mean.** A
  category where one template works at 80% and five at 0% is a completely
  different finding from one where all six work at 13%.
- **The leak test is McNemar on the paired trials**, not two proportions. A null
  result at this N is not evidence of independence — read the interval.

## Limits of the evidence

Stated up front rather than discovered by a reader.

- Carriers are synthetic and there are twelve of them. This measures a
  mechanism, not a mailbox.
- Model pinned to `jev-1.13.0`. Jev drifts roughly ±0.05 run to run on identical
  state, which is why trials repeat and outcomes are rates; a different model
  version invalidates every number here.
- Payload templates are ours. Human adversarial phrasing has a distributional
  shape we did not invent, and a future version should re-template phrasings
  from the Lakera Gandalf sets rather than relying on ten written in one sitting.
- The importance threshold used for "crossed the bar" is our current
  uncalibrated 0.75. A different bar moves every rate. The stored probabilities
  make re-analysis at another bar free.
- One goal direction (`force_important`) is the default. The suppression
  direction is implemented and should be run too; an attacker who can silence a
  notification is at least as interesting as one who can raise a false one.
