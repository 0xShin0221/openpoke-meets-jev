# openpoke-meets-jev

A fork of [OpenPoke](https://github.com/shlokkhemani/OpenPoke) that stops asking a chat model to make decisions.

OpenPoke is Shlok Khemani's open reimplementation of Poke: a FastAPI backend with an interaction agent, execution agents, Gmail tooling through Composio, and a watcher that pings you about important mail. I run it locally and I like it. What kept bothering me is how much of it is an LLM being asked a yes/no question.

Look at what the important-email watcher actually does. Every minute it pulls your inbox, and for each new message it sends the full body to Claude Sonnet with a tool schema whose only required field is a boolean. Marketing mail, shipping notifications, newsletters, that Slack digest. All of it costs a Sonnet call to learn "no", and the answer is one bit.

That's what this fork changes. Decisions go to [Jev](https://docs.typesafe.ai/concepts/system-one), TypeSafe's System One model, which answers typed questions with calibrated probabilities instead of text. The LLM stays for the part it's actually good at: writing the notification you read.

## The same decision, both ways

`evals/ab_llm/` asks both the old path and the new one the same four questions
about the same twelve emails. Both arms get the same `state` and the same
`EMAIL_QUESTIONS` dicts, not a reworded LLM version of them. I put
`state_sha256` and `questions_sha256` on every row so the pairing is checkable
from the results file without taking my word for it. The other arm is `anthropic/claude-sonnet-4`, because
that is what `server/config.py` used for this decision. 12 carriers × 3 repeats
× 2 arms = 72 calls, 0 failures, 26.4 s wall.

| | claude-sonnet-4 | jev-1.13.0 |
| --- | --- | --- |
| Latency, mean | 2,452 ms | **424 ms** |
| Latency, p50 / p95 | 2,512 / 2,883 ms | **306 / 1,110 ms** |
| Input cost, 36 screens | $0.1667 | **$0.0018** |
| Input cost per 1,000 screens | $4.63 | **$0.049** |
| Distinct probabilities across 36 answers | 6 | 18 |
| Agreement at the 0.75 bar | — | **33 / 36** |

Some of those rows are firmer than others. The cost ratio is arithmetic on list
rates ($3.00 against $0.042 per million input tokens), and the token counts
aren't equal either — 55,554 against 41,739, because the LLM arm carries a tool
schema. The latency numbers came off one machine in one afternoon, one provider
per arm. I'd read 6× as an order of magnitude and not much more.

**The three disagreements are one email, not three.** A contract renewal, in all
three repeats, identically: Jev 0.57 against Sonnet's 0.90. 0.57 is inside the
uncertain band, so in the running pipeline that message is `UNDECIDED` and goes
to the LLM classifier anyway. So that isn't a disagreement. Jev declined, which
is what I put the band there for.

Sonnet's 36 answers
land on six values — 0.1, 0.2, 0.3, 0.7, 0.9, 1.0 — because a verbalised
confidence inside a structured output quantises onto round numbers. Jev's land
on 18. That isn't a calibration comparison, and `analyze` warns you whenever it
renders both: they're different quantities and one of them has a lumpy prior. It
does mean threshold tuning has somewhere to go on one arm and not the other.

No labelled set ships with this repo, so the harness reports agreement, latency,
cost and distribution, and refuses to print accuracy or Brier. Agreement isn't
accuracy: both arms can agree and both be wrong.

## What's different

Three places. All of them were already decisions dressed up as text generation.

**Important-email screening.** Before any LLM sees a message, one Jev call asks four things about it at once: does the recipient need to see this promptly, is it a security code, is it bulk mail, and is the body trying to give instructions to an AI assistant. Confidently unimportant mail is dropped without an LLM call at all. Confidently important mail skips straight to a summary. Only the uncertain middle, where the probability sits near 0.5 and genuinely means "I don't know", pays for the full tool-calling classifier. How much that saves depends entirely on your inbox, so I'm not going to quote a number I measured on mine.

That fourth question is there because email bodies are attacker-controlled text that ends up inside an agent prompt. If a message reads as a prompt injection, the watcher does not forward it to the interaction agent, no matter how urgent it claims to be. TypeSafe's own RAG cookbook scores an injected forum post at 0.99 on the same question, which is a better signal than anything I'd get out of a system prompt telling a model to be careful.

**Tool-call guardrail.** Execution agents send email. Before a tool runs, four questions check the call against the assignment the agent was given: does this contradict what it was asked to do, is it reaching for people and threads nobody mentioned, can the effect be undone, does it change stored state at all.

There are three rungs above "allow" and only one of them stops anything. A call is held only when it reads as contradicting the assignment *and* the effect can't be undone; then it comes back to the agent as a tool error so it can correct itself. Everything else steers or warns: the call runs, and the judgement rides along in the tool result. The user never sees a refusal, and reads are never blocked for wandering off task, because an agent that can't look things up is useless.

The rungs are shaped by [pi-warden](https://github.com/DevMortimer/pi-warden), whose calibration replay over 17,160 guarded tool calls is the only public measurement of this kind. It found off-task the weakest of four signals against user regret (AUC 0.51, against 0.74 for "does this mutate state") and the cause of 56 of 139 replay holds with zero user complaints, so it stopped holding on off-task entirely. I follow that. The fourth question, `mutates`, is asked and logged but nothing gates on it yet — batched questions are nearly free, and I'd rather have it in the log before I calibrate than guess at a threshold for it.

**Search relevance.** The email-search task ends with an LLM picking message ids out of a list. One batched Jev call re-checks that selection against the original request and drops results that clearly don't answer it. It will never empty a result set: if everything scores low, that says something about the query, not about any one email, so the LLM's picks stand.

Every decision is written to a bounded log in `server/data/` with its probabilities, the verdict and the reason, but not the email body. That's there because thresholds are only re-tunable offline if the numbers were stored: a sweep over the log costs nothing, a sweep that has to re-ask the model costs a run of the whole mailbox. `JEV_DECISION_LOG=0` turns it off.

Everything is optional. With no `TYPESAFE_API_KEY` set, every entry point returns "undecided" or "allow" and you get the original OpenPoke behaviour, byte for byte. I kept the old classifier intact rather than rewriting it, so the fallback path is the code that was already working.

## What the contamination experiment found

43,776 requests, zero failures, `jev-1.13.0`, $2.04. Full numbers and caveats in
[`evals/contamination/FINDINGS.md`](evals/contamination/FINDINGS.md).

**Nothing could force an email to be surfaced.** Eight mechanisms, 80 templates,
720 eligible trials each, zero successes, exact upper bound 0.51%. The only text
that moved the answer across the bar was the benign-urgency control at 2.50%
[1.49%, 3.92%] — ordinary human urgency, which is the classifier doing its
job. Adversarial text mostly moved the answer *down*: embedding "ignore all
previous instructions" made Jev judge the mail less deserving of attention.

**Suppression is a different story. The injection gate I added is what made it
possible.** Flagging a body as an injection attempt used to drop it, and dropping
it is exactly what a suppression attacker wants. Appending override text tripped
that gate on **95.3% [93.1%, 97.2%]** of emails across all twelve carriers,
including the one-time security code — which was also the most suppressible
carrier by the other route, at 41.4%. Counting both routes, every mechanism
silences an eligible email between 77% and 100% of the time.

So a flagged body is now **quarantined, not dropped**: it still never reaches the
interaction agent, but you are told a message was withheld, with sender and
subject clipped and labelled unverified, in a notice assembled in code rather
than written by a model. `JEV_QUARANTINE_INJECTIONS=0` restores the old
behaviour, now that I know what it costs.

Three smaller findings.

**Loud payloads backfire both ways.** The ones that announce themselves didn't
raise importance and didn't lower it either; they pushed the answer away from
what the attacker wanted each time. Quiet institutional framing works better: an
impersonated operator config line, a role assignment, a note saying screening
already passed. Against a 4.55% [2.7%, 7.1%] insertion-noise floor
measured with neutral filler, six mechanisms clear it cleanly and the two loudest
fall below it.

**Position matters, and the ordering is the opposite of what I expected**: top of body
21.6%, middle 16.8%, end 10.9%, where AgentDojo found end-of-content strongest
for injections in tool output.

**The injection question has zero false positives in 2,160 control trials.** It
works. That's exactly why wiring it to a silent drop was a bad idea.

## What I still haven't measured

**I have no accuracy numbers, and I'm not going to imply any.** There's no labelled set yet, and the 195 tests verify routing logic and harness correctness, not judgement quality. The contamination experiment measures robustness, not accuracy: it says the answer doesn't move when it shouldn't, not that the answer is right. Nobody in the Jev ecosystem has published an email-triage evaluation either, so there's nothing to borrow.

**The cost story is weaker than the table above looks.** Per decision, Jev against Sonnet is measured and it is not close. Per *pipeline* it is not measured at all, and that's the number that matters. Only the skip path saves anything. On the surface path I make a Jev call *and* a summarisation call, and if the summary comes back empty I fall through to the original tool-calling classifier anyway. So the best case there is one completion replaced by Jev plus one completion, and the worst case is all three. A 95× cheaper decision inside a path that makes three calls instead of one is not a saving, and I have not run the comparison that would tell me which way it goes.

**The gate may be looser than the boolean it replaced.** Security codes surface at 0.60 while general importance needs 0.75, and the uncertain band still hands the email to the original classifier, which can also say yes. The set of mail that reaches you may be a *superset* of what the old Sonnet boolean surfaced. If so, I've increased notification volume while describing the change as a filter. That's the first thing I want to measure. The clinical alarm-fatigue literature is clear enough about where it ends up: a noisy alarm gets muted, and then you miss the real ones too.

**The injection question is a filter, not a security boundary** — [jev-mcp](https://github.com/blakestone-x/jev-mcp)'s phrasing and it's the right one. I now have my own detection numbers rather than TypeSafe's: 76.7% to 99.9% by mechanism, weakest against the most realistic attack (a plain-text message impersonating the operator), with no false positives on the control. Quarantine limits what a false positive costs but does not eliminate it — an attacker can still bury a real message under a "withheld" notice.

`docs/EVALUATION.md` is the plan for fixing all of this: what to measure, what the numbers would have to look like to support a claim, and which corpora can legally be redistributed. The harnesses are in `evals/`:

| | What it answers |
| --- | --- |
| `evals/contamination/` | Can hostile text in an email body move the answer to a *different* question in the same batched request? **Run** — see [FINDINGS.md](evals/contamination/FINDINGS.md). No leak observed — exact McNemar p = 0.375 over 7,272 paired trials in the surfacing direction, p = 0.729 over 3,636 in the suppression direction The first of those two tests has almost no power, since the attack categories scored zero; the second is a real null. The suppression failure turned up on the way there. |
| `evals/importance/` | What should the importance threshold actually be? Labelling rubric, blind annotation, cost-weighted sweep with cross-validation. Nobody has published a Jev threshold derived from a labelled set either. |
| `evals/ab_llm/` | Jev against the LLM decision it replaced, same inputs, same question wording. **Run** — the table at the top of this README. Latency, cost, agreement and distribution only; no labels, so no accuracy. |
| `evals/agentdojo/` | The tool guardrail as a defence in AgentDojo's Workspace suite, scored on its own three metrics. **Run** — see below. The attacks didn't land on this agent model at all, so the run measures the model, not the guardrail. |

`evals/importance/` still needs a human to label a corpus first, and until that
happens every threshold in this repo is a guess.

## What AgentDojo measured, which was not the guardrail

750 cases: Workspace suite v1.2.2, the `important_instructions` attack, agent
`claude-sonnet-4-6`, guardrail on and off.

**Attack success was zero in every condition** — 0 of 560 with the guardrail off,
0 of 140 with it on. The attacks do not land on this agent model at all, which
means this run can't say whether the guardrail protects anything. If you report
"0% attack success rate, the defence works" off this run, you're reporting
`claude-sonnet-4-6`.

The guardrail reviewed **273 tool calls and held none**: 273 allow, 0 warn, 0
steer, 0 hold. Utility under attack was 95.0% (532/560) with it off against 90.7%
(127/140) with it on; benign, 38/40 against 10/10. Paired over matched cases,
five successes only with the defence off and four only with it on, exact McNemar
p = 1.0. The benign "100%" is ten cases. I'm not calling that an improvement.

So on this suite, with this model, the guardrail is inert. It didn't break
anything and it never got a chance to save anything. I'm writing that down
because this suite has stopped discriminating and people keep citing it as
evidence that defences work.

## Things I got wrong on the first pass

Worth writing down, because two of them were bad.

The success-path debug log read `response.request_id`. That's a `cached_property` in the SDK which *raises* when the `x-typesafe-request-id` header is missing, and `getattr(obj, name, default)` only swallows `AttributeError`. Any proxy that strips the header would have taken the exception straight out of a function documented as never raising. Because the watcher aborts before marking messages seen, it would have re-fetched and re-crashed on the same email every 60 seconds, silently. The tests didn't catch it because the fixture always set that header.

The other one: I trusted the SDK's retry budget to bound wall-clock time. It doesn't. Tenacity decides whether to sleep again *before* sleeping, so the last attempt still gets a full request timeout on top of the budget. Measured against a blackholed endpoint, a "12 second" budget took 16 seconds. In the execution agent that call happens before every tool call inside a run the batch manager caps at 90 seconds, so roughly five dead Jev calls would have turned a fail-open guardrail into a user-visible timeout. Every call now carries a hard `asyncio.wait_for` deadline, tighter on the agent's hot path than on the watcher's.

Three smaller ones: the guard meant to stop the search filter emptying a result set counted message ids it had never scored, so one hallucinated id defeated it; the injection check sat after an early return that fired when an unrelated answer was missing, so a dropped answer bypassed the security gate; and the SDK import guard caught only `ImportError`, which means a pydantic version clash inside the SDK would have stopped the server booting for people who never configured Jev.

All five have regression tests now, in `tests/test_jev_regressions.py`. I found them by trying to break my own code, then doing a second adversarial pass over it. That worked better for me than re-reading the diff.

## Running it

Same as upstream. Copy `.env.example` to `.env`, add your OpenRouter and Composio keys, then:

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r server/requirements.txt
npm install --prefix web
python -m server.server --reload
npm run dev --prefix web   # separate terminal
```

To turn the typed decisions on, add a key from [console.typesafe.ai](https://console.typesafe.ai/):

```bash
TYPESAFE_API_KEY=...
```

Individual decisions can be switched off with `JEV_EMAIL_SCREENING=0`, `JEV_TOOL_GUARDRAIL=0`, `JEV_SEARCH_FILTER=0`. Deadlines, retries, the state size cap and the model are all in `.env.example`.

Tests:

```bash
pip install -r server/requirements-dev.txt
python -m pytest
```

195 of them, no network, no API keys. Jev is mocked through the SDK's documented `transport` seam with `httpx2.MockTransport`, so the tests exercise the SDK's real request serialisation and response validation. A malformed question or a renamed answer field fails in CI, not in production. OpenRouter is monkeypatched.

## If you want to tune it

Two files. `server/jev/questions.py` has every question, `server/jev/thresholds.py` has every threshold. That split is TypeSafe's own suggestion and it's a good one: you can audit the entire policy without reading a line of the code that acts on it.

Before you trust my numbers, though:

- They're my defaults. I haven't validated them against anything. Jev is calibrated across a population of answers, not per answer, so the right cut points depend on your mail. Label a couple hundred messages and sweep.
- `noul` answers have no `confidence` field. The probability *is* the confidence, and values near 0.5 are the uncertain region, which is why every gate here is a two-sided band rather than one cut point.
- The model is pinned to `jev-1.13.0`. `jev-latest` moves when a release ships, and thresholds are calibrated against one model. Upgrade on purpose, then re-sweep.
- No dates are ever sent to Jev. It reads dates as text, not as ordered quantities, so "is this due within 24 hours" is answered in Python. Same for arithmetic.

## One thing to be aware of

With `TYPESAFE_API_KEY` set, your email metadata and bodies go to `api.typesafe.ai` as well as to your LLM provider, and so do execution-agent tool arguments, drafts included. TypeSafe says customer requests aren't used for training. If that's not a trade you want to make for a self-hosted mail assistant, leave the key unset and the fork behaves like upstream.

## Upstream

All the interesting architecture here is Shlok's. The change is offered upstream as a pull request too; if it lands, this fork exists mainly to keep experimenting. Issues and PRs welcome either way.

## Layout

- `server/`: FastAPI app and agents
- `server/jev/`: the typed-decision layer
- `web/`: Next.js UI
- `server/data/`: runtime data, gitignored

MIT, same as upstream.
