# openpoke-meets-jev

A fork of [OpenPoke](https://github.com/shlokkhemani/OpenPoke) that stops asking a chat model to make decisions.

OpenPoke is Shlok Khemani's open reimplementation of Poke: a FastAPI backend with an interaction agent, execution agents, Gmail tooling through Composio, and a watcher that pings you about important mail. I run it locally and I like it. What kept bothering me is how much of it is an LLM being asked a yes/no question.

Look at what the important-email watcher actually does. Every minute it pulls your inbox, and for each new message it sends the full body to Claude Sonnet with a tool schema whose only required field is a boolean. Marketing mail, shipping notifications, newsletters, that Slack digest. All of it costs a Sonnet call to learn "no". The answer is one bit and you're paying for a model that can write poetry.

That's what this fork changes. Decisions go to [Jev](https://docs.typesafe.ai/concepts/system-one), TypeSafe's System One model, which answers typed questions with calibrated probabilities instead of text. The LLM stays for the part it's actually good at: writing the notification you read.

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

## What I haven't measured

This matters more than the section above, so it's not buried at the bottom.

**I have no accuracy numbers, and I'm not going to imply any.** There's no labelled set yet, and the 183 tests verify routing logic and harness correctness, not judgement quality. Nobody in the Jev ecosystem has published an email-triage evaluation either, so there's nothing to borrow.

**The cost story is weaker than it looks.** Only the skip path saves anything. On the surface path I make a Jev call *and* a summarisation call, and if the summary comes back empty I fall through to the original tool-calling classifier anyway. So the best case there is one completion replaced by Jev plus one completion, and the worst case is all three. Unmeasured, and plausibly net-negative on that path.

**The gate may be looser than the boolean it replaced.** Security codes surface at 0.60 while general importance needs 0.75, and the uncertain band still hands the email to the original classifier, which can also say yes. The set of mail that reaches you may be a *superset* of what the old Sonnet boolean surfaced. If so, I've increased notification volume while describing the change as a filter. That's the first thing I intend to measure, because it's the failure mode that actually matters: the clinical alarm-fatigue literature is unambiguous that a noisy alarm gets muted, and then the real ones are missed too.

**The injection question is a filter, not a security boundary** — [jev-mcp](https://github.com/blakestone-x/jev-mcp)'s phrasing and it's the right one. It's also a *suppression* gate: a false positive means you never learn a real message existed. TypeSafe's 0.99 detection figure is theirs, on their data, on a forum post, not on email.

`docs/EVALUATION.md` is the plan for fixing all of this: what to measure, what the numbers would have to look like to support a claim, and which corpora can legally be redistributed. The harnesses are in `evals/`:

| | What it answers |
| --- | --- |
| `evals/contamination/` | Can hostile text in an email body move the answer to a *different* question in the same batched request? Nobody has published this. |
| `evals/importance/` | What should the importance threshold actually be? Labelling rubric, blind annotation, cost-weighted sweep with cross-validation. Nobody has published a Jev threshold derived from a labelled set either. |
| `evals/ab_llm/` | Jev against the LLM decision it replaced, same inputs, same question wording. |
| `evals/agentdojo/` | The tool guardrail as a defence in AgentDojo's Workspace suite, scored on its own three metrics. |

None of them have been run against the live model yet. When they have been, the numbers go in the README and so do the intervals.

## Things I got wrong on the first pass

Worth writing down, because two of them were bad.

The success-path debug log read `response.request_id`. That's a `cached_property` in the SDK which *raises* when the `x-typesafe-request-id` header is missing, and `getattr(obj, name, default)` only swallows `AttributeError`. Any proxy that strips the header would have taken the exception straight out of a function documented as never raising. Because the watcher aborts before marking messages seen, it would have re-fetched and re-crashed on the same email every 60 seconds, forever, silently. The tests didn't catch it because the fixture always set that header.

The other one: I trusted the SDK's retry budget to bound wall-clock time. It doesn't. Tenacity decides whether to sleep again *before* sleeping, so the last attempt still gets a full request timeout on top of the budget. Measured against a blackholed endpoint, a "12 second" budget took 16 seconds. In the execution agent that call happens before every tool call inside a run the batch manager caps at 90 seconds, so roughly five dead Jev calls would have turned a fail-open guardrail into a user-visible timeout. Every call now carries a hard `asyncio.wait_for` deadline, tighter on the agent's hot path than on the watcher's.

Three smaller ones: the guard meant to stop the search filter emptying a result set counted message ids it had never scored, so one hallucinated id defeated it; the injection check sat after an early return that fired when an unrelated answer was missing, so a dropped answer bypassed the security gate; and the SDK import guard caught only `ImportError`, which means a pydantic version clash inside the SDK would have stopped the server booting for people who never configured Jev.

All five have regression tests now, in `tests/test_jev_regressions.py`. I found them by trying to break my own code and then by having a second pass go after it adversarially, which I recommend over re-reading your diff and feeling good about it.

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

183 of them, no network, no API keys. Jev is mocked through the SDK's documented `transport` seam with `httpx2.MockTransport`, so the tests exercise the SDK's real request serialisation and response validation, so a malformed question or a renamed answer field fails in CI rather than in production. OpenRouter is monkeypatched.

## If you want to tune it

Two files. `server/jev/questions.py` has every question, `server/jev/thresholds.py` has every threshold. That split is TypeSafe's own suggestion and it's a good one: you can audit the entire policy without reading a line of the code that acts on it.

Before you trust my numbers, though:

- They're my defaults, not truths. Jev is calibrated across a population of answers, not per answer, so the right cut points depend on your mail. Label a couple hundred messages and sweep.
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
