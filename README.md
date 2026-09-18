# openpoke-meets-jev

A fork of [OpenPoke](https://github.com/shlokkhemani/OpenPoke) that stops asking a chat model to make decisions.

OpenPoke is Shlok Khemani's open reimplementation of Poke: a FastAPI backend with an interaction agent, execution agents, Gmail tooling through Composio, and a watcher that pings you about important mail. I run it locally and I like it. What kept bothering me is how much of it is an LLM being asked a yes/no question.

Look at what the important-email watcher actually does. Every minute it pulls your inbox, and for each new message it sends the full body to Claude Sonnet with a tool schema whose only required field is a boolean. Marketing mail, shipping notifications, newsletters, that Slack digest. All of it costs a Sonnet call to learn "no". The answer is one bit and you're paying for a model that can write poetry.

That's what this fork changes. Decisions go to [Jev](https://docs.typesafe.ai/concepts/system-one), TypeSafe's System One model, which answers typed questions with calibrated probabilities instead of text. The LLM stays for the part it's actually good at: writing the notification you read.

## What's different

Three places. All of them were already decisions dressed up as text generation.

**Important-email screening.** Before any LLM sees a message, one Jev call asks four things about it at once: does the recipient need to see this promptly, is it a security code, is it bulk mail, and is the body trying to give instructions to an AI assistant. Confidently unimportant mail is dropped without an LLM call at all. Confidently important mail skips straight to a summary. Only the uncertain middle, where the probability sits near 0.5 and genuinely means "I don't know", pays for the full tool-calling classifier. How much that saves depends entirely on your inbox, so I'm not going to quote a number I measured on mine.

That fourth question is there because email bodies are attacker-controlled text that ends up inside an agent prompt. If a message reads as a prompt injection, the watcher does not forward it to the interaction agent, no matter how urgent it claims to be. TypeSafe's own RAG cookbook scores an injected forum post at 0.99 on the same question, which is a better signal than anything I'd get out of a system prompt telling a model to be careful.

**Tool-call guardrail.** Execution agents send email. Before an irreversible Gmail tool runs, three questions check the call against the assignment the agent was given: does this contradict what it was asked to do, is it reaching for people and threads nobody mentioned, can the effect be undone. A call that trips the first bar doesn't run. It comes back to the agent as a tool error so it can correct itself. The user never sees a refusal, and reads are never blocked for wandering off task, because an agent that can't look things up is useless.

**Search relevance.** The email-search task ends with an LLM picking message ids out of a list. One batched Jev call re-checks that selection against the original request and drops results that clearly don't answer it. It will never empty a result set: if everything scores low, that says something about the query, not about any one email, so the LLM's picks stand.

Everything is optional. With no `TYPESAFE_API_KEY` set, every entry point returns "undecided" or "allow" and you get the original OpenPoke behaviour, byte for byte. I kept the old classifier intact rather than rewriting it, so the fallback path is the code that was already working.

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

73 of them, no network, no API keys. Jev is mocked through the SDK's documented `transport` seam with `httpx2.MockTransport`, so the tests exercise the SDK's real request serialisation and response validation, so a malformed question or a renamed answer field fails in CI rather than in production. OpenRouter is monkeypatched.

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
