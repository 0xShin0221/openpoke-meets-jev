# OpenPoke 🌴

OpenPoke is a simplified, open-source take on [Interaction Company’s](https://interaction.co/about) [Poke](https://poke.com/) assistant—built to show how a multi-agent orchestration stack can feel genuinely useful. It keeps the handful of things Poke is great at (email triage, reminders, and persistent agents) while staying easy to spin up locally.

- Multi-agent FastAPI backend that mirrors Poke's interaction/execution split, powered by [OpenRouter](https://openrouter.ai/).
- Gmail tooling via [Composio](https://composio.dev/) for drafting/replying/forwarding without leaving chat.
- Trigger scheduler and background watchers for reminders and "important email" alerts.
- Next.js web UI that proxies everything through the shared `.env`, so plugging in API keys is the only setup.
- Optional typed-decision layer backed by [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), so classification, routing and guardrail judgements are calibrated probabilities instead of another LLM call.

## Requirements
- Python 3.10+
- Node.js 18+
- npm 9+

## Quickstart
1. **Clone and enter the repo.**
   ```bash
   git clone https://github.com/shlokkhemani/OpenPoke
   cd OpenPoke
   ```
2. **Create a shared env file.** Copy the template and open it in your editor:
   ```bash
   cp .env.example .env
   ```
3. **Get your API keys and add them to `.env`:**
   
   **OpenRouter (Required)**
   - Create an account at [openrouter.ai](https://openrouter.ai/)
   - Generate an API key
   - Replace `your_openrouter_api_key_here` with your actual key in `.env`
   
   **Composio (Required for Gmail)**
   - Sign in at [composio.dev](https://composio.dev/)
   - Create an API key
   - Set up Gmail integration and get your auth config ID
   - Replace `your_composio_api_key_here` and `your_gmail_auth_config_id_here` in `.env`
4. **(Required) Create and activate a Python 3.10+ virtualenv:**
   ```bash
   # Ensure you're using Python 3.10+
   python3.10 -m venv .venv
   source .venv/bin/activate
   
   # Verify Python version (should show 3.10+)
   python --version
   ```
   On Windows (PowerShell):
   ```powershell
   # Use Python 3.10+ (adjust path as needed)
   python3.10 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   
   # Verify Python version
   python --version
   ```

5. **Install backend dependencies:**
   ```bash
   pip install -r server/requirements.txt
   ```
6. **Install frontend dependencies:**
   ```bash
   npm install --prefix web
   ```
7. **Start the FastAPI server:**
   ```bash
   python -m server.server --reload
   ```
8. **Start the Next.js app (new terminal):**
   ```bash
   npm run dev --prefix web
   ```
9. **Connect Gmail for email workflows.** With both services running, open [http://localhost:3000](http://localhost:3000), head to *Settings → Gmail*, and complete the Composio OAuth flow. This step is required for email drafting, replies, and the important-email monitor.

The web app proxies API calls to the Python server using the values in `.env`, so keeping both processes running is required for end-to-end flows.

## Typed decisions with Jev (optional)
OpenPoke asks an LLM for a number of judgements that are not really text generation: is this email worth interrupting the user, does this tool call match what the agent was asked to do, does this search result actually answer the question. Those are typed decisions, and [Jev](https://docs.typesafe.ai/concepts/system-one) — TypeSafe's System One model — answers them as calibrated probabilities in roughly a quarter of a second.

Set `TYPESAFE_API_KEY` in `.env` to turn the layer on. **Leave it unset and nothing changes**: every decision falls back to the LLM path that was there before.

What it is wired into:

| Decision | Where | Effect |
| --- | --- | --- |
| Email importance | `server/services/gmail/importance_classifier.py` | Confidently unimportant mail never reaches an LLM; confidently important mail skips straight to summarisation; only the uncertain band pays for the full tool-calling classifier. A `prompt_injection` question also stops a confidently attacker-authored body from being pushed to the interaction agent by the watcher. |
| Tool-call guardrail | `server/agents/execution_agent/runtime.py` | Before an irreversible Gmail tool runs, three nouls check the call against the agent's assignment. A held call is handed back to the agent as a tool error so it can correct itself — it is never surfaced to the user as a refusal. |
| Search relevance | `server/agents/execution_agent/tasks/search_email/tool.py` | Verifies the results the search LLM selected, dropping ones that do not answer the request. Never empties a result set. |

Every question lives in `server/jev/questions.py` and every threshold in `server/jev/thresholds.py`, so the whole policy can be reviewed in two files. Notes worth reading before tuning them:

- **Thresholds are defaults, not truths.** Jev is calibrated across a population of answers rather than per answer, so sweep these against your own labelled mail before trusting them.
- **`noul` answers carry no `confidence` field.** The probability is the signal and values near `0.5` are the uncertain region, which is why each gate is a two-sided band.
- **The model is pinned** (`jev-1.13.0`). `jev-latest` moves when a release ships, and the thresholds are calibrated against one model.
- **Dates are never sent to Jev.** jev-1.13 reads dates as text rather than as ordered quantities, so email age and schedule reasoning stays in Python.
- **Failure is fail-open, and bounded.** A Jev outage, rate limit, timeout or malformed response degrades OpenPoke to its previous behaviour. Every call carries a hard wall-clock deadline (`JEV_DEADLINE_SECONDS`, and a tighter `JEV_GUARDRAIL_DEADLINE_SECONDS` on the agent's hot path) because the SDK's retry budget is checked before it sleeps again and so does not bound wall time on its own.
- **State size is capped, not trusted.** Bodies are clipped to `JEV_STATE_CHAR_BUDGET` and the search filter looks at at most `JEV_SEARCH_MAX_CANDIDATES` results, since the candidate count is chosen by an LLM.
- **Where your mail goes.** With `TYPESAFE_API_KEY` set, email metadata and bodies — and execution-agent tool arguments, including draft text — are sent to `api.typesafe.ai` as well as to your LLM provider. TypeSafe states customer requests are not used for training; if that trade is not one you want, leave the key unset.

## Tests
```bash
pip install -r server/requirements-dev.txt
python -m pytest
```
The suite mocks both models — OpenRouter through `monkeypatch`, Jev through the SDK's `transport` seam with `httpx2.MockTransport` — so it needs no API keys and makes no network calls.

## Project Layout
- `server/` – FastAPI application and agents
- `server/jev/` – optional typed-decision layer (questions, thresholds, decisions)
- `web/` – Next.js app
- `server/data/` – runtime data (ignored by git)

## License
MIT — see [LICENSE](LICENSE).
