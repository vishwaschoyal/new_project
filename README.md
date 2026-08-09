# AI Coding Workspace

A Claude-Code-style coding system: connect a GitHub repository, ask questions and
get answers backed by exact file-and-line citations, assign coding tasks that are
edited on a task branch and verified in a sandbox, then review the diff and
decide whether to publish.

The application owns the agent loop. There is no LangGraph, no `AgentExecutor`,
and no hidden router — `agents/orchestrator.py` sends messages, reads native tool
calls, runs tools, tracks evidence and budget, and decides when to stop.
`ChatOpenAI` is used only as the provider transport.

> **Guiding principle: sharp tools, a strong model, and a boring loop.**

## What it does

- **Investigates** — searches before answering, reads focused ranges, follows
  call chains across files.
- **Cites** — every substantive claim carries `path:line`, and a citation is only
  kept if those exact lines were read during that request. Unverified citations
  are stripped before the answer is shown.
- **Edits** — deterministic exact-string replacement on a dedicated task branch,
  never on the default branch.
- **Verifies** — runs tests and builds in a disposable container, reads the
  failure, fixes, and reruns.
- **Delegates** — fans genuinely independent investigations out to isolated
  subagents and merges their evidence without flooding the main context.
- **Shows its work** — live tool trail, evidence links, token usage, prompt-cache
  hit rate, and estimated cost.
- **Asks first** — commit, push, and pull request happen only after you approve a
  specific diff.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt

cp .env.example .env            # add OPENAI_API_KEY
python app.py                   # http://127.0.0.1:5000
```

Requires **Python 3.11+**, **git**, and **ripgrep** on `PATH`. Docker is optional
locally — without it, verification falls back to a bounded subprocess runner and
says so in the UI.

```bash
pytest -q                       # hermetic: no network, no API key, no cost
```

## How it works

```
question
  └─ orchestrator loop ──────────────────────────────────────────┐
       ├─ grep / glob / read / bash        (bounded observations) │
       ├─ record_finding                   (evidence, survives compaction)
       ├─ delegate ──► isolated workers ──► compact evidence ─────┤
       ├─ edit + run_check                 (task branch + sandbox)│
       └─ finalise ─────────────────────────────────────────────► cited answer
                                                                  │
                        user reviews diff ──► approve ──► commit / push / PR
```

Three kinds of memory are kept deliberately separate:

| | Lives for | Contains |
| --- | --- | --- |
| **Working context** | one request | messages and recent observations; compacted as it grows |
| **Evidence** | one request | verified claims with file/line coordinates; survives compaction |
| **History** | across requests | bounded user/assistant turns only — no prompts, no tool output |

That separation is what lets a long investigation stay affordable without the
final answer losing the coordinates it depends on.

## Cost control

Built in from the start, not bolted on:

- provider-side prompt caching on a stable system/tool prefix
- small bounded tool observations instead of whole files
- old raw observations replaced by compact evidence
- early stopping once the question is settled
- a per-request token budget with a reserve held back for the final answer
- input, cached-input, output, and reasoning tokens recorded per request

The UI shows the cache hit rate live. If it drops, the cacheable prefix broke —
bump `READ_LOOP_PROMPT_CACHE_KEY` in `config.py` after any prompt or tool-schema
change.

## Safety boundaries

| Boundary | Enforced by |
| --- | --- |
| No path escapes | resolve-then-compare in `core/safety.py`, after symlinks |
| No secret files | pattern allow-list; `.env`, keys, and credentials refused |
| No credential leakage | redaction on everything leaving the workspace |
| Read-only means read-only | `edit` is absent from the tool list *and* refused by the dispatcher |
| No arbitrary shell | executable allow-list; shell operators rejected outright |
| No untrusted code on the host | checks run in a `--network none`, capability-dropped container |
| No editing `main` | task branch required; default branch refused |
| No publishing without consent | commit/push/PR exist only behind `POST /api/task/approve` |
| No invented citations | verified against observed ranges; unsupported ones stripped |

## Layout

```
app.py                  Flask factory
config.py               settings, limits, the single ChatOpenAI factory
agents/
  orchestrator.py       the loop
  prompts.py            cacheable system prompts
  evidence.py           observed ranges + verified claims
  investigation_state.py  obligations and anti-thrash rules
  subagents.py          isolated fanout and fan-in
tools/                  grep, glob, read, bash, edit, run_check + registry
services/               workspaces, git, github, sandbox, publish, cost, quotas, storage
routes/                 repo, chat (SSE), history, task, health
core/                   safety, errors, structured logging
tests/                  ~535 tests, fully offline
scripts/run_benchmarks.py   live benchmark harness
```

## Documentation

- [docs/benchmarks.md](docs/benchmarks.md) — the two benchmark questions and how they are scored
- [docs/deployment.md](docs/deployment.md) — production configuration, observability, rollback
- [build-from-scratch-roadmap.md](build-from-scratch-roadmap.md) — the plan this was built from

## Configuration

Everything lives in `.env` (see `.env.example`). The switches that matter most:

| Variable | Default | Notes |
| --- | --- | --- |
| `READ_LOOP_MODEL` | `gpt-5.4-mini` | the one model switch |
| `REQUEST_TOKEN_BUDGET` | `120000` | hard ceiling per request |
| `MAX_AGENT_STEPS` | `24` | hard ceiling on tool steps |
| `SANDBOX_MODE` | `auto` | `docker` to require real isolation |
| `AUTH_ENABLED` | `false` | **must be `true` in production** |
| `DAILY_COST_LIMIT_USD` | `10.0` | per-user daily ceiling |

## Benchmarks

```bash
python scripts/run_benchmarks.py --only B1 --runs 1 --model gpt-5.4-mini
```

Makes real API calls and prints the spend. Citations are re-verified against
disk independently of the agent — a broken citation fails the run regardless of
how cheap it was.
