# Build-from-Scratch Roadmap: Claude-Code-Style AI Coding System

## Objective

Build and deploy a complete Claude-Code-style AI coding system that can:

- Connect to and manage GitHub repositories.
- Investigate code and follow execution paths across files and components.
- Answer questions with evidence-backed file-and-line citations.
- Plan and make controlled code edits on a task branch.
- Run tests and builds inside an isolated execution sandbox.
- Read failures, correct its changes, and rerun verification.
- Show the user its investigation trail, evidence, diff, checks, token usage, and cost.
- Commit, push, or open a pull request only after explicit user approval.
- Delegate large independent investigations to isolated subagents and combine their evidence without flooding the main context.
- Operate as a secure, observable, multi-user production application.

The system will be developed in safety-first phases. A reliable read path will provide the evidence and context needed by editing, verification, subagent, and publishing workflows; it is the foundation of the complete product, not the final product boundary.

## Core Technical Direction

The main agent and every subagent will use the same hand-written ReAct-style loop:

```text
User question or coding task
    -> main agent investigates with bounded tools
    -> large independent work may fan out to isolated subagents
    -> verified findings return as compact evidence
    -> for a coding task, create a task branch and apply controlled edits
    -> run tests/builds in a disposable sandbox
    -> inspect failures, fix, and verify again
    -> stream the cited answer or reviewable diff and checks
    -> after explicit approval, commit, push, or open a pull request
```

The model will use native tool calling. LangGraph, `AgentExecutor`, and a hardcoded router will not control the loop. The application will own the messages, evidence, edits, tool execution, verification results, token budget, stopping conditions, worker contexts, and approval boundaries.

**Model-client rule:** all application model calls will use `langchain_openai.ChatOpenAI`. Application code will not instantiate or call `openai.OpenAI`, `openai.AsyncOpenAI`, or another OpenAI SDK client directly. `ChatOpenAI` will be used only as the provider transport and message/tool-call adapter; the application will still own the agent loop.

Guiding principle: **sharp tools, a strong model, and a boring loop.**

## Technologies

| Area | Technology | Purpose |
| --- | --- | --- |
| Language | Python 3 | Backend, agent loop, tools, services, and tests |
| Web backend | Flask and Flask Blueprints | Web application, repository APIs, chat API, and history API |
| Configuration | `python-dotenv` and environment variables | API keys, model selection, logging, and runtime settings |
| Model client | `langchain_openai.ChatOpenAI` | The only application-level interface for model calls, native tools, streaming, usage metadata, and prompt caching |
| LLM provider | OpenAI Responses API through `ChatOpenAI` | Model reasoning and native tool execution without direct SDK calls in application code |
| Message types | `langchain-core` | System, human, assistant, and tool messages; not agent orchestration |
| Token accounting | `tiktoken` | Context estimation and request-budget enforcement |
| HTTP client | `requests` | GitHub API calls such as public profile and repository discovery |
| Repository operations | Git CLI | Clone, inspect, list branches, fetch, and human-requested branch switching |
| Code search | ripgrep (`rg`) | Fast, bounded content search and file discovery |
| Local tooling | `pathlib` and `subprocess` | Safe path resolution and controlled command execution |
| Code editing | Exact unique-string replacement | Make deterministic, reviewable edits while refusing missing or ambiguous targets |
| Execution isolation | Disposable OS/container sandbox | Run repository tests and builds without trusting repository code on the application host |
| Multi-agent execution | Isolated instances of the same agent loop | Investigate independent areas concurrently and return compact evidence to the main agent |
| Version-control workflow | Task branches, Git diffs, commits, and pushes | Keep changes reviewable and prevent direct unapproved publication |
| Collaboration workflow | GitHub pull requests | Publish an approved change for human review and merging |
| Streaming | Server-Sent Events, Python threads, and queues | Send tool progress, answer text, evidence, and cost data to the browser |
| Frontend | HTML5, CSS3, Jinja templates, and vanilla JavaScript | Chat, repository selection, file explorer, citations, and session UI |
| Browser APIs | Fetch, `ReadableStream`, and `localStorage` | Consume SSE responses and remember browser-side recent sessions |
| Frontend libraries | Marked, DOMPurify, and Highlight.js | Render sanitized Markdown and highlighted code |
| Conversation storage | Storage service interface | Begin with bounded process-local storage; replace its internals with a durable product database for deployment |
| Testing | Pytest, `unittest`, mocks, and Flask test client | Unit, API, safety, tool, orchestration, cost, and storage tests |

## Target Architecture

### Web layer

- `app.py` will create the Flask application and register route blueprints.
- `routes/repo_routes.py` will load and inspect GitHub repositories.
- `routes/chat_routes.py` will validate requests and stream the agent run through SSE.
- `routes/history_routes.py` will expose bounded conversation history.
- `templates/index.html`, `static/app.js`, and the CSS files will provide the browser UI.

### Agent layer

- `agents/orchestrator.py` will own the native tool-calling loop.
- The system prompt will require search-first investigation, bounded reads, call-chain tracing, exact citations, honest uncertainty, and early stopping.
- A request-local investigation state will track questions that must be answered, reviewed observations, evidence provenance, and completion conditions.
- A hard step limit and token budget will force graceful finalization instead of an uncontrolled loop.

### Editing and verification layer

- Every coding task will begin from an inspected and evidence-backed understanding of the relevant code.
- The agent will create or use a dedicated task branch before changing files.
- The `edit` tool will perform deterministic exact-string replacements and refuse ambiguous operations.
- Test and build commands will execute only inside a disposable, resource-limited sandbox.
- Failed checks will return bounded diagnostics to the loop so the agent can investigate, fix, and verify again.
- The backend will generate a reviewable diff and wait for explicit approval before commit, push, or pull-request creation.

### Subagent layer

- The main loop will divide only genuinely independent, context-heavy work into scoped investigations.
- Each subagent will receive a fresh context, bounded tools, its own budget, and one clear objective.
- Subagents will return compact findings and exact evidence records rather than their raw message history.
- The main agent will preserve those evidence records during fan-in and use them in planning, editing, verification, or the final answer.
- Subagents may run concurrently to reduce elapsed time, while total token usage will remain visible and budgeted.

### Tool layer

Phase 1 will expose four read-only tools:

| Tool | Responsibility | Bounded result |
| --- | --- | --- |
| `grep` | Search repository contents | Matching `file:line:text` entries |
| `glob` | Find paths by pattern | Sorted repository-relative paths |
| `read` | Inspect a focused line range | Line-numbered text only |
| `bash` | Perform approved read-only Git/search inspection | Limited stdout and stderr |

Phase 2 will add:

| Tool | Responsibility | Safety rule |
| --- | --- | --- |
| `edit` | Replace one exact string in one file | Refuse zero matches and ambiguous multiple matches |

All tools will reject workspace escapes, secret files, unsafe shell syntax, unapproved executables, oversized output, and timeouts.

### Memory and cost model

The application will keep three forms of memory separate:

- **Working context:** request-local model messages and recent observations. It will be compacted during the loop and discarded after the answer.
- **Evidence:** request-local verified claims with file and line coordinates. It will survive context trimming and power the final cited answer.
- **Conversation history:** bounded user/assistant messages stored by `thread_id` so follow-up questions retain context.

Cost control will be part of the first implementation:

- Enable provider-side prompt caching for the stable system/tool prefix.
- Return small tool observations instead of entire files.
- Replace old raw observations with compact evidence and active investigation state.
- Stop as soon as the question is sufficiently answered.
- Record input, cached input, output, reasoning tokens, and estimated cost for every request.

## Roadmap

### Phase 0 — Project safety and baseline

1. Create a clean Git repository and commit the starting point.
2. Protect the original implementation with an archive branch before any rebuild work.
3. Record two benchmark questions:
   - a focused repository question for cost and citation accuracy;
   - a tracing question that requires following a route through several functions.
4. Store the expected answer quality, token count, cost, and execution trail for comparison.

**Exit criteria**

- The starting state can be restored from Git.
- The benchmark questions and evaluation method are documented.

### Phase 1 — Flask and repository foundation

1. Create the Flask application factory and route blueprints.
2. Add environment-based configuration and one model setting: `READ_LOOP_MODEL`.
3. Build thread-scoped application state and the conversation storage interface.
4. Add GitHub repository/profile URL validation.
5. Clone each repository into a managed workspace owned by its thread.
6. Add repository status, file explorer, branch listing, safe file viewing, unloading, and human-requested branch switching.
7. Reject path traversal, symlink escapes, secret paths, binary files, oversized files, and unsafe deletion targets.

**Exit criteria**

- A user can connect a GitHub profile, load a repository, browse safe files, switch a clean branch, and unload the workspace.
- One thread cannot access another thread's files.

### Phase 2 — Read-only tools

1. Define JSON schemas for `grep`, `glob`, `read`, and `bash`.
2. Build one tool registry and dispatcher.
3. Stamp all search/read results with repository-relative paths and line numbers.
4. Bound result counts, characters, line ranges, subprocess output, and execution time.
5. Redact credentials and prevent tools from reading known secret paths.
6. Restrict `bash` to explicitly approved read-only commands.
7. Return structured success, error, timing, and truncation metadata.

**Exit criteria**

- Each tool is deterministic, bounded, workspace-confined, and covered by safety tests.
- No Phase 1 tool can modify repository contents.

### Phase 3 — Hand-written agent loop

1. Create the model client with `langchain_openai.ChatOpenAI` and enable native tool calling.
2. Route every exploration, finalization, and subagent model request through that `ChatOpenAI` client; do not add direct OpenAI SDK client calls.
3. Write the repository-investigation system prompt.
4. Build the loop that sends messages, receives tool calls, validates them, runs tools, and appends compact observations.
5. Track investigation obligations so multi-part questions cannot finish after checking only one part.
6. Reject repeated calls, overlapping reads, unsupported citations, malformed state updates, and premature final answers.
7. Capture answer-relevant evidence before old observations leave the working context.
8. Produce a best-effort evidence-based answer when a provider, tool, step, or token limit is reached.
9. Persist only bounded user/assistant conversation messages—not system prompts, raw tool outputs, or request evidence.

**Exit criteria**

- A repository question produces a grounded answer with exact citations.
- Every citation refers to lines actually observed during that request.
- The tool trail and termination reason are inspectable.

### Phase 4 — Cost controls, streaming, and browser experience

1. Keep the system prompt, tool definitions, history, and question as a stable cacheable prefix.
2. Enable OpenAI prompt caching and verify cached-token usage from provider metadata.
3. Estimate the next model call before sending it and reserve tokens for final synthesis.
4. Stream tool-start, tool-end, content, evidence, cost, and completion events through SSE.
5. Build the chat interface, GitHub profile/repository panel, file tree, branch selector, citation links, tool activity trail, and token/cost summary.
6. Render model Markdown through Marked and sanitize it with DOMPurify before inserting it into the page.
7. Preserve recent thread identifiers in `localStorage` and load server-side conversation history when a thread is reopened.

**Exit criteria**

- The UI shows live progress, the final cited answer, evidence links, token usage, cache usage, and estimated cost.
- Dynamic model output is sanitized before rendering.

### Phase 5 — Benchmark and tune

1. Run the focused and tracing benchmark questions repeatedly.
2. Compare answer correctness, citation correctness, tool steps, total tokens, cached tokens, and cost.
3. Tune the system prompt, read ranges, active evidence limits, step limit, and token budget.
4. Confirm that focused questions stop early while tracing questions follow the complete call chain.
5. Add regression tests for every failure discovered during benchmarking.

**Exit criteria**

- Focused questions meet or beat the recorded quality and cost baseline.
- The tracing benchmark follows every material edge and cites the inspected implementation.
- Repeated runs stay inside the configured request budget.

### Phase 6 — Safe editing and verification

1. Run repository code only inside an isolated OS/container sandbox.
2. Add the exact-unique-string `edit` tool.
3. Require a task branch before changing code.
4. Show the proposed diff and keep publishing under explicit user control.
5. After every edit, run the relevant tests or build in the sandbox.
6. If verification fails, inspect the error, correct the change, and rerun the checks.
7. Preserve a complete audit trail of edits, commands, results, and approval events.

**Exit criteria**

- The agent can make a small code change, verify it, and present a reviewable diff without modifying the default branch or publishing automatically.

### Phase 7 — Isolated subagent system

Build subagent fanout as a complete system capability, while invoking it only when a task contains large, genuinely independent investigations.

1. Give each worker one scoped task, a fresh context, and the same bounded tools.
2. Run independent workers concurrently where latency benefits.
3. Return compact evidence records rather than raw worker conversations.
4. Preserve every worker's useful findings during fan-in.
5. Keep the single loop as the default for focused questions and normal call-chain tracing.

Subagents will primarily provide context isolation. Parallelism may reduce elapsed time, but it will not reduce the total tokens required for independent investigations.

**Exit criteria**

- A large multi-area benchmark stays readable in the main context and retains exact evidence from every worker.
- Simple tasks do not pay subagent overhead.

### Phase 8 — Production hardening and deployment

1. Replace process-local thread storage with a durable product database through the existing storage interface.
2. Add authentication, per-user authorization, repository ownership checks, rate limits, and quota metering.
3. Store usage records for billing and operational reporting.
4. Run repository commands and tests in disposable, resource-limited containers.
5. Add structured logs, request IDs, metrics, error reporting, health checks, and alerts.
6. Configure secure secret management, HTTPS, restricted network access, and least-privilege service credentials.
7. Add production timeouts and limits for HTTP requests, model calls, tools, streams, files, tokens, and cost.
8. Package the Flask application for the selected hosting platform and configure the production web server, database, and sandbox workers.
9. Add CI/CD that runs the complete test suite, builds the deployable artifact, deploys to staging, runs smoke tests, and promotes an approved release to production.
10. Deploy the application, monitor real usage and cost, and keep a documented rollback procedure.

**Exit criteria**

- The application is deployed behind HTTPS in a production environment.
- User data and workspaces are isolated and durable.
- Repository execution occurs only in disposable sandboxes.
- Usage, errors, latency, and model cost are observable.
- A failed release can be rolled back safely.

## Validation Strategy

Run validation continuously instead of waiting for deployment:

- Unit tests for tool bounds, secret filtering, path resolution, cost calculations, memory limits, and investigation-state transitions.
- Flask route tests for request validation, repository lifecycle, history, and SSE completion behavior.
- Orchestrator tests with mocked model responses and native tool calls.
- An architecture test that rejects direct OpenAI SDK client imports or construction in application modules.
- Security tests for traversal, symlinks, shell syntax, secret references, unsafe Git operations, binary files, and oversized inputs.
- Browser verification for streaming, Markdown sanitization, citations, history, file navigation, and cost display.
- Benchmark tests for focused lookup, call-chain tracing, budget exhaustion, provider failure, and partial-evidence fallback.
- Staging smoke tests before every production promotion.

## Project Rules

- Search before answering; never rely on assumptions about a repository.
- Use `langchain_openai.ChatOpenAI` for every model call; never use the OpenAI SDK client directly in application code.
- Read focused ranges rather than entire files.
- Follow symbols through their callers and dependencies when the question requires tracing.
- Cite every substantive repository claim with exact observed lines.
- Keep request evidence separate from conversation history.
- Treat prompt caching, bounded observations, context compaction, and early stopping as core architecture.
- Keep Phase 1 read-only until an actual execution sandbox exists.
- Never publish code without explicit user approval.
- Prefer the single agent loop; add fanout only for independent, context-heavy work.
- Build deployment security, observability, quotas, and rollback into the production phase.

## Final Outcome

The deployed product will be a complete AI coding workspace. A user will be able to connect a repository, ask questions, assign coding tasks, watch the investigation, inspect cited source locations, review the plan and diff, see test/build results, and approve publication. For large tasks, the main agent will delegate independent investigations to isolated subagents and combine their verified findings before answering or editing.

The system will remain understandable and controllable because the application—not an agent framework—will own the main loop, worker loops, tools, memory, evidence, edits, verification, budgets, approvals, and production safety boundaries.
