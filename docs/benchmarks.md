# Benchmark Questions and Evaluation Method

Phase 0 artifact. These two questions are the fixed yardstick for every later
phase. They are re-run after prompt changes, budget changes, tool-bound changes,
and before every release promotion.

Both benchmarks run against a **pinned target repository and commit** so that
answer quality is comparable across runs. Changing the target invalidates the
recorded baseline and requires re-recording it.

```
BENCHMARK_REPO   = https://github.com/pallets/flask
BENCHMARK_COMMIT = (pinned at first baseline run; recorded in results/baseline.json)
```

## B1 — Focused lookup

> **Question:** Where is the Flask application's `secret_key` read when signing a
> session cookie, and what happens if it is unset?

Exercises: search-first behaviour, a bounded read, early stopping. A correct run
should *not* wander the repository — this is the cost-discipline benchmark.

**Scored on**

| Criterion | Target |
| --- | --- |
| Answer correctness | Names the signing path and the unset-key failure |
| Citation correctness | Every cited line was actually read during the request |
| Tool steps | ≤ 6 |
| Stops early | Yes — no reads after the question is answerable |

## B2 — Call-chain trace

> **Question:** Trace what happens from the moment a request arrives at the WSGI
> entry point to the moment a view function runs. Name every function in the
> chain and cite each one.

Exercises: multi-hop tracing, investigation obligations, evidence survival
across context compaction. A correct run follows every material edge instead of
answering from the first plausible file.

**Scored on**

| Criterion | Target |
| --- | --- |
| Chain completeness | Every material hop named, none invented |
| Citation correctness | One exact observed citation per hop |
| Tool steps | ≤ 20 |
| Finishes inside budget | Yes — no truncated/best-effort finalisation |

## Recorded per run

Written to `benchmarks/results/<timestamp>.json` by `scripts/run_benchmarks.py`:

- `answer` and full `citations[]` (path, line range)
- `citation_verified` — recomputed by re-reading each cited range
- `tool_calls[]` — name, arguments, duration, truncation flag
- `tokens` — input, cached input, output, reasoning
- `cost_usd` — estimated from the pricing table in `services/cost_service.py`
- `termination_reason` — answered / step limit / token budget / provider error
- `wall_seconds`

## Evaluation method

1. Run each benchmark **three times** and record all three (model output varies).
2. Compare against `benchmarks/results/baseline.json`.
3. A change is an improvement only if answer and citation correctness hold or
   improve **and** median cost does not regress by more than 10%.
4. Any wrong citation is a hard failure regardless of cost — it becomes a
   regression test in `tests/` before anything else is tuned.

## Restoring the starting point

The pre-rebuild implementation is preserved outside this repository at
`../git_project-v2-main` and on the `archive/pre-rebuild` branch there. This
repository's own starting point is its initial commit:

```bash
git log --oneline           # find the initial commit
git restore --source=<sha> .
```
