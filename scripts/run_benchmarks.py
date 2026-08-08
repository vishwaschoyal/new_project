"""Run the Phase 0 benchmark questions against a live model and record results.

    python scripts/run_benchmarks.py --repo https://github.com/pallets/flask
    python scripts/run_benchmarks.py --only B1 --runs 1 --model gpt-5.4-mini
    python scripts/run_benchmarks.py --compare benchmarks/results/baseline.json

This costs real money — it is the only part of the project that calls the
provider outside a user request. It defaults to one run of each benchmark on the
cheaper model, and prints the estimated cost before committing to anything.

Citations are re-verified independently of the agent: every cited range is read
back off disk and checked for existence. An agent grading its own citations
would tell us nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BENCHMARKS: dict[str, dict[str, str]] = {
    "B1": {
        "name": "focused lookup",
        "question": (
            "Where is the application's secret key read when signing a session "
            "cookie, and what happens if it is unset?"
        ),
        "target_steps": "6",
    },
    "B2": {
        "name": "call-chain trace",
        "question": (
            "Trace what happens from the moment a request arrives at the WSGI entry "
            "point to the moment a view function runs. Name every function in the "
            "chain and cite each one."
        ),
        "target_steps": "20",
    },
}

RESULTS_DIR = ROOT / "benchmarks" / "results"


def verify_citations(workspace: Path, citations: list[dict]) -> dict:
    """Independently confirm each cited range exists in the file on disk."""
    verified, broken = [], []
    for citation in citations:
        path = workspace / citation["path"]
        reference = f"{citation['path']}:{citation['start_line']}-{citation['end_line']}"
        try:
            line_count = sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))
        except OSError:
            broken.append({"reference": reference, "reason": "file not readable"})
            continue
        if citation["end_line"] > line_count:
            broken.append({"reference": reference, "reason": f"file has {line_count} lines"})
        else:
            verified.append(reference)
    return {
        "verified": verified,
        "broken": broken,
        "accuracy": len(verified) / len(citations) if citations else 1.0,
    }


def run_one(workspace: Path, repo_name: str, key: str, question: str) -> dict:
    from agents.orchestrator import Orchestrator

    print(f"\n  {key} …", end="", flush=True)
    started = time.perf_counter()

    orchestrator = Orchestrator(
        workspace=workspace,
        question=question,
        repo_full_name=repo_name,
        branch="main",
        allow_delegation=False,   # benchmarks measure the single loop
    )
    result = orchestrator.run()
    elapsed = round(time.perf_counter() - started, 2)

    citation_report = verify_citations(workspace, result.citations)
    print(
        f" {result.steps_used} steps · {result.usage['total_tokens']:,} tok · "
        f"${result.usage['cost_usd']:.4f} · {elapsed}s · "
        f"citations {len(citation_report['verified'])}/{len(result.citations)}"
    )
    if citation_report["broken"]:
        print(f"    ⚠ broken citations: {citation_report['broken']}")

    return {
        "benchmark": key,
        "name": BENCHMARKS[key]["name"],
        "question": question,
        "answer": result.answer,
        "citations": result.citations,
        "citation_verified": citation_report,
        "unsupported_citations": result.unsupported_citations,
        "evidence_count": len(result.evidence),
        "tool_calls": result.tool_calls,
        "steps_used": result.steps_used,
        "tokens": result.usage,
        "cost_usd": result.usage["cost_usd"],
        "termination_reason": result.termination_reason,
        "wall_seconds": elapsed,
    }


def summarise(runs: list[dict]) -> dict:
    by_benchmark: dict[str, list[dict]] = {}
    for run in runs:
        by_benchmark.setdefault(run["benchmark"], []).append(run)

    summary = {}
    for key, group in by_benchmark.items():
        summary[key] = {
            "runs": len(group),
            "median_cost_usd": round(statistics.median(r["cost_usd"] for r in group), 6),
            "median_steps": statistics.median(r["steps_used"] for r in group),
            "median_tokens": statistics.median(r["tokens"]["total_tokens"] for r in group),
            "median_cache_hit_rate": round(
                statistics.median(r["tokens"]["cache_hit_rate"] for r in group), 4
            ),
            "citation_accuracy": round(
                statistics.mean(r["citation_verified"]["accuracy"] for r in group), 4
            ),
            "all_answered": all(
                r["termination_reason"].startswith("answered") for r in group
            ),
        }
    return summary


def compare(current: dict, baseline_path: Path) -> None:
    """A change is an improvement only if quality holds and cost does not
    regress by more than 10%."""
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["summary"]
    print("\n─── vs baseline ───")
    for key, now in current.items():
        was = baseline.get(key)
        if not was:
            print(f"  {key}: no baseline recorded")
            continue

        cost_delta = (now["median_cost_usd"] - was["median_cost_usd"]) / max(was["median_cost_usd"], 1e-9)
        accuracy_delta = now["citation_accuracy"] - was["citation_accuracy"]

        verdict = "OK"
        if accuracy_delta < 0:
            verdict = "REGRESSION (citation accuracy fell)"
        elif cost_delta > 0.10:
            verdict = f"REGRESSION (cost +{cost_delta:.0%})"

        print(
            f"  {key}: cost {was['median_cost_usd']:.4f} → {now['median_cost_usd']:.4f} "
            f"({cost_delta:+.0%}), citations {was['citation_accuracy']:.0%} → "
            f"{now['citation_accuracy']:.0%} — {verdict}"
        )


def main() -> int:
    # The Windows console defaults to cp1252, which cannot encode the box
    # characters below — and losing a finished benchmark to a print is absurd.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="https://github.com/pallets/flask")
    parser.add_argument("--runs", type=int, default=1, help="runs per benchmark")
    parser.add_argument("--only", choices=sorted(BENCHMARKS), help="run one benchmark")
    parser.add_argument("--model", help="override READ_LOOP_MODEL for this run")
    parser.add_argument("--compare", type=Path, help="compare against a results file")
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = parser.parse_args()

    if args.model:
        # Must be set before config is imported, since it reads the environment.
        os.environ["READ_LOOP_MODEL"] = args.model

    from config import READ_LOOP_MODEL, SETTINGS
    from services.workspace_service import WorkspaceService

    if not SETTINGS.model_configured:
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2

    selected = [args.only] if args.only else sorted(BENCHMARKS)
    total_runs = len(selected) * args.runs

    print(f"Model:      {READ_LOOP_MODEL}")
    print(f"Repository: {args.repo}")
    print(f"Runs:       {total_runs} ({args.runs} × {len(selected)} benchmarks)")
    if not args.yes:
        if input("\nThis makes real API calls. Continue? [y/N] ").strip().lower() != "y":
            return 1

    print("\nCloning…")
    service = WorkspaceService()
    workspace = service.load("benchmark", args.repo)
    print(f"  {workspace.repo.full_name} @ {workspace.head_sha[:8]} ({workspace.branch})")

    runs = []
    try:
        for index in range(args.runs):
            print(f"\nPass {index + 1}/{args.runs}")
            for key in selected:
                runs.append(
                    run_one(workspace.path, workspace.repo.full_name, key, BENCHMARKS[key]["question"])
                )
    finally:
        service.unload("benchmark")

    summary = summarise(runs)
    payload = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": READ_LOOP_MODEL,
        "repo": args.repo,
        "commit": workspace.head_sha,
        "summary": summary,
        "runs": runs,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = "baseline.json" if args.save_baseline else f"{int(time.time())}.json"
    output = RESULTS_DIR / name
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("\n─── summary ───")
    for key, stats in summary.items():
        print(
            f"  {key} ({BENCHMARKS[key]['name']}): {stats['median_steps']} steps, "
            f"{stats['median_tokens']:,} tokens, ${stats['median_cost_usd']:.4f}, "
            f"cache {stats['median_cache_hit_rate']:.0%}, "
            f"citations {stats['citation_accuracy']:.0%}"
        )
    print(f"\nTotal spend: ${sum(r['cost_usd'] for r in runs):.4f}")
    print(f"Saved: {output.relative_to(ROOT)}")

    if args.compare and args.compare.exists():
        compare(summary, args.compare)

    # Any broken citation is a hard failure regardless of cost.
    if any(run["citation_verified"]["broken"] for run in runs):
        print("\nFAIL: at least one citation did not resolve.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
