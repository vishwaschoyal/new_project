"""Isolated subagent fanout.

A subagent is the *same* loop with a fresh context, its own budget, bounded
read-only tools, and one objective. It exists for **context isolation**, not
speed: three independent investigations in one context crowd each other out and
degrade every answer, while three separate contexts each stay sharp.

Parallelism is a secondary benefit. Running workers concurrently reduces elapsed
time; it does not reduce total tokens, and the cost view reflects that honestly.

The fan-in rule matters most. Workers return **compact evidence records**, never
their raw message history. Replaying a worker's conversation into the main
context would defeat the entire purpose — the main agent would end up carrying
every context it was trying to avoid.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from agents import prompts
from agents.evidence import EvidenceLedger
from config import LIMITS
from tools.base import ToolResult
from tools.registry import Capability

if TYPE_CHECKING:
    from agents.orchestrator import AgentEvent, Orchestrator

log = logging.getLogger(__name__)

DELEGATE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "delegate",
        "description": (
            "Split genuinely independent, context-heavy investigations across isolated "
            "workers that run concurrently. Each worker gets a fresh context and returns "
            "compact findings with citations.\n\n"
            "Use this ONLY when the areas do not depend on each other and each needs "
            "substantial reading — for example 'how does auth work' and 'how does billing "
            "work' in a large repository. Do NOT use it to trace one call chain: the hops "
            "depend on each other and belong in a single investigation. Do NOT use it for "
            "a focused question you could answer in a few reads."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "description": "Between 2 and 4 independent objectives.",
                    "minItems": 2,
                    "maxItems": LIMITS.max_subagents,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Short label, e.g. 'auth' or 'billing'.",
                            },
                            "objective": {
                                "type": "string",
                                "description": (
                                    "One self-contained investigation objective. The worker "
                                    "cannot see the conversation, so state everything it needs."
                                ),
                            },
                        },
                        "required": ["name", "objective"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["tasks"],
            "additionalProperties": False,
        },
    },
}


def run_delegation(parent: "Orchestrator", arguments: dict[str, Any]) -> ToolResult:
    """Run workers concurrently and fold their evidence into the parent ledger."""
    from agents.orchestrator import AgentEvent, Orchestrator

    tasks = arguments.get("tasks") or []
    if not isinstance(tasks, list) or len(tasks) < 2:
        return ToolResult.failure(
            "delegate",
            "delegate needs at least 2 independent tasks. For a single line of "
            "investigation, use the read tools directly.",
        )

    cleaned: list[dict[str, str]] = []
    for task in tasks[: LIMITS.max_subagents]:
        if not isinstance(task, dict):
            continue
        name = str(task.get("name") or "").strip()[:40]
        objective = str(task.get("objective") or "").strip()
        if name and len(objective) > 15:
            cleaned.append({"name": name, "objective": objective})

    if len(cleaned) < 2:
        return ToolResult.failure(
            "delegate", "Each task needs a name and a substantial objective."
        )

    # Split the parent's *remaining* budget so delegation cannot overspend the
    # request. Each worker also gets a hard ceiling of its own.
    remaining = max(0, parent.token_budget - parent.usage.total_tokens)
    per_worker = max(
        8_000,
        min(LIMITS.subagent_token_budget, (remaining - LIMITS.final_synthesis_reserve) // len(cleaned)),
    )

    parent.emit(
        AgentEvent(
            "subagents_start",
            {
                "count": len(cleaned),
                "names": [t["name"] for t in cleaned],
                "objectives": {t["name"]: t["objective"] for t in cleaned},
                "budget_each": per_worker,
                "steps_each": LIMITS.subagent_max_steps,
            },
        )
    )

    def _run_worker(task: dict[str, str]) -> tuple[str, Any, "Orchestrator"]:
        worker_name = task["name"]

        def worker_emit(event: "AgentEvent") -> None:
            # Workers stream progress, never answer text: two workers writing
            # prose into one stream would interleave into nonsense.
            #
            # `step` is forwarded because a watcher needs to see a worker that
            # is thinking, not only one that has just finished a tool call —
            # without it a slow worker is indistinguishable from a stalled one.
            if event.type in {"step", "tool_start", "tool_end", "finding"}:
                parent.emit(AgentEvent(f"subagent_{event.type}", {**event.data, "worker": worker_name}))

        worker = Orchestrator(
            workspace=parent.workspace,
            question=task["objective"],
            repo_full_name=parent.repo_full_name,
            branch=parent.branch,
            capabilities={Capability.READ},   # workers never edit
            emit=worker_emit,
            max_steps=LIMITS.subagent_max_steps,
            token_budget=per_worker,
            system_prompt=prompts.SUBAGENT_SYSTEM_PROMPT,
            ledger=EvidenceLedger(),          # a fresh, isolated ledger
            name=worker_name,
            allow_delegation=False,           # workers cannot fan out again
        )
        result = worker.run()

        # A compact completion event, built here rather than by forwarding the
        # worker's own `done`: that one carries the full answer, evidence, and
        # tool trail, and pushing four of those through the stream would cost
        # more bandwidth than the entire rest of the run.
        parent.emit(
            AgentEvent(
                "subagent_done",
                {
                    "worker": worker_name,
                    "steps_used": result.steps_used,
                    "steps_total": LIMITS.subagent_max_steps,
                    "findings": len(result.evidence),
                    "files": len(worker.ledger.observed_files),
                    "termination_reason": result.termination_reason,
                    "cost_usd": worker.usage.cost_usd,
                    "wall_seconds": result.wall_seconds,
                },
            )
        )
        return worker_name, result, worker

    results: list[tuple[str, Any, "Orchestrator"]] = []
    with ThreadPoolExecutor(max_workers=len(cleaned)) as pool:
        futures = {pool.submit(_run_worker, task): task["name"] for task in cleaned}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                log.exception("subagent failed", extra={"worker": name})
                parent.emit(AgentEvent("subagent_error", {"worker": name, "message": str(exc)}))

    # Fan-in. Order by the requested task order so the report is stable even
    # though completion order is not.
    order = {task["name"]: index for index, task in enumerate(cleaned)}
    results.sort(key=lambda item: order.get(item[0], 99))

    sections: list[str] = []
    merged_total = 0
    for worker_name, result, worker in results:
        merged_total += parent.ledger.merge(worker.ledger)
        parent.usage.add(worker.usage)

        lines = [f"## {worker_name}", result.answer.strip() or "(no answer produced)"]
        if result.evidence:
            lines.append("")
            lines.append("Evidence:")
            lines += [
                f"- {item['claim']} ({item['reference']})" for item in result.evidence
            ]
        if result.termination_reason not in {"answered", "answered_partial"}:
            lines.append(f"\n_(worker stopped early: {result.termination_reason})_")
        sections.append("\n".join(lines))

    parent.emit(
        AgentEvent(
            "subagents_end",
            {
                "count": len(results),
                "evidence_merged": merged_total,
                "usage": parent.usage.to_dict(),
            },
        )
    )

    if not sections:
        return ToolResult.failure("delegate", "Every worker failed. Investigate directly instead.")

    covered = sorted({path for worker in (w for _, _, w in results)
                      for path in worker.ledger.observed_files})
    report = (
        f"{len(results)} worker(s) finished. Their findings are in your evidence and you "
        f"may cite them directly, exactly as if you had read those lines yourself.\n\n"
        f"This work is DONE. Do not re-read the files below to confirm it — the answer "
        f"you are being paid to produce is the synthesis of these reports, not a second "
        f"investigation of the same ground. Read further only where a worker explicitly "
        f"reported a gap, or where the question needs something no worker covered.\n\n"
        f"Already covered: {', '.join(covered) if covered else '(no files reported)'}\n\n"
        + "\n\n".join(sections)
    )
    return ToolResult.success(
        "delegate", report, workers=len(results), evidence_merged=merged_total
    )
