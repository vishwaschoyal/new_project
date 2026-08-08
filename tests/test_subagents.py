"""Subagent fanout: isolation, fan-in, and the guard against needless overhead."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from agents.orchestrator import AgentEvent, Orchestrator
from agents.subagents import run_delegation
from tests.fake_model import FakeModel, tool_call, turn
from tools.registry import Capability


class ScriptedByQuestion:
    """A fake model that answers based on which question it was asked.

    Workers share one model object here, so the script must be selected by
    content rather than by call order — which also proves each worker really
    does get its own separate context.
    """

    def __init__(self, scripts: dict[str, list], default: list | None = None):
        self.scripts = scripts
        self.default = default or [turn(text="No script matched.")]
        self.positions: dict[str, int] = {}
        self.seen_questions: list[str] = []
        self.lock = threading.Lock()

    def bind_tools(self, tools, **_kwargs):
        return self

    def _script_for(self, messages) -> tuple[str, list]:
        text = " ".join(str(m.content) for m in messages).lower()
        for key, script in self.scripts.items():
            if key.lower() in text:
                return key, script
        return "__default__", self.default

    def stream(self, messages, **_kwargs):
        from langchain_core.messages import AIMessageChunk

        with self.lock:
            key, script = self._script_for(messages)
            self.seen_questions.append(key)
            index = self.positions.get(key, 0)
            self.positions[key] = index + 1

        step = script[index] if index < len(script) else turn(text="Finished.")
        yield AIMessageChunk(
            content=step["text"], tool_calls=step["calls"], usage_metadata=step["usage"]
        )

    def invoke(self, messages, **kwargs):
        aggregate = None
        for chunk in self.stream(messages, **kwargs):
            aggregate = chunk if aggregate is None else aggregate + chunk
        return aggregate


@pytest.fixture
def parent(repo: Path):
    return Orchestrator(
        workspace=repo,
        question="How do configuration and request handling work?",
        repo_full_name="acme/sample-repo",
        allow_delegation=True,
    )


class TestDelegationGuards:
    def test_refuses_a_single_task(self, parent):
        """One 'independent' task is just an investigation with extra cost."""
        result = run_delegation(parent, {"tasks": [{"name": "solo", "objective": "look at everything"}]})
        assert not result.ok
        assert "at least 2" in result.content

    def test_refuses_empty_objectives(self, parent):
        result = run_delegation(parent, {
            "tasks": [{"name": "a", "objective": "x"}, {"name": "b", "objective": "y"}]
        })
        assert not result.ok

    def test_caps_the_worker_count(self, parent, monkeypatch):
        from config import LIMITS

        model = ScriptedByQuestion({}, default=[turn(text="Nothing found.")])
        monkeypatch.setattr("agents.orchestrator.create_read_loop_model", lambda **_k: model)

        tasks = [
            {"name": f"w{i}", "objective": f"investigate area number {i} thoroughly"}
            for i in range(10)
        ]
        result = run_delegation(parent, {"tasks": tasks})
        assert result.metadata["workers"] <= LIMITS.max_subagents

    def test_workers_cannot_delegate_again(self, repo: Path):
        """Recursive fanout would multiply cost without bound."""
        worker = Orchestrator(
            workspace=repo, question="scoped objective", allow_delegation=False
        )
        names = [tool["function"]["name"] for tool in worker._tool_schemas()]
        assert "delegate" not in names

    def test_main_agent_has_delegate(self, parent):
        names = [tool["function"]["name"] for tool in parent._tool_schemas()]
        assert "delegate" in names


class TestFanIn:
    def test_merges_evidence_from_every_worker(self, parent, monkeypatch):
        model = ScriptedByQuestion({
            "configuration": [
                turn(calls=[tool_call("read", path="config.py", offset=1, limit=10)]),
                turn(calls=[tool_call(
                    "record_finding", claim="Settings come from the environment",
                    path="config.py", start_line=4, end_line=8,
                )]),
                turn(text="Configuration loads from the environment (config.py:4-8)."),
            ],
            "request handling": [
                turn(calls=[tool_call("read", path="handlers.py", offset=1, limit=15)]),
                turn(calls=[tool_call(
                    "record_finding", claim="Requests are validated then answered",
                    path="handlers.py", start_line=1, end_line=3,
                )]),
                turn(text="Requests are validated then answered (handlers.py:1-3)."),
            ],
        })
        monkeypatch.setattr("agents.orchestrator.create_read_loop_model", lambda **_k: model)

        result = run_delegation(parent, {"tasks": [
            {"name": "config", "objective": "Explain how configuration is loaded"},
            {"name": "handling", "objective": "Explain how request handling works"},
        ]})

        assert result.ok
        assert result.metadata["evidence_merged"] == 2

        # The parent can now cite both workers' lines as its own.
        assert parent.ledger.was_observed("config.py", 4, 8)
        assert parent.ledger.was_observed("handlers.py", 1, 3)
        sources = {record.source for record in parent.ledger.records}
        assert sources == {"config", "handling"}

    def test_report_contains_findings_not_raw_history(self, parent, monkeypatch):
        """Fan-in must not replay worker conversations — that would reimport
        every context the isolation was meant to avoid."""
        model = ScriptedByQuestion({
            "configuration": [
                turn(calls=[tool_call("read", path="config.py", offset=1, limit=10)]),
                turn(calls=[tool_call(
                    "record_finding", claim="Settings come from the environment",
                    path="config.py", start_line=4, end_line=8,
                )]),
                turn(text="Configuration loads from the environment (config.py:4-8)."),
            ],
            "handling": [turn(text="Handled in handlers.py.")],
        })
        monkeypatch.setattr("agents.orchestrator.create_read_loop_model", lambda **_k: model)

        result = run_delegation(parent, {"tasks": [
            {"name": "config", "objective": "Explain how configuration is loaded"},
            {"name": "handling", "objective": "Explain how handling works"},
        ]})

        assert "Settings come from the environment" in result.content
        assert "1 | import os" not in result.content     # no raw tool output

    def test_worker_usage_is_billed_to_the_parent(self, parent, monkeypatch):
        model = ScriptedByQuestion({}, default=[turn(text="Done.", input_tokens=800, output_tokens=40)])
        monkeypatch.setattr("agents.orchestrator.create_read_loop_model", lambda **_k: model)

        before = parent.usage.total_tokens
        run_delegation(parent, {"tasks": [
            {"name": "a", "objective": "investigate the first independent area"},
            {"name": "b", "objective": "investigate the second independent area"},
        ]})
        assert parent.usage.total_tokens > before
        assert parent.usage.cost_usd > 0

    def test_budget_is_split_not_multiplied(self, parent, monkeypatch):
        """Delegation must not let a run exceed the request budget."""
        model = ScriptedByQuestion({}, default=[turn(text="Done.")])
        monkeypatch.setattr("agents.orchestrator.create_read_loop_model", lambda **_k: model)

        events: list[AgentEvent] = []
        parent.emit = events.append
        run_delegation(parent, {"tasks": [
            {"name": "a", "objective": "investigate the first independent area"},
            {"name": "b", "objective": "investigate the second independent area"},
        ]})

        start = next(e for e in events if e.type == "subagents_start")
        assert start.data["budget_each"] * 2 <= parent.token_budget

    def test_a_failing_worker_does_not_lose_the_others(self, parent, monkeypatch):
        class PartlyBroken(ScriptedByQuestion):
            def stream(self, messages, **kwargs):
                text = " ".join(str(m.content) for m in messages).lower()
                if "broken area" in text:
                    raise RuntimeError("worker transport failed")
                yield from super().stream(messages, **kwargs)

        model = PartlyBroken({
            "working area": [
                turn(calls=[tool_call("read", path="config.py", offset=1, limit=10)]),
                turn(calls=[tool_call(
                    "record_finding", claim="The working area loads settings",
                    path="config.py", start_line=4, end_line=8,
                )]),
                turn(text="The working area loads settings (config.py:4-8)."),
            ],
        }, default=[turn(text="ok")])
        monkeypatch.setattr("agents.orchestrator.create_read_loop_model", lambda **_k: model)

        result = run_delegation(parent, {"tasks": [
            {"name": "good", "objective": "investigate the working area of the system"},
            {"name": "bad", "objective": "investigate the broken area of the system"},
        ]})

        # The failing worker degrades to a best-effort report rather than
        # taking the whole delegation — and the healthy worker's evidence —
        # down with it.
        assert result.ok
        assert "The working area loads settings" in result.content
        assert result.metadata["evidence_merged"] == 1
        assert "provider_error" in result.content   # the failure is reported, not hidden


class TestIsolation:
    def test_each_worker_gets_its_own_ledger(self, parent, monkeypatch):
        model = ScriptedByQuestion({
            "alpha": [
                turn(calls=[tool_call("read", path="config.py", offset=1, limit=10)]),
                turn(text="Alpha done."),
            ],
            "beta": [turn(text="Beta found nothing.")],
        })
        monkeypatch.setattr("agents.orchestrator.create_read_loop_model", lambda **_k: model)

        run_delegation(parent, {"tasks": [
            {"name": "alpha", "objective": "investigate the alpha subsystem in detail"},
            {"name": "beta", "objective": "investigate the beta subsystem in detail"},
        ]})
        # Only alpha read config.py; beta's context never contained it.
        assert model.seen_questions.count("alpha") >= 1
        assert model.seen_questions.count("beta") >= 1

    def test_workers_are_read_only(self, repo: Path):
        worker = Orchestrator(
            workspace=repo, question="objective", capabilities={Capability.READ}
        )
        assert "edit" not in worker.registry.names
