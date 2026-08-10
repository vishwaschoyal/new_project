"""Orchestrator tests — the loop, with a scripted model and no network."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.orchestrator import AgentEvent, Orchestrator
from tests.fake_model import FakeModel, install, tool_call, turn
from tools.registry import EDITING_CAPABILITIES, READ_ONLY_CAPABILITIES


def build(repo: Path, monkeypatch, script, **kwargs) -> tuple[Orchestrator, FakeModel, list]:
    model = install(monkeypatch, FakeModel(script))
    events: list[AgentEvent] = []
    orchestrator = Orchestrator(
        workspace=repo,
        question=kwargs.pop("question", "Where is the entry point?"),
        repo_full_name="acme/sample-repo",
        branch="main",
        emit=events.append,
        **kwargs,
    )
    return orchestrator, model, events


class TestBasicLoop:
    def test_answers_after_investigating(self, repo: Path, monkeypatch):
        orchestrator, _, _ = build(repo, monkeypatch, [
            turn(calls=[tool_call("grep", pattern="def create_app")]),
            turn(calls=[tool_call("read", path="app.py", offset=1, limit=20)]),
            turn(calls=[tool_call(
                "record_finding",
                claim="create_app builds the application",
                path="app.py", start_line=6, end_line=8,
            )]),
            turn(text="The entry point is `create_app` in app.py:6-8."),
        ])
        result = orchestrator.run()

        assert result.termination_reason == "answered"
        assert "create_app" in result.answer
        assert len(result.citations) == 1
        assert result.citations[0]["path"] == "app.py"

    def test_records_the_tool_trail(self, repo: Path, monkeypatch):
        orchestrator, _, _ = build(repo, monkeypatch, [
            turn(calls=[tool_call("grep", pattern="def")]),
            turn(text="Answer."),
        ])
        result = orchestrator.run()
        assert [call["name"] for call in result.tool_calls] == ["grep"]
        assert result.tool_calls[0]["ok"] is True
        assert "duration_ms" in result.tool_calls[0]

    def test_emits_the_expected_events(self, repo: Path, monkeypatch):
        orchestrator, _, events = build(repo, monkeypatch, [
            turn(calls=[tool_call("glob", pattern="*.py")]),
            turn(text="Done."),
        ])
        orchestrator.run()
        kinds = [event.type for event in events]
        assert "tool_start" in kinds and "tool_end" in kinds
        assert "usage" in kinds and "done" in kinds

    def test_accumulates_usage_across_steps(self, repo: Path, monkeypatch):
        orchestrator, _, _ = build(repo, monkeypatch, [
            turn(calls=[tool_call("read", path="app.py", offset=1, limit=20)],
                 input_tokens=1000, output_tokens=50),
            turn(text="The entry point is app.py:6.", input_tokens=1500, output_tokens=80),
        ])
        result = orchestrator.run()
        assert result.usage["input_tokens"] == 2500
        assert result.usage["output_tokens"] == 130
        assert result.usage["requests"] == 2
        assert result.usage["cost_usd"] > 0

    def test_reports_cached_input_separately(self, repo: Path, monkeypatch):
        """Cache hits are billed at a discount, so they cannot be folded into
        one blended input number."""
        orchestrator, _, _ = build(repo, monkeypatch, [
            turn(calls=[tool_call("read", path="app.py", offset=1, limit=20)],
                 input_tokens=2000, cached_tokens=0),
            turn(text="Entry point: app.py:6.", input_tokens=2000, cached_tokens=1800),
        ])
        result = orchestrator.run()
        assert result.usage["cached_input_tokens"] == 1800
        assert result.usage["cache_hit_rate"] == pytest.approx(0.45)


class TestCitationIntegrity:
    def test_strips_citations_that_were_never_read(self, repo: Path, monkeypatch):
        """The failure this system exists to prevent: a confident answer citing
        a file the agent never opened."""
        orchestrator, _, _ = build(repo, monkeypatch, [
            turn(calls=[tool_call("read", path="app.py", offset=1, limit=20)]),
            turn(text="The entry point is app.py:6, and configuration lives in "
                      "nonexistent_module.py:42."),
        ])
        result = orchestrator.run()

        assert result.unsupported_citations
        assert "nonexistent_module.py:42" not in result.answer
        assert "app.py:6" in result.answer          # the real one survives
        assert "not read during this investigation" in result.answer

    def test_keeps_citations_that_were_read(self, repo: Path, monkeypatch):
        orchestrator, _, _ = build(repo, monkeypatch, [
            turn(calls=[tool_call("read", path="config.py", offset=1, limit=10)]),
            turn(text="Settings load in config.py:4."),
        ])
        result = orchestrator.run()
        assert not result.unsupported_citations
        assert "config.py:4" in result.answer

    def test_record_finding_refuses_unobserved_lines(self, repo: Path, monkeypatch):
        orchestrator, _, _ = build(repo, monkeypatch, [
            turn(calls=[tool_call(
                "record_finding", claim="Invented", path="ghost.py", start_line=1, end_line=5
            )]),
            turn(text="Done."),
        ])
        result = orchestrator.run()
        assert result.evidence == []
        assert result.tool_calls[0]["ok"] is False
        assert "have not read" in result.tool_calls[0]["summary"]


class TestLoopDiscipline:
    def test_challenges_a_premature_answer(self, repo: Path, monkeypatch):
        """A two-part question answered after covering one part gets pushed
        back before it reaches the user."""
        orchestrator, _, events = build(
            repo, monkeypatch,
            [
                turn(text="Auth is in app.py."),        # premature
                turn(calls=[tool_call("read", path="config.py", offset=1, limit=10)]),
                turn(calls=[tool_call(
                    "record_finding", claim="Settings load from the environment",
                    path="config.py", start_line=4, end_line=8,
                )]),
                turn(text="Both parts answered: config.py:4-8."),
            ],
            question="Where is configuration loaded? What happens when it is missing?",
        )
        result = orchestrator.run()
        assert any(event.type == "challenge" for event in events)
        assert result.steps_used >= 3

    def test_warns_on_a_repeated_identical_call(self, repo: Path, monkeypatch):
        orchestrator, _, _ = build(repo, monkeypatch, [
            turn(calls=[tool_call("grep", pattern="def create_app")]),
            turn(calls=[tool_call("grep", pattern="def create_app")]),
            turn(text="Done."),
        ])
        result = orchestrator.run()
        assert result.tool_calls[1]["ok"] is False
        assert "already ran" in result.tool_calls[1]["summary"]

    def test_refuses_a_read_of_lines_already_held(self, repo: Path, monkeypatch):
        """Regression: overlapping reads walked one file eight times.

        The arguments differ every call, so the repeated-call guard never sees
        them. What they have in common is that almost every line requested is
        already in context.
        """
        orchestrator, _, _ = build(repo, monkeypatch, [
            turn(calls=[tool_call("read", path="handlers.py", offset=1, limit=250)]),
            turn(calls=[tool_call("read", path="handlers.py", offset=3, limit=250)]),
            turn(text="Done."),
        ])
        result = orchestrator.run()

        assert result.tool_calls[0]["ok"] is True
        assert result.tool_calls[1]["ok"] is False
        assert "already read" in result.tool_calls[1]["summary"]

    def test_allows_a_read_that_extends_past_what_is_held(self, repo: Path, monkeypatch):
        """The guard must not block genuinely new ground, or tracing a long
        file becomes impossible."""
        long_file = repo / "long.py"
        long_file.write_text("".join(f"line_{n} = {n}\n" for n in range(1, 601)), encoding="utf-8")

        orchestrator, _, _ = build(repo, monkeypatch, [
            turn(calls=[tool_call("read", path="long.py", offset=1, limit=200)]),
            turn(calls=[tool_call("read", path="long.py", offset=201, limit=200)]),
            turn(text="Done."),
        ])
        result = orchestrator.run()

        assert [call["ok"] for call in result.tool_calls] == [True, True]

    def test_forces_an_answer_after_repeated_refusals(self, repo: Path, monkeypatch):
        """Regression: in a live run, a weaker model kept retrying one already-
        refused call twelve times instead of changing approach — a clearer
        warning message did not stop it, because it was not a wording problem.
        Past a handful of refusals in one request, the loop must force an
        answer rather than pay for yet another identical retry.
        """
        from agents.orchestrator import MAX_THRASH_REFUSALS

        # The first call is legitimate; each identical repeat after it is a
        # refusal. Deriving the expectation from the ceiling keeps this test
        # honest when the ceiling is retuned — a refusal costs a full context
        # re-send, so that number is expected to move.
        expected_steps = 1 + MAX_THRASH_REFUSALS
        script = [
            turn(calls=[tool_call("grep", pattern="def create_app")])
            for _ in range(expected_steps + 2)
        ]
        orchestrator, _, _ = build(repo, monkeypatch, script, max_steps=30)
        result = orchestrator.run()

        assert result.termination_reason == "thrash_detected"
        assert result.steps_used == expected_steps

    def test_stops_at_the_step_limit(self, repo: Path, monkeypatch):
        script = [turn(calls=[tool_call("glob", pattern=f"*{i}.py")]) for i in range(20)]
        script.append(turn(text="Finalised."))
        orchestrator, _, _ = build(repo, monkeypatch, script, max_steps=3)
        result = orchestrator.run()
        assert result.termination_reason == "step_limit"
        assert result.steps_used == 3

    def test_stops_when_the_token_budget_is_exhausted(self, repo: Path, monkeypatch):
        orchestrator, _, _ = build(
            repo, monkeypatch,
            [turn(calls=[tool_call("glob", pattern="*.py")], input_tokens=9_000),
             turn(text="Partial answer.")],
            token_budget=9_500,
        )
        result = orchestrator.run()
        assert result.termination_reason == "token_budget"

    def test_unavailable_tool_is_refused_not_executed(self, repo: Path, monkeypatch):
        orchestrator, _, _ = build(
            repo, monkeypatch,
            [turn(calls=[tool_call("edit", path="app.py", old_string="import os", new_string="")]),
             turn(text="Could not edit.")],
            capabilities=READ_ONLY_CAPABILITIES,
        )
        result = orchestrator.run()
        assert result.tool_calls[0]["ok"] is False
        assert "import os" in (repo / "app.py").read_text(encoding="utf-8")


class TestFailureHandling:
    def test_provider_failure_returns_recorded_evidence(self, repo: Path, monkeypatch):
        """A run that dies mid-flight must not throw away findings the user
        already paid for."""
        orchestrator, model, _ = build(repo, monkeypatch, [
            turn(calls=[tool_call("read", path="app.py", offset=1, limit=20)]),
            turn(calls=[tool_call(
                "record_finding", claim="create_app is the entry point",
                path="app.py", start_line=6, end_line=8,
            )]),
        ])

        original_stream = model.stream
        state = {"count": 0}

        def failing_stream(messages, **kwargs):
            state["count"] += 1
            if state["count"] > 2:
                raise RuntimeError("upstream connection reset")
            yield from original_stream(messages, **kwargs)

        monkeypatch.setattr(model, "stream", failing_stream)

        result = orchestrator.run()
        assert result.termination_reason == "provider_error"
        assert "create_app is the entry point" in result.answer

    def test_total_provider_failure_is_reported_honestly(self, repo: Path, monkeypatch):
        model = install(monkeypatch, FakeModel([], fail_with=RuntimeError("503 unavailable")))
        orchestrator = Orchestrator(
            workspace=repo, question="Where is the entry point?", repo_full_name="acme/x"
        )
        result = orchestrator.run()
        assert result.termination_reason == "provider_error"
        assert "could not complete" in result.answer.lower()
        assert "503" in result.answer

    def test_a_failing_tool_does_not_end_the_run(self, repo: Path, monkeypatch):
        orchestrator, _, _ = build(repo, monkeypatch, [
            turn(calls=[tool_call("read", path="does_not_exist.py")]),
            turn(calls=[tool_call("grep", pattern="does_not_exist")]),
            turn(text="No such file exists in this repository."),
        ])
        result = orchestrator.run()
        assert result.termination_reason.startswith("answered")
        assert result.tool_calls[0]["ok"] is False
        assert len(result.tool_calls) == 2   # the run continued past the failure


class TestContextManagement:
    def test_compacts_when_observations_grow(self, repo: Path, monkeypatch):
        from agents.orchestrator import (
            COMPACTION_TRIGGER_CHARS,
            KEEP_RECENT_OBSERVATIONS,
            MAX_OBSERVATION_CHARS,
        )

        # How many distinct reads it takes to push the working context past the
        # trigger, computed rather than hardcoded: the threshold is a tuning
        # constant, and a test that pins it silently stops asserting anything
        # the moment it is raised.
        per_read = MAX_OBSERVATION_CHARS
        reads = COMPACTION_TRIGGER_CHARS // per_read + KEEP_RECENT_OBSERVATIONS + 1
        lines_per_read = 200

        big = repo / "big.py"
        big.write_text(
            "\n".join(f"# padding line {i}" for i in range(reads * lines_per_read + 10)),
            encoding="utf-8",
        )

        script = [
            turn(calls=[tool_call(
                "read", path="big.py", offset=i * lines_per_read + 1, limit=lines_per_read
            )])
            for i in range(reads)
        ]
        script.append(turn(text="Done."))
        orchestrator, _, events = build(
            repo, monkeypatch, script, max_steps=reads + 2, token_budget=1_000_000
        )
        orchestrator.run()

        assert any(event.type == "compaction" for event in events)

    def test_evidence_survives_compaction(self, repo: Path, monkeypatch):
        big = repo / "big.py"
        big.write_text("\n".join(f"# padding line {i}" for i in range(4000)), encoding="utf-8")

        script = [
            turn(calls=[tool_call("read", path="big.py", offset=1, limit=200)]),
            turn(calls=[tool_call(
                "record_finding", claim="Padding starts at the top",
                path="big.py", start_line=1, end_line=10,
            )]),
        ]
        script += [
            turn(calls=[tool_call("read", path="big.py", offset=i * 200 + 201, limit=200)])
            for i in range(8)
        ]
        script.append(turn(text="Answer citing big.py:1-10."))

        orchestrator, _, _ = build(repo, monkeypatch, script, max_steps=14)
        result = orchestrator.run()

        assert any("Padding starts" in item["claim"] for item in result.evidence)
        assert result.citations


class TestBatchedToolCalls:
    """Several calls arriving in one model response.

    A step re-sends the whole conversation, so the bill is roughly
    steps x context. Batching independent calls is the loop's main defence
    against that, and it is worth nothing if the loop mishandles a batch.
    """

    def test_provider_is_asked_for_batched_calls(self):
        """The setting that makes batching possible at all."""
        import inspect

        from config import create_read_loop_model

        source = inspect.getsource(create_read_loop_model)
        assert '"parallel_tool_calls"] = True' in source

    def test_every_call_in_one_step_runs_and_is_observed(self, repo: Path, monkeypatch):
        orchestrator, model, _ = build(repo, monkeypatch, [
            turn(calls=[
                tool_call("read", path="app.py", offset=1, limit=20),
                tool_call("glob", pattern="*.py"),
                tool_call("grep", pattern="def create_app"),
            ]),
            # Cites a line the batch actually read, so the answer settles the
            # obligation and the run is not challenged into extra steps.
            turn(text="The entry point is `create_app` in app.py:6-8."),
        ])
        result = orchestrator.run()

        assert [call["name"] for call in result.tool_calls] == ["read", "glob", "grep"]
        # One step, not three: that is the entire point.
        assert {call["step"] for call in result.tool_calls} == {1}
        assert result.steps_used == 2

        # Each call needs its own observation, or the model sees a batch it
        # cannot match up to what it asked for.
        from langchain_core.messages import ToolMessage

        observations = [m for m in orchestrator._messages if isinstance(m, ToolMessage)]
        assert len(observations) == 3
        assert [m.name for m in observations] == ["read", "glob", "grep"]

    def test_a_finding_batched_with_the_read_that_earned_it(self, repo: Path, monkeypatch):
        """The pattern the prompt asks for: no round trip spent on bookkeeping.

        `record_finding` is refused unless the lines were observed, so this
        also pins the ordering guarantee — the read must be dispatched before
        the finding that depends on it, within the same batch.
        """
        orchestrator, _, _ = build(repo, monkeypatch, [
            turn(calls=[
                tool_call("read", path="app.py", offset=1, limit=20),
                tool_call(
                    "record_finding",
                    claim="create_app builds the application",
                    path="app.py", start_line=6, end_line=8,
                ),
            ]),
            turn(text="The entry point is `create_app` in app.py:6-8."),
        ])
        result = orchestrator.run()

        assert result.termination_reason == "answered"
        assert len(result.evidence) == 1
        assert len(result.citations) == 1
        assert result.steps_used == 2

    def test_a_failing_call_does_not_abandon_the_rest_of_its_batch(self, repo: Path, monkeypatch):
        orchestrator, _, _ = build(repo, monkeypatch, [
            turn(calls=[
                tool_call("read", path="does-not-exist.py", offset=1, limit=20),
                tool_call("glob", pattern="*.py"),
            ]),
            turn(text="Done."),
        ])
        result = orchestrator.run()

        assert [call["ok"] for call in result.tool_calls] == [False, True]


class TestPromptStability:
    def test_system_prompt_is_first_and_constant(self, repo: Path, monkeypatch):
        """The cacheable prefix must not drift between steps."""
        orchestrator, model, _ = build(repo, monkeypatch, [
            turn(calls=[tool_call("glob", pattern="*.py")]),
            turn(calls=[tool_call("glob", pattern="*.md")]),
            turn(text="Done."),
        ])
        orchestrator.run()

        prefixes = {str(messages[0].content) for messages in model.calls_received}
        assert len(prefixes) == 1
        assert model.calls_received[0][0].type == "system"

    def test_tool_schemas_are_bound_once_and_ordered(self, repo: Path, monkeypatch):
        orchestrator, model, _ = build(repo, monkeypatch, [turn(text="Done.")])
        orchestrator.run()
        names = [tool["function"]["name"] for tool in model.bound_tools]
        assert names[:4] == ["bash", "glob", "grep", "read"]
        assert "record_finding" in names

    def test_volatile_context_is_last(self, repo: Path, monkeypatch):
        orchestrator, model, _ = build(repo, monkeypatch, [turn(text="Done.")])
        orchestrator.run()
        assert "Investigation context" in str(model.calls_received[0][-1].content)


class TestEditingMode:
    def test_edit_then_verify(self, repo: Path, monkeypatch):
        orchestrator, _, _ = build(
            repo, monkeypatch,
            [
                turn(calls=[tool_call("read", path="config.py", offset=1, limit=20)]),
                turn(calls=[tool_call(
                    "edit", path="config.py",
                    old_string="raise RuntimeError('SECRET_KEY is required')",
                    new_string="raise RuntimeError('SECRET_KEY must be set')",
                )]),
                turn(text="Updated the error message in config.py:7."),
            ],
            capabilities=EDITING_CAPABILITIES,
        )
        result = orchestrator.run()
        assert result.tool_calls[1]["ok"] is True
        assert "must be set" in (repo / "config.py").read_text(encoding="utf-8")

    def test_editing_prompt_is_used_for_edit_capability(self, repo: Path, monkeypatch):
        orchestrator, _, _ = build(
            repo, monkeypatch, [turn(text="ok")], capabilities=EDITING_CAPABILITIES
        )
        assert "repository engineer" in orchestrator.system_prompt
