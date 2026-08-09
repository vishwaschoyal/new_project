"""Evidence ledger and investigation state."""

from __future__ import annotations

import pytest

from agents.evidence import Citation, EvidenceLedger, EvidenceRecord
from agents.investigation_state import InvestigationState, decompose_question


class TestEvidenceLedger:
    def test_records_observations_from_a_read(self):
        ledger = EvidenceLedger()
        ledger.observe_tool_result(
            "read", {"path": "app.py", "start_line": 10, "end_line": 40}, ""
        )
        assert ledger.was_observed("app.py", 15, 20)
        assert not ledger.was_observed("app.py", 5, 20)
        assert not ledger.was_observed("other.py", 15)

    def test_records_observations_from_grep_hits(self):
        ledger = EvidenceLedger()
        ledger.observe_tool_result(
            "grep", {}, "handlers.py:12:def validate(request):\napp.py:6:def create_app():"
        )
        assert ledger.was_observed("handlers.py", 12)
        assert ledger.was_observed("app.py", 6)
        assert not ledger.was_observed("app.py", 7)

    def test_merges_adjacent_ranges(self):
        ledger = EvidenceLedger()
        ledger.record_observation("a.py", 1, 10)
        ledger.record_observation("a.py", 11, 20)
        assert ledger.observed_ranges()["a.py"] == [(1, 20)]

    def test_refuses_unobserved_evidence(self):
        """The central honesty rule: no citation without an observation."""
        ledger = EvidenceLedger()
        record = EvidenceRecord("Something is true", Citation("never_read.py", 5, 9))
        assert ledger.add(record) is False
        assert len(ledger) == 0

    def test_accepts_observed_evidence(self):
        ledger = EvidenceLedger()
        ledger.record_observation("app.py", 1, 50)
        assert ledger.add(EvidenceRecord("create_app builds it", Citation("app.py", 6, 8))) is True
        assert len(ledger) == 1

    def test_replaces_duplicate_citations(self):
        ledger = EvidenceLedger()
        ledger.record_observation("app.py", 1, 50)
        ledger.add(EvidenceRecord("first", Citation("app.py", 6, 8)))
        ledger.add(EvidenceRecord("revised", Citation("app.py", 6, 8)))
        assert len(ledger) == 1
        assert ledger.records[0].claim == "revised"

    def test_extracts_citations_from_text(self):
        ledger = EvidenceLedger()
        found = ledger.extract_citations(
            "See app.py:12 and handlers.py:5-9 for detail."
        )
        assert Citation("app.py", 12, 12) in found
        assert Citation("handlers.py", 5, 9) in found

    def test_verifies_an_answer(self):
        ledger = EvidenceLedger()
        ledger.record_observation("app.py", 1, 50)
        report = ledger.verify_answer("Real: app.py:10. Invented: ghost.py:99.")
        assert report["all_supported"] is False
        assert len(report["supported"]) == 1
        assert report["unsupported"][0]["path"] == "ghost.py"

    def test_merge_preserves_worker_findings(self):
        """Fan-in must merge observations before records, or every worker
        finding would be rejected as unobserved by the parent."""
        parent, worker = EvidenceLedger(), EvidenceLedger()
        worker.record_observation("billing.py", 1, 100)
        worker.add(EvidenceRecord("Invoices are generated here", Citation("billing.py", 40, 55), source="billing"))

        assert parent.merge(worker) == 1
        assert parent.was_observed("billing.py", 40, 55)
        assert parent.records[0].source == "billing"

    def test_renders_compact_prompt_block(self):
        ledger = EvidenceLedger()
        ledger.record_observation("app.py", 1, 50)
        ledger.add(EvidenceRecord("Entry point", Citation("app.py", 6, 8)))
        rendered = ledger.render_for_prompt()
        assert "Entry point" in rendered and "app.py:6-8" in rendered


class TestObservationCoverage:
    """Overlap arithmetic behind the redundant-read refusal.

    Regression: a loop tracing one file read it eight times in overlapping
    windows (184-433, then 232-481, then 280-529...). Every call was a new
    argument set, so the repeated-call guard never fired, and the run paid for
    the same lines repeatedly while inflating the context into needless
    compaction.
    """

    def test_full_overlap_is_total_coverage(self):
        ledger = EvidenceLedger()
        ledger.record_observation("a.py", 1, 100)
        assert ledger.coverage_of("a.py", 20, 40) == 1.0

    def test_partial_overlap_is_measured(self):
        ledger = EvidenceLedger()
        ledger.record_observation("a.py", 184, 433)
        # The real second read from the transcript: mostly lines already held.
        assert ledger.coverage_of("a.py", 232, 481) > 0.8

    def test_a_genuine_extension_is_not_covered(self):
        ledger = EvidenceLedger()
        ledger.record_observation("a.py", 184, 433)
        assert ledger.coverage_of("a.py", 434, 683) == 0.0

    def test_unread_file_is_uncovered(self):
        assert EvidenceLedger().coverage_of("never.py", 1, 50) == 0.0

    def test_resume_point_skips_what_is_held(self):
        ledger = EvidenceLedger()
        ledger.record_observation("a.py", 184, 433)
        assert ledger.first_unobserved_line("a.py", 232, 481) == 434

    def test_no_resume_point_when_fully_read(self):
        ledger = EvidenceLedger()
        ledger.record_observation("a.py", 1, 100)
        assert ledger.first_unobserved_line("a.py", 10, 90) is None


class TestCitationContentVerification:
    """A citation must describe the lines it points at, not merely have visited
    them.

    Regression: an answer cited `static/app.js:537-596` for `send()` and
    `handleEvent()`. Those lines had genuinely been read, so range checking
    passed them — but they hold unrelated trail-rendering code, and the real
    functions live hundreds of lines away.
    """

    @pytest.fixture
    def ledger(self, repo):
        return EvidenceLedger(workspace=repo)

    def test_claim_matching_the_cited_lines_is_supported(self, ledger):
        ledger.record_observation("handlers.py", 1, 12)
        report = ledger.verify_answer("validate(request) raises on a missing path. handlers.py:6-8")
        assert len(report["supported"]) == 1
        assert report["contradicted"] == []

    def test_claim_naming_a_symbol_that_is_elsewhere_is_contradicted(self, tmp_path):
        """The exact shape of the real failure: lines read, claim unrelated.

        The reported answer cited app.js:537-596 for `send()`, which is really
        defined at 889. Both regions had been read, so only the content check
        can tell them apart.
        """
        source = ["function trailAdd(item) {}"] * 888 + ["async function send(question) {}"]
        (tmp_path / "app.js").write_text("\n".join(source), encoding="utf-8")

        ledger = EvidenceLedger(workspace=tmp_path)
        ledger.record_observation("app.js", 520, 640)
        report = ledger.verify_answer("send(question) submits the message. app.js:537-596")

        assert len(report["contradicted"]) == 1
        assert report["all_supported"] is False

    def test_unobserved_lines_stay_unsupported_not_contradicted(self, ledger):
        report = ledger.verify_answer("Something happens. never_read.py:5")
        assert len(report["unsupported"]) == 1
        assert report["contradicted"] == []

    def test_a_claim_with_no_identifiers_is_left_alone(self, ledger):
        """Prose cannot be checked against source, so it is never rejected."""
        ledger.record_observation("handlers.py", 1, 14)
        report = ledger.verify_answer("This is where the work happens. handlers.py:1-3")
        assert len(report["supported"]) == 1

    def test_citing_a_body_rather_than_the_def_line_still_passes(self, ledger):
        """Slack absorbs the normal habit of citing a function's body."""
        ledger.record_observation("app.py", 1, 16)
        report = ledger.verify_answer("create_app() builds the application. app.py:7-8")
        assert len(report["supported"]) == 1

    def test_a_symbol_named_only_inside_a_comment_does_not_count(self, tmp_path):
        """Whole-identifier matching: 'sends' in prose is not the `send` symbol."""
        (tmp_path / "ui.js").write_text(
            "// the loop never sends its reasoning\nfunction render() {}\n", encoding="utf-8"
        )
        ledger = EvidenceLedger(workspace=tmp_path)
        ledger.record_observation("ui.js", 1, 2)
        report = ledger.verify_answer("send() posts the message. ui.js:1-2")
        assert len(report["contradicted"]) == 1

    def test_a_trailing_citation_keeps_the_line_above_it(self, ledger):
        """A citation alone on its line is judged with the claim that earned it."""
        ledger.record_observation("handlers.py", 1, 14)
        report = ledger.verify_answer("validate(request) checks the path.\nhandlers.py:6-8")
        assert len(report["supported"]) == 1

    def test_without_a_workspace_only_ranges_are_checked(self):
        """No disk access configured means no opinion — never a rejection."""
        ledger = EvidenceLedger()
        ledger.record_observation("handlers.py", 1, 3)
        report = ledger.verify_answer("respond() lives here. handlers.py:1-3")
        assert len(report["supported"]) == 1

    def test_a_citation_to_a_secret_file_is_not_opened(self, ledger):
        """Citations come from model output and go through the same path rules."""
        ledger.record_observation(".env", 1, 1)
        report = ledger.verify_answer("The key is set here. .env:1")
        assert report["contradicted"] == []


class TestDecomposition:
    def test_single_question_stays_single(self):
        assert len(decompose_question("Where is the entry point?")) == 1

    def test_splits_two_questions(self):
        parts = decompose_question("Where is auth handled? How are tokens refreshed?")
        assert len(parts) == 2

    def test_splits_explicit_connectors(self):
        parts = decompose_question(
            "Explain how configuration is loaded and also what happens when it is missing"
        )
        assert len(parts) == 2

    def test_splits_enumerations(self):
        parts = decompose_question("Cover these:\n1. how routing works\n2. how errors surface")
        assert len(parts) == 2

    def test_handles_empty(self):
        assert decompose_question("") == []


class TestInvestigationState:
    def test_multipart_question_is_incomplete_at_the_start(self):
        state = InvestigationState("Where is auth handled? How are tokens refreshed?")
        assert len(state.open_obligations) == 2
        assert state.is_complete is False

    def test_resolving_all_parts_completes_it(self):
        state = InvestigationState("Where is auth handled? How are tokens refreshed?")
        for obligation in list(state.open_obligations):
            state.resolve(obligation.id, answer="done", citations=["a.py:1"])
        assert state.is_complete is True

    def test_challenges_a_premature_finish(self):
        state = InvestigationState("Where is auth handled? How are tokens refreshed?")
        challenge = state.challenge_finalisation()
        assert challenge and "not yet established" in challenge

    def test_stops_challenging_eventually(self):
        """Discipline, not deadlock — after a bounded number of pushbacks the
        loop must be allowed to answer."""
        state = InvestigationState("Where is auth? How are tokens refreshed?")
        challenges = [state.challenge_finalisation() for _ in range(5)]
        assert challenges[0] is not None
        assert challenges[-1] is None

    def test_no_challenge_when_complete(self):
        state = InvestigationState("Where is the entry point?")
        state.resolve(state.open_obligations[0].id, answer="found", citations=["app.py:6"])
        assert state.challenge_finalisation() is None

    def test_detects_repeated_calls(self):
        state = InvestigationState("q")
        assert state.register_call("grep", {"pattern": "x"}) is None
        warning = state.register_call("grep", {"pattern": "x"})
        assert warning and "already ran" in warning

    def test_argument_order_does_not_hide_a_repeat(self):
        state = InvestigationState("q")
        state.register_call("grep", {"pattern": "x", "glob": "*.py"})
        assert state.register_call("grep", {"glob": "*.py", "pattern": "x"}) is not None

    def test_different_calls_are_not_repeats(self):
        state = InvestigationState("q")
        state.register_call("grep", {"pattern": "x"})
        assert state.register_call("grep", {"pattern": "y"}) is None

    def test_matches_a_finding_to_the_right_obligation(self):
        state = InvestigationState(
            "How does authentication work? How does billing invoicing work?"
        )
        resolved = state.resolve_best_match(
            answer="Billing invoicing is generated by the invoice builder",
            citations=["billing.py:40"],
        )
        assert resolved is not None
        answered = [o for o in state.obligations if o.status == "answered"]
        assert "billing" in answered[0].question.lower()

    def test_discovered_questions_can_be_added(self):
        state = InvestigationState("Trace the request path")
        assert state.add_obligation("What does the dispatcher call next?") is not None
        assert len(state.open_obligations) == 2

    def test_rejects_trivial_additions(self):
        state = InvestigationState("q")
        assert state.add_obligation("hi") is None
