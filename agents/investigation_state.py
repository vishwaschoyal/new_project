"""Investigation obligations and loop-discipline checks.

A model asked "where is X configured **and** what happens when it is missing?"
will cheerfully answer the first half and stop. This module makes that
structurally difficult: the question is decomposed into explicit obligations,
and finalisation is challenged while any remain open.

It also owns one of the loop's anti-thrash rules: a tool call repeated argument
for argument. The other two live where the knowledge they need lives — reads
that overlap what is already in context are caught by the orchestrator against
the evidence ledger, and citations to unobserved lines by the ledger itself.
Each of those wastes budget and, worse, produces confident answers built on
nothing.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field, asdict
from typing import Any

MAX_OBLIGATIONS = 12
# One challenge is discipline; repeated challenges are an infinite loop. After
# this many pushbacks the loop accepts a best-effort answer and says so.
MAX_PREMATURE_CHALLENGES = 2

_MULTIPART_CONNECTORS = re.compile(
    r"\b(?:and also|and then|as well as|additionally|also,)\b", re.IGNORECASE
)
_ENUMERATION = re.compile(r"(?:^|\n)\s*(?:\d+[.)]|[-*])\s+(.{6,})")


@dataclass
class Obligation:
    """One thing that must be established before the question is answered."""

    id: str
    question: str
    status: str = "open"          # open | answered
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    origin: str = "question"      # question | discovered

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decompose_question(question: str, *, max_parts: int = MAX_OBLIGATIONS) -> list[str]:
    """Split a question into the parts an answer must cover.

    Intentionally conservative. Over-splitting invents obligations the user never
    asked about and burns budget proving them; under-splitting is corrected at
    runtime because the model may register obligations it discovers while
    tracing.
    """
    text = (question or "").strip()
    if not text:
        return []

    bullets = [m.group(1).strip() for m in _ENUMERATION.finditer(text)]
    if len(bullets) >= 2:
        return bullets[:max_parts]

    # Sentences that are themselves questions.
    parts = [p.strip() for p in re.split(r"(?<=\?)\s+", text) if p.strip()]
    if len(parts) >= 2:
        return parts[:max_parts]

    # A single sentence joined by an explicit connector.
    joined = _MULTIPART_CONNECTORS.split(text)
    if len(joined) >= 2:
        segments = [seg.strip(" ,;") for seg in joined if len(seg.strip()) > 12]
        if len(segments) >= 2:
            return segments[:max_parts]

    return [text]


class InvestigationState:
    """Request-local investigation bookkeeping. Thread-safe for subagent use."""

    def __init__(self, question: str):
        self.question = question
        self._obligations: dict[str, Obligation] = {}
        self._call_signatures: dict[str, int] = {}
        self._premature_challenges = 0
        self._lock = threading.RLock()

        for index, part in enumerate(decompose_question(question), start=1):
            self.add_obligation(part, origin="question", key=f"q{index}")

    # -- obligations -----------------------------------------------------
    def add_obligation(self, question: str, *, origin: str = "discovered", key: str | None = None) -> str | None:
        text = (question or "").strip()
        if len(text) < 6:
            return None
        with self._lock:
            if len(self._obligations) >= MAX_OBLIGATIONS:
                return None
            for existing in self._obligations.values():
                if existing.question.lower() == text.lower():
                    return existing.id
            obligation_id = key or f"d{len(self._obligations) + 1}"
            self._obligations[obligation_id] = Obligation(
                id=obligation_id, question=text, origin=origin
            )
            return obligation_id

    def resolve(self, obligation_id: str, *, answer: str, citations: list[str]) -> bool:
        with self._lock:
            obligation = self._obligations.get(obligation_id)
            if obligation is None:
                return False
            obligation.status = "answered"
            obligation.answer = answer
            obligation.citations = list(citations)
            return True

    def resolve_best_match(self, *, answer: str, citations: list[str]) -> str | None:
        """Attach a finding to the open obligation it most plausibly settles.

        The model records findings in its own words rather than quoting an
        obligation ID, so matching is by word overlap. A finding that matches
        nothing still counts as progress — it just does not close an obligation.
        """
        with self._lock:
            open_ones = [o for o in self._obligations.values() if o.status == "open"]
        if not open_ones:
            return None

        finding_words = _significant_words(answer)
        best, best_score = None, 0.0
        for obligation in open_ones:
            obligation_words = _significant_words(obligation.question)
            if not obligation_words:
                continue
            overlap = len(finding_words & obligation_words) / len(obligation_words)
            if overlap > best_score:
                best, best_score = obligation, overlap

        if best is not None and best_score >= 0.34:
            self.resolve(best.id, answer=answer, citations=citations)
            return best.id
        if len(open_ones) == 1 and citations:
            self.resolve(open_ones[0].id, answer=answer, citations=citations)
            return open_ones[0].id
        return None

    @property
    def open_obligations(self) -> list[Obligation]:
        with self._lock:
            return [o for o in self._obligations.values() if o.status == "open"]

    @property
    def obligations(self) -> list[Obligation]:
        with self._lock:
            return list(self._obligations.values())

    @property
    def is_complete(self) -> bool:
        return not self.open_obligations

    # -- finalisation discipline -----------------------------------------
    def challenge_finalisation(self) -> str | None:
        """Return a pushback message, or None if finalising is acceptable now."""
        outstanding = self.open_obligations
        if not outstanding:
            return None
        with self._lock:
            if self._premature_challenges >= MAX_PREMATURE_CHALLENGES:
                return None
            self._premature_challenges += 1

        listed = "\n".join(f"- {o.question}" for o in outstanding)
        return (
            "You have not yet established every part of the question. Still open:\n"
            f"{listed}\n\n"
            "Investigate these with tools and call record_finding for each, or, if one "
            "genuinely cannot be determined from this repository, say so explicitly in "
            "your final answer rather than leaving it unaddressed."
        )

    @property
    def was_challenged(self) -> bool:
        with self._lock:
            return self._premature_challenges > 0

    # -- anti-thrash -----------------------------------------------------
    def register_call(self, name: str, arguments: dict[str, Any]) -> str | None:
        """Record a tool call; return a warning if it repeats an earlier one."""
        signature = f"{name}:{_stable_signature(arguments)}"
        with self._lock:
            count = self._call_signatures.get(signature, 0) + 1
            self._call_signatures[signature] = count
        if count > 1:
            return (
                f"You already ran this exact {name} call earlier in this investigation "
                f"({count - 1}x). Its result has not changed. Use what you already found, "
                f"or search differently."
            )
        return None

    def render_status(self) -> str:
        """Compact obligation block injected into the working context."""
        with self._lock:
            obligations = list(self._obligations.values())
        if not obligations:
            return ""
        lines = []
        for obligation in obligations:
            marker = "[x]" if obligation.status == "answered" else "[ ]"
            lines.append(f"{marker} {obligation.question}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "obligations": [o.to_dict() for o in self.obligations],
            "complete": self.is_complete,
        }


_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "of",
        "to", "in", "on", "for", "and", "or", "but", "with", "at", "by", "from",
        "as", "it", "its", "this", "that", "these", "those", "what", "when",
        "where", "which", "who", "how", "why", "does", "do", "did", "can", "could",
        "should", "would", "will", "if", "then", "than", "there", "here", "into",
        "about", "any", "all", "not", "no", "so", "such", "you", "your", "we",
    }
)


def _significant_words(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS}


def _stable_signature(arguments: dict[str, Any]) -> str:
    """Order-independent argument signature for duplicate detection."""
    if not arguments:
        return ""
    return "|".join(f"{k}={str(arguments[k]).strip()}" for k in sorted(arguments))
