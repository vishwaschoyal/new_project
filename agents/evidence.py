"""The evidence ledger.

Working context is disposable; evidence is not. During a long investigation the
raw tool observations are compacted away to control cost, but the *coordinates*
the answer depends on must survive. That separation is the reason this is its
own structure rather than a slice of the message list.

The ledger also enforces the project's central honesty rule: **a citation is
only valid if those exact lines were observed during this request.** Every
successful ``read`` records an observed range; every ``grep`` hit records an
observed line. At finalisation, citations are checked against that record and
unsupported ones are rejected rather than shown to the user.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

# Matches `path/to/file.py:120` and `path/to/file.py:120-160`.
CITATION_RE = re.compile(
    r"\b(?P<path>[A-Za-z0-9_./\-]+\.[A-Za-z0-9_]+):(?P<start>\d+)(?:\s*-\s*(?P<end>\d+))?\b"
)


@dataclass(frozen=True)
class Citation:
    path: str
    start_line: int
    end_line: int

    def __str__(self) -> str:
        if self.end_line > self.start_line:
            return f"{self.path}:{self.start_line}-{self.end_line}"
        return f"{self.path}:{self.start_line}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceRecord:
    """One verified claim, anchored to lines that were actually read."""

    claim: str
    citation: Citation
    snippet: str = ""
    # "main" or a subagent's name — preserved through fan-in so the user can
    # see which investigation produced a finding.
    source: str = "main"

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "citation": self.citation.to_dict(),
            "reference": str(self.citation),
            "snippet": self.snippet,
            "source": self.source,
        }


class EvidenceLedger:
    """Observed ranges plus verified claims for one request."""

    def __init__(self, *, max_records: int = 60):
        self._observed: dict[str, list[tuple[int, int]]] = {}
        self._records: list[EvidenceRecord] = []
        self._max_records = max_records
        self._lock = threading.Lock()

    # -- observation -----------------------------------------------------
    def record_observation(self, path: str, start_line: int, end_line: int) -> None:
        """Note that ``path`` lines ``start..end`` were genuinely seen."""
        if not path or start_line < 1:
            return
        end_line = max(start_line, end_line)
        with self._lock:
            ranges = self._observed.setdefault(path, [])
            ranges.append((start_line, end_line))
            self._observed[path] = _merge_ranges(ranges)

    def observe_tool_result(self, tool: str, result_metadata: dict[str, Any], content: str) -> None:
        """Extract observed coordinates from a successful tool result."""
        if tool == "read":
            path = result_metadata.get("path")
            start = result_metadata.get("start_line")
            end = result_metadata.get("end_line")
            if path and start:
                self.record_observation(str(path), int(start), int(end or start))

        elif tool == "grep":
            # Each hit proves that one specific line exists and was seen.
            for line in content.splitlines():
                parts = line.split(":", 2)
                if len(parts) >= 2 and parts[1].isdigit():
                    self.record_observation(parts[0], int(parts[1]), int(parts[1]))

    def was_observed(self, path: str, start_line: int, end_line: int | None = None) -> bool:
        end_line = end_line or start_line
        with self._lock:
            for begin, finish in self._observed.get(path, ()):
                if begin <= start_line and end_line <= finish:
                    return True
        return False

    @property
    def observed_files(self) -> list[str]:
        with self._lock:
            return sorted(self._observed)

    def observed_ranges(self) -> dict[str, list[tuple[int, int]]]:
        with self._lock:
            return {path: list(ranges) for path, ranges in self._observed.items()}

    # -- records ---------------------------------------------------------
    def add(self, record: EvidenceRecord) -> bool:
        """Store a verified claim. Refuses citations that were never observed."""
        if not self.was_observed(
            record.citation.path, record.citation.start_line, record.citation.end_line
        ):
            return False
        with self._lock:
            # Replace an identical citation rather than accumulating duplicates.
            for index, existing in enumerate(self._records):
                if existing.citation == record.citation and existing.source == record.source:
                    self._records[index] = record
                    return True
            self._records.append(record)
            if len(self._records) > self._max_records:
                self._records.pop(0)
        return True

    def merge(self, other: "EvidenceLedger") -> int:
        """Fan-in: absorb a subagent's observations and records.

        Observations are merged first so the subagent's records still validate
        against the combined ledger — otherwise every worker finding would be
        rejected as unobserved by the main agent.
        """
        for path, ranges in other.observed_ranges().items():
            for start, end in ranges:
                self.record_observation(path, start, end)
        added = 0
        for record in other.records:
            if self.add(record):
                added += 1
        return added

    @property
    def records(self) -> list[EvidenceRecord]:
        with self._lock:
            return list(self._records)

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    # -- citation verification -------------------------------------------
    def extract_citations(self, text: str) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[tuple[str, int, int]] = set()
        for match in CITATION_RE.finditer(text or ""):
            start = int(match.group("start"))
            end = int(match.group("end") or start)
            key = (match.group("path"), start, end)
            if key in seen:
                continue
            seen.add(key)
            citations.append(Citation(match.group("path"), start, max(start, end)))
        return citations

    def verify_answer(self, answer: str) -> dict[str, Any]:
        """Split an answer's citations into supported and unsupported."""
        supported: list[Citation] = []
        unsupported: list[Citation] = []
        for citation in self.extract_citations(answer):
            target = supported if self.was_observed(
                citation.path, citation.start_line, citation.end_line
            ) else unsupported
            target.append(citation)
        return {
            "supported": [c.to_dict() for c in supported],
            "unsupported": [c.to_dict() for c in unsupported],
            "all_supported": not unsupported,
        }

    # -- prompt rendering -------------------------------------------------
    def render_for_prompt(self, *, limit: int = 30) -> str:
        """Compact evidence block that replaces trimmed raw observations."""
        records = self.records[-limit:]
        if not records:
            return "No evidence recorded yet."
        lines = []
        for record in records:
            attribution = "" if record.source == "main" else f" [via {record.source}]"
            lines.append(f"- {record.claim} ({record.citation}){attribution}")
        return "\n".join(lines)

    def render_observed_files(self, *, limit: int = 40) -> str:
        ranges = self.observed_ranges()
        if not ranges:
            return "Nothing read yet."
        lines = []
        for path in sorted(ranges)[:limit]:
            spans = ", ".join(f"{a}-{b}" for a, b in ranges[path][:6])
            lines.append(f"- {path}: {spans}")
        return "\n".join(lines)


def _merge_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Coalesce overlapping/adjacent ranges so membership checks stay cheap."""
    ordered = sorted(ranges)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged
