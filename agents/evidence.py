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

Observation alone is a weaker guarantee than it sounds, though. It proves the
model *looked* at those coordinates, not that the sentence attached to them is
about what is there — and a model that read `app.js:520-640` while hunting for
`send()` will cheerfully report that `send()` is defined there. So a citation is
also checked against the file on disk: the identifiers the claim names must
actually appear near the lines it points at. That check only ever rejects; when
it cannot form an opinion it stays silent, because stripping a true citation
costs the user more than letting a vague one through.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

from core.errors import SafetyError
from core.safety import resolve_in_workspace

# Matches `path/to/file.py:120` and `path/to/file.py:120-160`.
CITATION_RE = re.compile(
    r"\b(?P<path>[A-Za-z0-9_./\-]+\.[A-Za-z0-9_]+):(?P<start>\d+)(?:\s*-\s*(?P<end>\d+))?\b"
)

# How far off a citation may land and still be considered to describe the code
# it points at. Citing a function's body rather than its `def` line is normal
# and correct; being sixty lines away is not.
CITATION_LINE_SLACK = 10

# What counts as a checkable token in an answer. Prose cannot be confirmed
# against a line range, so only things written like code are extracted — and
# `name(` is deliberately written without allowing a space, because "the
# function (which…)" is English and `send(` is not.
_CODE_TOKEN_PATTERNS = (
    re.compile(r"`([^`\n]{2,60})`"),                            # `explicitly quoted`
    re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\("),                # call or definition
    re.compile(r"\b([A-Za-z_]+(?:_[A-Za-z0-9]+)+)\b"),          # snake_case
    re.compile(r"\b([a-z]+[A-Z][A-Za-z0-9]*)\b"),               # camelCase
    re.compile(r"\b([A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+)\b"),  # PascalCase
)

# Words that reach the patterns above without being identifiers.
_NOT_IDENTIFIERS = frozenset(
    {"function", "method", "class", "module", "file", "line", "lines", "this",
     "that", "the", "and", "for", "with", "see", "note", "step", "call", "calls",
     "returns", "return", "which", "where", "when", "then", "from", "into"}
)


def _identifiers(text: str) -> set[str]:
    """Code-like names mentioned in a claim."""
    found: set[str] = set()
    for pattern in _CODE_TOKEN_PATTERNS:
        for match in pattern.finditer(text or ""):
            head = re.match(r"[A-Za-z_][A-Za-z0-9_]*", match.group(1).strip())
            if not head:
                continue
            word = head.group(0)
            if len(word) >= 3 and word.lower() not in _NOT_IDENTIFIERS:
                found.add(word)
    return found


def _whole_identifier(haystack: str, name: str) -> bool:
    """Whether ``name`` appears as a complete identifier, not inside a longer one.

    Plain substring matching is far too generous: searching for `send` in a file
    finds the word "sends" in a comment and concludes the citation was fine.
    """
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", haystack) is not None


def _mentions(haystack: str, name: str) -> bool:
    if _whole_identifier(haystack, name):
        return True
    # An answer says "AgentEvents" about the AgentEvent class often enough that
    # the plural should not read as a different symbol.
    return (
        name.endswith("s")
        and len(name) > 4
        and _whole_identifier(haystack, name[:-1])
    )


def _claim_segments(answer: str) -> list[tuple[str, str]]:
    """Each line of an answer paired with the line before it.

    A line is the right unit: answers put a claim and its citation together on
    one bullet. The previous line is carried along because a citation sometimes
    trails on its own after the sentence that earned it, and judging that line
    alone would find nothing to check.
    """
    lines = (answer or "").splitlines()
    return [(line, lines[index - 1] if index else "") for index, line in enumerate(lines)]


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

    def __init__(self, *, workspace: Path | None = None, max_records: int = 60):
        # Without a workspace the ledger still verifies coordinates; it just
        # cannot open the files to check what is written at them.
        self._workspace = Path(workspace) if workspace else None
        self._observed: dict[str, list[tuple[int, int]]] = {}
        # File lengths as the read tool reported them. A request for lines
        # 5-255 of a 16-line file asks for 14 new lines, not 250, and judging
        # redundancy without that would call a total re-read a fresh one.
        self._lengths: dict[str, int] = {}
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
            total = result_metadata.get("total_lines")
            if path and start:
                self.record_observation(str(path), int(start), int(end or start))
            if path and total:
                with self._lock:
                    self._lengths[str(path)] = int(total)

        elif tool == "grep":
            # Each hit proves that one specific line exists and was seen.
            for line in content.splitlines():
                parts = line.split(":", 2)
                if len(parts) >= 2 and parts[1].isdigit():
                    self.record_observation(parts[0], int(parts[1]), int(parts[1]))

        elif tool == "create":
            # A file the agent just wrote is observed in the strongest sense
            # available: it authored every line. Without this, describing the
            # change it made would fail citation checking as if it had guessed.
            path = result_metadata.get("path")
            lines = result_metadata.get("lines")
            if path and lines:
                self.record_observation(str(path), 1, int(lines))

    def was_observed(self, path: str, start_line: int, end_line: int | None = None) -> bool:
        end_line = end_line or start_line
        with self._lock:
            for begin, finish in self._observed.get(path, ()):
                if begin <= start_line and end_line <= finish:
                    return True
        return False

    def coverage_of(self, path: str, start_line: int, end_line: int) -> float:
        """Fraction of ``start..end`` that has already been observed.

        ``was_observed`` asks a yes/no question about full containment, which is
        the right test for a citation. Deciding whether a *read* is worth paying
        for needs the shape of the overlap instead: a request that is 90% lines
        already in context is thrash even though the remaining tail means it is
        not fully contained.
        """
        if not path or start_line < 1:
            return 0.0
        end_line = max(start_line, end_line)
        with self._lock:
            ranges = list(self._observed.get(path, ()))
            known_length = self._lengths.get(path)

        if known_length:
            if start_line > known_length:
                return 0.0  # let the read tool report the offset itself
            end_line = min(end_line, known_length)
        requested = end_line - start_line + 1

        # Stored ranges are merged, so they never overlap each other and the
        # overlaps can simply be summed.
        seen = 0
        for begin, finish in ranges:
            overlap = min(end_line, finish) - max(start_line, begin) + 1
            if overlap > 0:
                seen += overlap
        return seen / requested

    def first_unobserved_line(self, path: str, start_line: int, end_line: int) -> int | None:
        """Where a read of ``start..end`` would start returning new lines."""
        if not path or start_line < 1:
            return start_line
        end_line = max(start_line, end_line)
        with self._lock:
            ranges = sorted(self._observed.get(path, ()))
            known_length = self._lengths.get(path)
        if known_length:
            end_line = min(end_line, known_length)

        line = start_line
        for begin, finish in ranges:
            if finish < line:
                continue
            if begin > line:
                break
            line = finish + 1
        return line if line <= end_line else None

    def observed_spans(self, path: str) -> str:
        """Human-readable list of what has been read of one file."""
        with self._lock:
            ranges = list(self._observed.get(path, ()))
        return ", ".join(f"{a}-{b}" for a, b in ranges)

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
        # File lengths come along too, or the parent's redundancy check loses
        # precision on exactly the files a worker just covered for it.
        with other._lock:
            lengths = dict(other._lengths)
        with self._lock:
            self._lengths.update(lengths)
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

    def _file_lines(self, path: str, cache: dict[str, list[str] | None]) -> list[str] | None:
        """The file behind a citation, or None if it cannot be read.

        Goes through `resolve_in_workspace` because the path came out of model
        output: a citation naming `.env` or `../secrets` is refused here exactly
        as it would be for a tool.
        """
        if path not in cache:
            lines: list[str] | None = None
            if self._workspace is not None:
                try:
                    target = resolve_in_workspace(self._workspace, path)
                    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
                except (SafetyError, OSError, ValueError):
                    lines = None
            cache[path] = lines
        return cache[path]

    def _contradicted(
        self, citation: Citation, claim: str, cache: dict[str, list[str] | None]
    ) -> bool:
        """Whether the cited lines fail to contain anything the claim names.

        Deliberately hard to trigger. One matching identifier anywhere in the
        window is enough, and a claim with no identifiers at all is left alone.
        A false rejection deletes a true citation from the user's answer, which
        is a worse failure than the one being prevented.
        """
        lines = self._file_lines(citation.path, cache)
        if lines is None:
            return False

        names = _identifiers(CITATION_RE.sub(" ", claim))
        if not names:
            return False

        start = max(1, citation.start_line - CITATION_LINE_SLACK)
        if start > len(lines):
            return True  # cited past the end of the file entirely
        end = min(len(lines), citation.end_line + CITATION_LINE_SLACK)
        window = "\n".join(lines[start - 1 : end])
        return not any(_mentions(window, name) for name in names)

    def verify_answer(self, answer: str) -> dict[str, Any]:
        """Sort an answer's citations into supported, unsupported, contradicted.

        *Unsupported* means the lines were never observed during this request.
        *Contradicted* means they were observed, but nothing the claim names is
        written at them — the citation points somewhere real and irrelevant.
        Both are stripped by the caller; they are reported apart because they
        are different failures and only one of them is the model inventing.
        """
        supported: list[Citation] = []
        unsupported: list[Citation] = []
        contradicted: list[Citation] = []
        cache: dict[str, list[str] | None] = {}
        seen: set[tuple[str, int, int]] = set()

        for line, previous in _claim_segments(answer):
            for citation in self.extract_citations(line):
                key = (citation.path, citation.start_line, citation.end_line)
                if key in seen:
                    continue
                seen.add(key)

                if not self.was_observed(citation.path, citation.start_line, citation.end_line):
                    unsupported.append(citation)
                elif self._contradicted(citation, line, cache) and self._contradicted(
                    citation, f"{previous}\n{line}", cache
                ):
                    # Checked twice: once against the line, once with the line
                    # before it. A trailing citation keeps its context that way.
                    contradicted.append(citation)
                else:
                    supported.append(citation)

        return {
            "supported": [c.to_dict() for c in supported],
            "unsupported": [c.to_dict() for c in unsupported],
            "contradicted": [c.to_dict() for c in contradicted],
            "all_supported": not unsupported and not contradicted,
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
