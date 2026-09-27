"""
code_diff.py
============
Comparison layer, part 1 — line-level diff engine.

Given two code snippets, this reports:
  - added lines / deleted lines / modified lines
  - changed variables   (e.g. max_value -> largest, arr -> numbers)
  - changed operators   (e.g. `>` -> `<`, `+` -> `-` on an otherwise identical line)
  - changed statements  (e.g. a `for` line replaced by a `while` line)

Self-contained: stdlib only (ast, difflib, tokenize, keyword, io).
"""

from __future__ import annotations

import difflib
import io
import keyword
import tokenize
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SideBySideRow:
    left: Optional[str]
    right: Optional[str]
    op: str  # "equal" | "modified" | "added" | "deleted"
    left_no: Optional[int] = None
    right_no: Optional[int] = None


@dataclass
class OperatorChange:
    left_line_no: int
    right_line_no: int
    old_operator: str
    new_operator: str


@dataclass
class StatementChange:
    left_line_no: int
    right_line_no: int
    left_text: str
    right_text: str
    reason: str


@dataclass
class DiffReport:
    rows: list[SideBySideRow] = field(default_factory=list)
    added_lines: list[str] = field(default_factory=list)
    deleted_lines: list[str] = field(default_factory=list)
    modified_lines: list[tuple[str, str]] = field(default_factory=list)
    changed_variables: dict[str, str] = field(default_factory=dict)
    changed_operators: list[OperatorChange] = field(default_factory=list)
    changed_statements: list[StatementChange] = field(default_factory=list)
    unchanged_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {
            "rows": [r.__dict__ for r in self.rows],
            "added_lines": self.added_lines,
            "deleted_lines": self.deleted_lines,
            "modified_lines": [{"before": a, "after": b} for a, b in self.modified_lines],
            "changed_variables": self.changed_variables,
            "changed_operators": [c.__dict__ for c in self.changed_operators],
            "changed_statements": [c.__dict__ for c in self.changed_statements],
            "unchanged_ratio": round(self.unchanged_ratio, 4),
        }


# ---------------------------------------------------------------------------
# Tokenizing helpers
# ---------------------------------------------------------------------------

_STMT_KEYWORDS = {
    "for": "for-loop", "while": "while-loop", "if": "conditional",
    "elif": "conditional", "else": "conditional", "def": "function definition",
    "return": "return statement", "try": "try-block", "except": "except-block",
    "with": "with-block", "class": "class definition",
}


def _line_tokens(line: str) -> list[tokenize.TokenInfo]:
    try:
        return [t for t in tokenize.generate_tokens(io.StringIO(line).readline)
                if t.type not in (tokenize.ENCODING, tokenize.NEWLINE, tokenize.NL,
                                   tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT)]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []


def _leading_keyword(line: str) -> Optional[str]:
    stripped = line.strip()
    for kw in _STMT_KEYWORDS:
        if stripped == kw or stripped.startswith(kw + " ") or stripped.startswith(kw + ":") or stripped.startswith(kw + "("):
            return kw
    return None


def _identifiers_in_order(source: str) -> list[str]:
    names: list[str] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.NAME and not keyword.iskeyword(tok.string):
                names.append(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        import re
        names = [w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", source) if not keyword.iskeyword(w)]
    return names


def _infer_variable_renames(code1: str, code2: str) -> dict[str, str]:
    """Positional first-appearance alignment of identifiers across the two
    snippets -- surfaces obvious renames like max_value -> largest,
    arr -> numbers without requiring full semantic equivalence proof."""
    left, right = [], []
    seen = set()
    for n in _identifiers_in_order(code1):
        if n not in seen:
            left.append(n)
            seen.add(n)
    seen = set()
    for n in _identifiers_in_order(code2):
        if n not in seen:
            right.append(n)
            seen.add(n)

    mapping = {}
    for l, r in zip(left, right):
        if l != r:
            mapping[l] = r
    return mapping


def _classify_replace(left_line: str, right_line: str, l_no: int, r_no: int
                       ) -> tuple[Optional[OperatorChange], Optional[StatementChange]]:
    """Decides whether a replaced line pair is best explained by an
    operator swap, a statement-type change, or neither (handled elsewhere
    as a generic modified line)."""
    l_kw, r_kw = _leading_keyword(left_line), _leading_keyword(right_line)
    if l_kw and r_kw and l_kw != r_kw:
        return None, StatementChange(
            l_no, r_no, left_line.strip(), right_line.strip(),
            f"{_STMT_KEYWORDS[l_kw]} replaced with {_STMT_KEYWORDS[r_kw]}",
        )

    l_toks = [t for t in _line_tokens(left_line) if t.type == tokenize.OP]
    r_toks = [t for t in _line_tokens(right_line) if t.type == tokenize.OP]
    l_names = [t for t in _line_tokens(left_line) if t.type == tokenize.NAME]
    r_names = [t for t in _line_tokens(right_line) if t.type == tokenize.NAME]

    if len(l_toks) == len(r_toks) and len(l_names) == len(r_names):
        op_diffs = [(a.string, b.string) for a, b in zip(l_toks, r_toks) if a.string != b.string]
        # Exactly one differing operator, rest of the shape identical ->
        # this is an operator swap (e.g. `>` vs `<`, `+` vs `-`).
        if len(op_diffs) == 1:
            old_op, new_op = op_diffs[0]
            return OperatorChange(l_no, r_no, old_op, new_op), None

    return None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_diff(code1: str, code2: str) -> DiffReport:
    """
    Feature 1 — full line-level comparison between two snippets.

    Example
    -------
    >>> r = compare_diff("max_value = arr[0]\\nfor x in arr:\\n    pass",
    ...                   "largest = numbers[0]\\nfor value in numbers:\\n    pass")
    >>> r.changed_variables
    {'max_value': 'largest', 'arr': 'numbers', 'x': 'value'}
    """
    left_lines = code1.splitlines()
    right_lines = code2.splitlines()
    matcher = difflib.SequenceMatcher(a=left_lines, b=right_lines, autojunk=False)

    report = DiffReport(unchanged_ratio=matcher.ratio())

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for li, rj in zip(range(i1, i2), range(j1, j2)):
                report.rows.append(SideBySideRow(left_lines[li], right_lines[rj], "equal", li + 1, rj + 1))

        elif tag == "delete":
            for li in range(i1, i2):
                report.rows.append(SideBySideRow(left_lines[li], None, "deleted", li + 1, None))
                report.deleted_lines.append(left_lines[li])

        elif tag == "insert":
            for rj in range(j1, j2):
                report.rows.append(SideBySideRow(None, right_lines[rj], "added", None, rj + 1))
                report.added_lines.append(right_lines[rj])

        elif tag == "replace":
            span = max(i2 - i1, j2 - j1)
            for k in range(span):
                li = i1 + k if i1 + k < i2 else None
                rj = j1 + k if j1 + k < j2 else None
                left_text = left_lines[li] if li is not None else None
                right_text = right_lines[rj] if rj is not None else None

                if left_text is None:
                    report.rows.append(SideBySideRow(None, right_text, "added", None, rj + 1))
                    report.added_lines.append(right_text)
                    continue
                if right_text is None:
                    report.rows.append(SideBySideRow(left_text, None, "deleted", li + 1, None))
                    report.deleted_lines.append(left_text)
                    continue

                report.rows.append(SideBySideRow(left_text, right_text, "modified", li + 1, rj + 1))
                report.modified_lines.append((left_text, right_text))

                op_change, stmt_change = _classify_replace(left_text, right_text, li + 1, rj + 1)
                if op_change:
                    report.changed_operators.append(op_change)
                if stmt_change:
                    report.changed_statements.append(stmt_change)

    report.changed_variables = _infer_variable_renames(code1, code2)
    return report
