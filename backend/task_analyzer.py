"""
task_analyzer.py
=================
Comparison layer, part 3 — optional problem-statement support.

Lets the caller optionally supply what the assignment actually asked for,
e.g. "Find the largest element in an array.", and keeps two questions
cleanly separate:

    Problem/task similarity      -- do these two snippets look like they're
                                     solving the same kind of problem?
    Implementation similarity    -- (computed elsewhere, in approach_analyzer
                                     / report_generator) how similar is HOW
                                     they solved it?

This is what lets CodeXray distinguish:
    "same task + different implementation"        (expected, healthy)
  from
    "same task + suspiciously similar implementation"

IMPORTANT: CodeXray cannot know the true intended task with certainty from
code alone -- every task guess here is a heuristic label, not a fact, and
is reported with that caveat.

Self-contained: stdlib only (re). Uses approach_analyzer for structural
features/labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from approach_analyzer import ApproachMatch, CodeFeatures, detect_approach


# ---------------------------------------------------------------------------
# Task-category guessing from detected approach + structural features
# ---------------------------------------------------------------------------

_APPROACH_TO_TASK = {
    "Bubble Sort": "Sorting",
    "Selection Sort": "Sorting",
    "Insertion Sort": "Sorting",
    "Binary Search": "Searching",
    "Linear Search": "Searching",
    "Recursive Factorial": "Factorial",
    "Iterative Factorial": "Factorial",
    "Iterative Sum": "Summation",
    "Reverse via Slicing": "Reversal",
    "Iterative/Two-Pointer Reverse": "Reversal",
}

_BUILTIN_TO_TASK = {
    "max": "Maximum/Minimum Search", "min": "Maximum/Minimum Search",
    "sum": "Summation", "sorted": "Sorting", "reversed": "Reversal",
}


def _guess_task(match: ApproachMatch) -> tuple[str, str]:
    """Returns (task_guess, confidence_word). Confidence is intentionally
    coarse -- this is a heuristic label, never a certain classification."""
    if match.label in _APPROACH_TO_TASK:
        return _APPROACH_TO_TASK[match.label], "Likely"

    f = match.features
    for name, task in _BUILTIN_TO_TASK.items():
        if name in f.builtin_calls:
            return task, "Likely"

    if match.label == "Iterative Manual Computation":
        if f.max_loop_nesting == 1 and f.branch_count >= 1 and \
                ("Gt" in f.comparison_ops or "Lt" in f.comparison_ops):
            return "Maximum/Minimum Search", "Uncertain"
        return "Generic Iterative Task", "Uncertain"

    if match.label == "Recursive / Divide & Conquer":
        return "Recursive Computation", "Uncertain"

    if match.label == "Simple Sequential Code":
        return "Simple Computation / Assignment", "Uncertain"

    return "Unknown", "Uncertain"


# ---------------------------------------------------------------------------
# Problem-statement keyword handling
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "that", "this", "with", "from", "given", "find", "write", "return",
    "using", "your", "you", "be", "it", "its", "as", "each", "which",
}


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_]*", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


_TASK_HINT_WORDS = {
    "Sorting": {"sort", "sorted", "order", "ordering", "ascending", "descending"},
    "Searching": {"search", "find", "locate", "lookup", "index"},
    "Factorial": {"factorial"},
    "Summation": {"sum", "total", "add", "addition"},
    "Reversal": {"reverse", "reversed", "flip"},
    "Maximum/Minimum Search": {"largest", "maximum", "max", "smallest", "minimum", "min", "greatest"},
}


def _statement_task_hint(keywords: set[str]) -> Optional[str]:
    best, best_overlap = None, 0
    for task, hints in _TASK_HINT_WORDS.items():
        overlap = len(keywords & hints)
        if overlap > best_overlap:
            best, best_overlap = task, overlap
    return best


# ---------------------------------------------------------------------------
# Public data model / API
# ---------------------------------------------------------------------------

@dataclass
class TaskAnalysis:
    task_guess_1: str
    task_confidence_1: str
    task_guess_2: str
    task_confidence_2: str
    problem_statement: Optional[str]
    problem_keywords: set[str] = field(default_factory=set)
    statement_task_hint: Optional[str] = None
    task_similarity_label: str = "Uncertain"
    task_similarity_agrees_with_statement: Optional[bool] = None
    disclaimer: str = (
        "Task/objective is inferred heuristically from code structure and cannot "
        "be claimed with certainty -- treat this as a supporting signal, not a fact."
    )

    def to_dict(self) -> dict:
        return {
            "task_guess_1": self.task_guess_1,
            "task_confidence_1": self.task_confidence_1,
            "task_guess_2": self.task_guess_2,
            "task_confidence_2": self.task_confidence_2,
            "problem_statement": self.problem_statement,
            "problem_keywords": sorted(self.problem_keywords),
            "statement_task_hint": self.statement_task_hint,
            "task_similarity_label": self.task_similarity_label,
            "task_similarity_agrees_with_statement": self.task_similarity_agrees_with_statement,
            "disclaimer": self.disclaimer,
        }


def analyze_task(code1: str, code2: str, problem_statement: Optional[str] = None) -> TaskAnalysis:
    """
    Feature 3 — task/objective analysis, kept deliberately separate from
    implementation-similarity scoring (see approach_analyzer / report_generator).
    """
    m1, m2 = detect_approach(code1), detect_approach(code2)
    task1, conf1 = _guess_task(m1)
    task2, conf2 = _guess_task(m2)

    same_guess = task1 == task2 and task1 not in ("Unknown",)
    if same_guess:
        task_similarity_label = "Likely Same Task"
    elif "Uncertain" in (conf1, conf2) or task1 == "Unknown" or task2 == "Unknown":
        task_similarity_label = "Uncertain"
    else:
        task_similarity_label = "Likely Different Tasks"

    kw = set()
    hint = None
    agrees = None
    stmt = problem_statement.strip() if problem_statement and problem_statement.strip() else None
    if stmt:
        kw = _keywords(stmt)
        hint = _statement_task_hint(kw)
        if hint is not None:
            agrees = (hint == task1) or (hint == task2)

    return TaskAnalysis(
        task_guess_1=task1, task_confidence_1=conf1,
        task_guess_2=task2, task_confidence_2=conf2,
        problem_statement=stmt, problem_keywords=kw,
        statement_task_hint=hint,
        task_similarity_label=task_similarity_label,
        task_similarity_agrees_with_statement=agrees,
    )
