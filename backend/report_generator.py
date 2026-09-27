"""
report_generator.py
====================
Comparison layer, part 4 — the single structured output that becomes the
interface between this Python backend and the Streamlit frontend:

    {
        "overall_similarity": ...,
        "text_similarity": ...,
        "ast_similarity": ...,
        "semantic_similarity": ...,
        "ai_indicator_code1": ...,
        "ai_indicator_code2": ...,
        "same_objective": ...,
        "same_approach": ...,
        "detected_patterns": [...],
        "code1_statistics": {...},
        "code2_statistics": {...}
    }

Run this file directly (`python report_generator.py`) to execute the demo
test-pair suite requested for the CodeXray viva/demo: same task + same
approach, same task + different approach, different tasks + similar
syntax, variable-renamed code, reformatted code, and unrelated code --
plus the six named comparisons (max: loop vs max(), sorting: bubble vs
selection, search: linear vs binary, factorial: loop vs recursion,
sum: loop vs sum(), reverse: loop vs slicing).

Self-contained: stdlib only (ast, difflib, re, json). Imports the three
sibling modules in this same folder.
"""

from __future__ import annotations

import ast
import difflib
import json
import re
from typing import Any, Optional

from code_diff import compare_diff, DiffReport
from approach_analyzer import compare_approaches, ApproachComparison, CodeFeatures
from task_analyzer import analyze_task, TaskAnalysis


# ---------------------------------------------------------------------------
# Similarity scores: text / AST / semantic  (0..100)
# ---------------------------------------------------------------------------

def _normalize_text(code: str) -> str:
    lines = []
    for line in code.splitlines():
        line = re.sub(r"#.*$", "", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def text_similarity(code1: str, code2: str) -> float:
    a, b = _normalize_text(code1), _normalize_text(code2)
    if not a and not b:
        return 100.0
    return round(difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio() * 100, 2)


def _alpha_normalized_dump(source: str) -> str:
    """Identifier-blind, literal-blind AST dump so `a=10` and `x=10`
    compare as identical structure."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""

    rename: dict[str, str] = {}
    counter = 0

    class Normalizer(ast.NodeTransformer):
        def visit_Name(self, node):
            nonlocal counter
            if node.id not in rename:
                rename[node.id] = f"VAR{counter}"
                counter += 1
            node.id = rename[node.id]
            return node

        def visit_arg(self, node):
            nonlocal counter
            if node.arg not in rename:
                rename[node.arg] = f"VAR{counter}"
                counter += 1
            node.arg = rename[node.arg]
            return node

        def visit_Constant(self, node):
            if isinstance(node.value, (int, float)):
                node.value = 0
            elif isinstance(node.value, str):
                node.value = ""
            return node

    normalized = Normalizer().visit(tree)
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized)


def ast_similarity(code1: str, code2: str) -> float:
    d1, d2 = _alpha_normalized_dump(code1), _alpha_normalized_dump(code2)
    if not d1 and not d2:
        return 100.0
    if not d1 or not d2:
        return 0.0
    return round(difflib.SequenceMatcher(a=d1, b=d2, autojunk=False).ratio() * 100, 2)


def semantic_similarity(f1: CodeFeatures, f2: CodeFeatures) -> float:
    """Behavioral-overlap approximation from the shared structural feature
    vector (loop shape, recursion, accumulate/swap idioms, builtins)."""
    if not f1.parsed_ok and not f2.parsed_ok:
        return 100.0
    if not f1.parsed_ok or not f2.parsed_ok:
        return 0.0

    def norm(x, cap):
        return min(x, cap) / cap

    signals = [
        1.0 - abs(norm(f1.loop_count, 5) - norm(f2.loop_count, 5)),
        1.0 - abs(norm(f1.max_loop_nesting, 3) - norm(f2.max_loop_nesting, 3)),
        1.0 if f1.has_recursion == f2.has_recursion else 0.0,
        1.0 if f1.has_swap_pattern == f2.has_swap_pattern else 0.0,
        1.0 if f1.has_mult_accumulate == f2.has_mult_accumulate else 0.0,
        1.0 if f1.has_add_accumulate == f2.has_add_accumulate else 0.0,
        1.0 if f1.uses_negative_slice == f2.uses_negative_slice else 0.0,
        1.0 if f1.uses_for == f2.uses_for else 0.0,
        1.0 if f1.uses_while == f2.uses_while else 0.0,
        (len(f1.comparison_ops & f2.comparison_ops) / max(len(f1.comparison_ops | f2.comparison_ops), 1)
         if (f1.comparison_ops or f2.comparison_ops) else 1.0),
        (len(f1.builtin_calls & f2.builtin_calls) / max(len(f1.builtin_calls | f2.builtin_calls), 1)
         if (f1.builtin_calls or f2.builtin_calls) else 1.0),
    ]
    return round((sum(signals) / len(signals)) * 100, 2)


_TIERS = ((85.0, "Very High Similarity"), (65.0, "High Similarity"),
          (45.0, "Moderate Similarity"), (25.0, "Low Similarity"), (0.0, "Very Low Similarity"))


def _classify(score: float) -> str:
    for threshold, label in _TIERS:
        if score >= threshold:
            return label
    return _TIERS[-1][1]


# ---------------------------------------------------------------------------
# AI-generation indicator (heuristic style signal, NOT a detector)
# ---------------------------------------------------------------------------

def ai_indicator(code: str) -> dict:
    """
    Produces a rough 0-100 "AI-style-likelihood" score from surface style
    signals (docstrings/type hints, descriptive multi-word naming, comment
    density, consistent formatting, generic boilerplate variable names).

    This is a coarse heuristic based on style conventions, not a
    plagiarism or AI-detection verdict -- clean, well-commented human code
    will also score high here, and terse human code can score low. It
    should only ever be shown alongside that caveat.
    """
    lines = [l for l in code.splitlines() if l.strip()]
    if not lines:
        return {"score": 0, "confidence": "Uncertain", "signals": [], "disclaimer": _AI_DISCLAIMER}

    signals_hit = []
    score = 0.0

    has_docstring = '"""' in code or "'''" in code
    if has_docstring:
        score += 20
        signals_hit.append("docstring present")

    has_type_hints = bool(re.search(r":\s*(int|str|float|bool|list|dict|List|Dict|Optional)\b", code))
    if has_type_hints:
        score += 15
        signals_hit.append("type hints used")

    comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
    comment_ratio = comment_lines / len(lines)
    if comment_ratio > 0.15:
        score += 15
        signals_hit.append("high comment density")

    generic_names = re.findall(r"\b(result|value|data|item|temp|index|output|element)\b", code.lower())
    if len(generic_names) >= 3:
        score += 15
        signals_hit.append("generic/boilerplate variable naming")

    descriptive_names = re.findall(r"\b[a-z]+_[a-z]+[a-z_]*\b", code)
    if len(descriptive_names) >= 2:
        score += 15
        signals_hit.append("descriptive multi-word snake_case naming")

    try:
        ast.parse(code)
        score += 10
        signals_hit.append("syntactically clean / no leftover errors")
    except SyntaxError:
        pass

    blank_line_ratio = (len(code.splitlines()) - len(lines)) / max(len(code.splitlines()), 1)
    if 0.05 <= blank_line_ratio <= 0.25:
        score += 10
        signals_hit.append("consistent spacing/formatting")

    score = round(min(score, 100))
    if score >= 60:
        confidence = "Possibly AI-assisted style"
    elif score >= 30:
        confidence = "Mixed signals"
    else:
        confidence = "Likely hand-written / terse style"

    return {"score": score, "confidence": confidence, "signals": signals_hit, "disclaimer": _AI_DISCLAIMER}


_AI_DISCLAIMER = ("Heuristic style signal only -- based on formatting/naming conventions, "
                   "not a reliable AI-generation detector. Do not treat as proof.")


# ---------------------------------------------------------------------------
# Per-code statistics
# ---------------------------------------------------------------------------

def _code_statistics(code: str, features: CodeFeatures) -> dict:
    lines = code.splitlines()
    non_blank = [l for l in lines if l.strip()]
    comment_lines = sum(1 for l in non_blank if l.strip().startswith("#"))
    return {
        "line_count": len(lines),
        "non_blank_line_count": len(non_blank),
        "comment_line_count": comment_lines,
        "loop_count": features.loop_count,
        "max_loop_nesting": features.max_loop_nesting,
        "branch_count": features.branch_count,
        "assignment_count": features.assignment_count,
        "function_count": len(features.function_defs),
        "uses_recursion": features.has_recursion,
        "builtin_calls_used": sorted(features.builtin_calls),
        "parsed_successfully": features.parsed_ok,
    }


# ---------------------------------------------------------------------------
# Detected patterns (human-readable highlights)
# ---------------------------------------------------------------------------

def _detected_patterns(diff: DiffReport, approach_cmp: ApproachComparison, task: TaskAnalysis) -> list[str]:
    patterns: list[str] = []

    if approach_cmp.same_approach:
        patterns.append(f"Matching algorithmic pattern: {approach_cmp.approach_1.label}")
    else:
        patterns.append(f"Different algorithmic patterns: {approach_cmp.approach_1.label} vs "
                         f"{approach_cmp.approach_2.label}")

    if diff.changed_variables:
        pairs = ", ".join(f"{k}\u2192{v}" for k, v in list(diff.changed_variables.items())[:6])
        patterns.append(f"Identifier renaming detected: {pairs}"
                         f"{'...' if len(diff.changed_variables) > 6 else ''}")

    if diff.changed_operators:
        ops = ", ".join(f"{c.old_operator}\u2192{c.new_operator}" for c in diff.changed_operators[:5])
        patterns.append(f"Operator-only changes detected: {ops}")

    if diff.changed_statements:
        patterns.append(f"{len(diff.changed_statements)} statement-type change(s) detected "
                         f"(e.g. {diff.changed_statements[0].reason})")

    if diff.unchanged_ratio >= 0.95:
        patterns.append("Near-identical line sequence (formatting-level differences only)")

    if task.task_guess_1 == task.task_guess_2 and task.task_guess_1 != "Unknown":
        patterns.append(f"Inferred shared task category: {task.task_guess_1}")

    if approach_cmp.different_approaches_detected:
        patterns.append("Different approaches detected despite a likely shared objective")

    return patterns


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(code1: str, code2: str, problem_statement: Optional[str] = None) -> dict[str, Any]:
    """
    Runs the full Comparison + Approach Analysis + Task Analysis pipeline
    and returns the structured report dict consumed by Streamlit.
    """
    if code1 is None or code2 is None:
        raise ValueError("Both code1 and code2 must be provided (non-None) strings.")

    diff = compare_diff(code1, code2)
    approach_cmp = compare_approaches(code1, code2)
    task = analyze_task(code1, code2, problem_statement)

    t = text_similarity(code1, code2)
    a = ast_similarity(code1, code2)
    s = semantic_similarity(approach_cmp.approach_1.features, approach_cmp.approach_2.features)
    overall = round(t * 0.25 + a * 0.40 + s * 0.35, 2)

    ai1 = ai_indicator(code1)
    ai2 = ai_indicator(code2)

    report: dict[str, Any] = {
        "overall_similarity": round(overall),
        "text_similarity": round(t),
        "ast_similarity": round(a),
        "semantic_similarity": round(s),
        "ai_indicator_code1": ai1,
        "ai_indicator_code2": ai2,
        "same_objective": approach_cmp.same_objective,
        "same_approach": approach_cmp.same_approach,
        "detected_patterns": _detected_patterns(diff, approach_cmp, task),
        "code1_statistics": _code_statistics(code1, approach_cmp.approach_1.features),
        "code2_statistics": _code_statistics(code2, approach_cmp.approach_2.features),

        # supporting detail (not in the minimal schema, but useful to the UI)
        "similarity_classification": _classify(overall),
        "caution_note": ("This score reflects potentially similar implementation, not proof of "
                          "plagiarism. Common, well-known solutions naturally score higher -- "
                          "always review approach and task context before drawing conclusions."),
        "implementation_similarity": approach_cmp.implementation_similarity,
        "different_approaches_detected": approach_cmp.different_approaches_detected,
        "approach_detail": approach_cmp.to_dict(),
        "task_analysis": task.to_dict(),
        "diff": diff.to_dict(),
    }
    return report


# ---------------------------------------------------------------------------
# Demo / self-test suite -- run with `python report_generator.py`
# ---------------------------------------------------------------------------

def _demo_pairs() -> dict[str, tuple[str, str, Optional[str]]]:
    return {
        "Maximum: loop vs max()": (
            "def find_max(arr):\n    maximum = arr[0]\n    for x in arr:\n        if x > maximum:\n            maximum = x\n    return maximum",
            "def find_max(arr):\n    return max(arr)",
            "Find the largest element in an array.",
        ),
        "Sorting: Bubble vs Selection": (
            "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr",
            "def selection_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        min_idx = i\n        for j in range(i+1, n):\n            if arr[j] < arr[min_idx]:\n                min_idx = j\n        arr[i], arr[min_idx] = arr[min_idx], arr[i]\n    return arr",
            "Sort a list of numbers in ascending order.",
        ),
        "Search: Linear vs Binary": (
            "def linear_search(arr, target):\n    for i in range(len(arr)):\n        if arr[i] == target:\n            return i\n    return -1",
            "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1",
            "Search for a target value in a sorted array.",
        ),
        "Factorial: loop vs recursion": (
            "def factorial(n):\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result",
            "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
            "Compute the factorial of a number.",
        ),
        "Sum: loop vs sum()": (
            "def total(arr):\n    result = 0\n    for x in arr:\n        result += x\n    return result",
            "def total(arr):\n    return sum(arr)",
            "Find the sum of all elements in a list.",
        ),
        "Reverse: loop vs slicing": (
            "def reverse_list(arr):\n    left, right = 0, len(arr) - 1\n    while left < right:\n        arr[left], arr[right] = arr[right], arr[left]\n        left += 1\n        right -= 1\n    return arr",
            "def reverse_list(arr):\n    return arr[::-1]",
            "Reverse the order of elements in a list.",
        ),
        "Same task + same approach (near-identical)": (
            "def add(a, b):\n    return a + b",
            "def add(a, b):\n    return a + b",
            None,
        ),
        "Variable-renamed code": (
            "def compute(a, b):\n    total = a + b\n    return total",
            "def compute(x, y):\n    result = x + y\n    return result",
            None,
        ),
        "Reformatted code": (
            "def add(a,b):\n  return a+b",
            "def add(a, b):\n    return a + b",
            None,
        ),
        "Different tasks + similar syntax": (
            "def process(arr):\n    for x in arr:\n        if x > 0:\n            print(x)",
            "def process(arr):\n    for x in arr:\n        if x < 0:\n            print(x)",
            None,
        ),
        "Completely unrelated code": (
            "def greet(name):\n    return f'Hello, {name}!'",
            "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)",
            None,
        ),
    }


def _run_demo():
    print("=" * 78)
    print("CodeXray Backend -- Demo / Test-Pair Suite")
    print("=" * 78)
    for name, (c1, c2, problem) in _demo_pairs().items():
        report = generate_report(c1, c2, problem_statement=problem)
        print(f"\n--- {name} ---")
        print(f"  overall_similarity: {report['overall_similarity']}  "
              f"(text={report['text_similarity']}, ast={report['ast_similarity']}, "
              f"semantic={report['semantic_similarity']})")
        print(f"  classification: {report['similarity_classification']}")
        print(f"  same_objective={report['same_objective']}  same_approach={report['same_approach']}  "
              f"different_approaches_detected={report['different_approaches_detected']}")
        print(f"  approach_1={report['approach_detail']['approach_1']['label']}  "
              f"approach_2={report['approach_detail']['approach_2']['label']}")
        print(f"  task_guess_1={report['task_analysis']['task_guess_1']}  "
              f"task_guess_2={report['task_analysis']['task_guess_2']}")
        print(f"  ai_indicator_code1={report['ai_indicator_code1']['score']} "
              f"({report['ai_indicator_code1']['confidence']})  "
              f"ai_indicator_code2={report['ai_indicator_code2']['score']} "
              f"({report['ai_indicator_code2']['confidence']})")
        for p in report["detected_patterns"]:
            print(f"    * {p}")

    # Confirm the full report is JSON-serializable end to end (what Streamlit needs).
    sample = generate_report(*_demo_pairs()["Sorting: Bubble vs Selection"][:2])
    json.dumps(sample)
    print("\n" + "=" * 78)
    print("All demo pairs ran without errors. Full report is JSON-serializable.")
    print("=" * 78)


if __name__ == "__main__":
    _run_demo()
