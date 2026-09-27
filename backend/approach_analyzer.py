"""
approach_analyzer.py
=====================
Comparison layer, part 2 — the flagship CodeXray feature: telling two
snippets with the *same objective* apart from two snippets with the *same
implementation strategy*.

Example
-------
    Code A: for x in arr: if x > maximum: maximum = x
    Code B: maximum = max(arr)

    -> same_objective: "Likely"
       same_approach:   False
       implementation_similarity: "Low"
       different_approaches_detected: True

Recognized approach families (chosen to cover the CodeXray demo set):
    Sorting   : Bubble Sort, Selection Sort, Insertion Sort
    Searching : Linear Search, Binary Search
    Max/Min   : Iterative Max/Min, Built-in max()/min()
    Factorial : Recursive Factorial, Iterative Factorial
    Sum       : Iterative Sum, Built-in sum()
    Reverse   : Iterative/Two-Pointer Reverse, Slice Reverse ([::-1])
    Generic   : Recursive/Divide & Conquer, Iterative Manual Computation,
                Built-in/Library Shortcut, Simple Sequential Code

Self-contained: stdlib only (ast).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

_KNOWN_BUILTINS = {"max", "min", "sorted", "sum", "reversed", "filter", "map", "any", "all"}
_KNOWN_MODULE_FUNCS = {"bisect_left", "bisect_right", "bisect", "heappush", "heappop", "nlargest", "nsmallest"}


@dataclass
class CodeFeatures:
    loop_count: int = 0
    max_loop_nesting: int = 0
    has_recursion: bool = False
    has_swap_pattern: bool = False
    has_mult_accumulate: bool = False     # x *= n  /  x = x * n   (factorial-style)
    has_add_accumulate: bool = False      # x += n  /  x = x + n   (sum-style)
    uses_negative_slice: bool = False     # arr[::-1]
    comparison_ops: set[str] = field(default_factory=set)
    builtin_calls: set[str] = field(default_factory=set)
    module_calls: set[str] = field(default_factory=set)
    branch_count: int = 0
    assignment_count: int = 0
    function_defs: list[str] = field(default_factory=list)
    uses_while: bool = False
    uses_for: bool = False
    parsed_ok: bool = False

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["comparison_ops"] = sorted(self.comparison_ops)
        d["builtin_calls"] = sorted(self.builtin_calls)
        d["module_calls"] = sorted(self.module_calls)
        return d


def _dump_no_ctx(node: ast.AST) -> str:
    """ast.dump() bakes in Load()/Store()/Del() context, which always
    differs between an assignment's target side and its value side even
    for the exact same expression -- so raw dump() can never match target
    vs. value. Strip that noise before comparing."""
    return re.sub(r", ctx=(Load|Store|Del)\(\)", "", ast.dump(node))


def _is_tuple_swap(target: ast.expr, value: ast.expr) -> bool:
    """True only for a genuine swap: `A, B = B, A` (names or subscripts,
    e.g. `arr[i], arr[j] = arr[j], arr[i]`) -- NOT plain tuple unpacking
    like `low, high = 0, len(arr) - 1`, which has unrelated left/right
    expressions and must not be mistaken for a swap idiom."""
    if not (isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple)):
        return False
    if len(target.elts) != 2 or len(value.elts) != 2:
        return False
    t0, t1 = _dump_no_ctx(target.elts[0]), _dump_no_ctx(target.elts[1])
    v0, v1 = _dump_no_ctx(value.elts[0]), _dump_no_ctx(value.elts[1])
    return t0 == v1 and t1 == v0 and t0 != v0


class _FeatureVisitor(ast.NodeVisitor):
    def __init__(self):
        self.f = CodeFeatures()
        self._loop_depth = 0
        self._func_name_hint: Optional[str] = None

    def _enter_loop(self, node):
        self.f.loop_count += 1
        self._loop_depth += 1
        self.f.max_loop_nesting = max(self.f.max_loop_nesting, self._loop_depth)
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_For(self, node):
        self.f.uses_for = True
        self._enter_loop(node)

    def visit_While(self, node):
        self.f.uses_while = True
        self._enter_loop(node)

    def visit_If(self, node):
        self.f.branch_count += 1
        self.generic_visit(node)

    def visit_Call(self, node):
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
        if name in _KNOWN_BUILTINS:
            self.f.builtin_calls.add(name)
        if name in _KNOWN_MODULE_FUNCS:
            self.f.module_calls.add(name)
        if self._func_name_hint and name == self._func_name_hint:
            self.f.has_recursion = True
        self.generic_visit(node)

    def visit_Compare(self, node):
        for op in node.ops:
            self.f.comparison_ops.add(type(op).__name__)
        self.generic_visit(node)

    def visit_BinOp(self, node):
        # Recursive accumulate patterns written as a bare expression rather
        # than an assignment, e.g. `return n * factorial(n - 1)` (factorial)
        # or `return n + total(rest)` (recursive sum) -- these never show up
        # as an Assign/AugAssign, so they need their own check here.
        if self._func_name_hint:
            sides = (node.left, node.right)
            calls_self = any(
                isinstance(s, ast.Call) and (
                    (isinstance(s.func, ast.Name) and s.func.id == self._func_name_hint) or
                    (isinstance(s.func, ast.Attribute) and s.func.attr == self._func_name_hint)
                ) for s in sides
            )
            if calls_self:
                if isinstance(node.op, ast.Mult):
                    self.f.has_mult_accumulate = True
                elif isinstance(node.op, ast.Add):
                    self.f.has_add_accumulate = True
        self.generic_visit(node)

    def visit_Subscript(self, node):
        sl = node.slice
        # arr[::-1]  ->  Slice(step=UnaryOp(USub, Constant(1)))
        if isinstance(sl, ast.Slice) and sl.step is not None:
            step = sl.step
            if isinstance(step, ast.UnaryOp) and isinstance(step.op, ast.USub):
                self.f.uses_negative_slice = True
            elif isinstance(step, ast.Constant) and isinstance(step.value, int) and step.value < 0:
                self.f.uses_negative_slice = True
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self.f.assignment_count += 1
        # `i += 1` / `right -= 1` is a loop counter/index step, not a sum
        # accumulator -- only unit-step Add is excluded (Mult by +/-1 is
        # not a meaningful "counter" pattern, so it's left untouched).
        is_unit_step = (isinstance(node.value, ast.Constant)
                         and isinstance(node.value.value, (int, float))
                         and abs(node.value.value) == 1)
        if isinstance(node.op, ast.Mult):
            self.f.has_mult_accumulate = True
        elif isinstance(node.op, ast.Add) and not is_unit_step:
            self.f.has_add_accumulate = True
        self.generic_visit(node)

    def visit_Assign(self, node):
        self.f.assignment_count += 1
        if _is_tuple_swap(node.targets[0], node.value):
            self.f.has_swap_pattern = True
        # x = x * n  /  x = x + n  (accumulate written without augassign)
        if (isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.BinOp)
                and isinstance(node.value.left, ast.Name) and node.value.left.id == node.targets[0].id):
            is_unit_step = (isinstance(node.value.right, ast.Constant)
                             and isinstance(node.value.right.value, (int, float))
                             and abs(node.value.right.value) == 1)
            if isinstance(node.value.op, ast.Mult):
                self.f.has_mult_accumulate = True
            elif isinstance(node.value.op, ast.Add) and not is_unit_step:
                self.f.has_add_accumulate = True
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.f.function_defs.append(node.name)
        old = self._func_name_hint
        self._func_name_hint = node.name
        self.generic_visit(node)
        self._func_name_hint = old


def extract_features(source: str) -> CodeFeatures:
    """Never raises: returns an empty (parsed_ok=False) feature set for
    unparsable input so the rest of the pipeline degrades gracefully."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return CodeFeatures()

    v = _FeatureVisitor()
    v.visit(tree)
    v.f.parsed_ok = True

    if not v.f.has_swap_pattern:
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                names = [t.id for n in ast.walk(node) if isinstance(n, ast.Assign)
                         for t in n.targets if isinstance(t, ast.Name)]
                if any(n.lower() in ("temp", "tmp") for n in names) and len(names) >= 3:
                    v.f.has_swap_pattern = True
                    break

    return v.f


# ---------------------------------------------------------------------------
# Signature library (gated scorers: 0.0 if a defining feature is absent)
# ---------------------------------------------------------------------------

def _bubble_sort(f: CodeFeatures) -> float:
    if f.max_loop_nesting < 2 or not f.has_swap_pattern:
        return 0.0
    s = 0.55
    s += 0.15 if ("Gt" in f.comparison_ops or "Lt" in f.comparison_ops) else 0.0
    s += 0.1 if not f.has_recursion and not f.builtin_calls else 0.0
    s += 0.2 if f.assignment_count <= 3 else -0.25
    return max(0.0, min(s, 1.0))


def _selection_sort(f: CodeFeatures) -> float:
    if f.max_loop_nesting < 2 or not f.has_swap_pattern:
        return 0.0
    s = 0.35
    s += 0.15 if f.branch_count >= 1 else 0.0
    s += 0.5 if f.assignment_count >= 4 else -0.2
    return max(0.0, min(s, 1.0))


def _insertion_sort(f: CodeFeatures) -> float:
    if not (f.uses_while and f.uses_for) or f.max_loop_nesting < 2:
        return 0.0
    s = 0.4
    s += 0.2 if ("Gt" in f.comparison_ops or "Lt" in f.comparison_ops) else 0.0
    s += 0.2 if f.assignment_count >= 2 else 0.0
    s += 0.2 if not f.has_swap_pattern else 0.0
    return max(0.0, min(s, 1.0))


def _binary_search(f: CodeFeatures) -> float:
    if not (f.uses_while and f.max_loop_nesting == 1):
        return 0.0
    s = 0.2
    s += 0.3 if ("Eq" in f.comparison_ops and ("Gt" in f.comparison_ops or "Lt" in f.comparison_ops)) else 0.0
    s += 0.4 if f.assignment_count >= 3 else 0.0
    return max(0.0, min(s, 1.0))


def _linear_search(f: CodeFeatures) -> float:
    if not (f.uses_for and f.max_loop_nesting == 1) or f.has_recursion or f.builtin_calls:
        return 0.0
    s = 0.5
    s += 0.4 if f.branch_count >= 1 else 0.0
    return max(0.0, min(s, 1.0))


def _recursive_factorial(f: CodeFeatures) -> float:
    if not f.has_recursion:
        return 0.0
    s = 0.5
    s += 0.3 if f.has_mult_accumulate else 0.0
    s += 0.2 if f.loop_count == 0 else 0.0
    return max(0.0, min(s, 1.0))


def _iterative_factorial(f: CodeFeatures) -> float:
    if f.has_recursion or not f.has_mult_accumulate:
        return 0.0
    s = 0.5
    s += 0.3 if f.max_loop_nesting == 1 else 0.0
    s += 0.2 if not f.builtin_calls else 0.0
    return max(0.0, min(s, 1.0))


def _iterative_sum(f: CodeFeatures) -> float:
    if f.has_recursion or "sum" in f.builtin_calls or not f.has_add_accumulate:
        return 0.0
    s = 0.5
    s += 0.3 if f.max_loop_nesting == 1 else 0.0
    s += 0.2 if not f.builtin_calls else 0.0
    return max(0.0, min(s, 1.0))


def _reverse_slicing(f: CodeFeatures) -> float:
    if not f.uses_negative_slice:
        return 0.0
    s = 0.7
    s += 0.3 if f.loop_count == 0 else 0.0
    return max(0.0, min(s, 1.0))


def _reverse_loop(f: CodeFeatures) -> float:
    if f.uses_negative_slice or f.loop_count == 0:
        return 0.0
    if "reversed" in f.builtin_calls:
        return 0.0
    s = 0.4
    s += 0.35 if f.has_swap_pattern else 0.0
    s += 0.25 if f.max_loop_nesting == 1 else 0.0
    return max(0.0, min(s, 1.0))


def _builtin_shortcut(f: CodeFeatures) -> float:
    if not (f.builtin_calls or f.module_calls):
        return 0.0
    s = 0.6
    s += 0.3 if f.loop_count == 0 else 0.0
    s += 0.1 if not f.has_recursion else 0.0
    return max(0.0, min(s, 1.0))


def _recursive_divide_conquer(f: CodeFeatures) -> float:
    if not f.has_recursion or f.has_mult_accumulate:
        return 0.0
    s = 0.6
    s += 0.2 if f.branch_count >= 1 else 0.0
    s += 0.2 if f.loop_count == 0 else 0.0
    return max(0.0, min(s, 1.0))


def _iterative_manual(f: CodeFeatures) -> float:
    if f.max_loop_nesting != 1 or f.has_recursion or f.builtin_calls:
        return 0.0
    if f.has_mult_accumulate or f.has_add_accumulate or f.has_swap_pattern:
        return 0.0
    s = 0.35
    s += 0.35 if f.branch_count >= 1 else 0.0
    s += 0.15 if f.assignment_count >= 1 else 0.0
    return max(0.0, min(s, 1.0))


def _simple_sequential(f: CodeFeatures) -> float:
    if f.loop_count != 0 or f.branch_count != 0 or f.has_recursion:
        return 0.0
    if f.builtin_calls or f.module_calls:
        return 0.0
    s = 0.5
    s += 0.3 if f.assignment_count >= 1 else 0.0
    s += 0.1 if not f.function_defs else 0.0
    return max(0.0, min(s, 1.0))


_SIGNATURES = {
    "Bubble Sort": _bubble_sort,
    "Selection Sort": _selection_sort,
    "Insertion Sort": _insertion_sort,
    "Binary Search": _binary_search,
    "Linear Search": _linear_search,
    "Recursive Factorial": _recursive_factorial,
    "Iterative Factorial": _iterative_factorial,
    "Iterative Sum": _iterative_sum,
    "Reverse via Slicing": _reverse_slicing,
    "Iterative/Two-Pointer Reverse": _reverse_loop,
    "Built-in / Library Shortcut": _builtin_shortcut,
    "Recursive / Divide & Conquer": _recursive_divide_conquer,
    "Iterative Manual Computation": _iterative_manual,
    "Simple Sequential Code": _simple_sequential,
}

_EXPLANATIONS = {
    "Bubble Sort": "Nested loops with an adjacent-element swap and no extra index bookkeeping.",
    "Selection Sort": "Nested loops tracking a running min/max index before one swap per pass.",
    "Insertion Sort": "Combined for/while loops shifting elements into position (no swap-tuple idiom).",
    "Binary Search": "Single while-loop narrowing a low/high/mid range with equality + ordering checks.",
    "Linear Search": "Single for-loop with a conditional check, no recursion or shortcut call.",
    "Recursive Factorial": "Function calls itself and multiplies an accumulator, no explicit loop.",
    "Iterative Factorial": "Single loop multiplying an accumulator (x *= n / x = x * n).",
    "Iterative Sum": "Single loop adding into an accumulator (x += n / x = x + n), sum() not used.",
    "Reverse via Slicing": "Uses a negative-step slice (e.g. arr[::-1]) instead of manual iteration.",
    "Iterative/Two-Pointer Reverse": "Manual loop with element swapping, no slicing shortcut.",
    "Built-in / Library Shortcut": "Relies on a built-in/library call (max, sorted, sum, bisect...) instead of manual iteration.",
    "Recursive / Divide & Conquer": "Function calls itself with a branching base case, no explicit loop.",
    "Iterative Manual Computation": "Hand-written single loop with conditional updates, no accumulator/builtin shortcut.",
    "Simple Sequential Code": "Flat sequence of statements with no loops, branches, recursion, or calls.",
}


@dataclass
class ApproachMatch:
    label: str
    confidence: float
    explanation: str
    features: CodeFeatures


def detect_approach(source: str) -> ApproachMatch:
    f = extract_features(source)
    if not f.parsed_ok:
        return ApproachMatch("Unrecognized / Unparsable", 0.0,
                              "Could not parse the snippet into a valid syntax tree.", f)

    scored = {label: fn(f) for label, fn in _SIGNATURES.items()}
    best_label = max(scored, key=scored.get)
    best_score = scored[best_label]

    if best_score < 0.35:
        return ApproachMatch("Generic / Unclassified Approach", round(best_score, 3),
                              "No strong match to a known algorithmic signature.", f)
    return ApproachMatch(best_label, round(best_score, 3), _EXPLANATIONS[best_label], f)


# ---------------------------------------------------------------------------
# Cross-snippet comparison
# ---------------------------------------------------------------------------

@dataclass
class ApproachComparison:
    approach_1: ApproachMatch
    approach_2: ApproachMatch
    same_objective: str            # "Likely" | "Uncertain" | "Unlikely"
    same_approach: bool
    implementation_similarity: str  # Very Low..Very High
    implementation_similarity_score: float
    different_approaches_detected: bool
    notes: str

    def to_dict(self) -> dict:
        return {
            "approach_1": {"label": self.approach_1.label, "confidence": self.approach_1.confidence,
                            "explanation": self.approach_1.explanation},
            "approach_2": {"label": self.approach_2.label, "confidence": self.approach_2.confidence,
                            "explanation": self.approach_2.explanation},
            "same_objective": self.same_objective,
            "same_approach": self.same_approach,
            "implementation_similarity": self.implementation_similarity,
            "implementation_similarity_score": round(self.implementation_similarity_score, 4),
            "different_approaches_detected": self.different_approaches_detected,
            "notes": self.notes,
        }


def _feature_distance_score(f1: CodeFeatures, f2: CodeFeatures) -> float:
    if not f1.parsed_ok or not f2.parsed_ok:
        return 0.0

    def norm(x, cap):
        return min(x, cap) / cap

    dims = [
        1.0 - abs(norm(f1.loop_count, 5) - norm(f2.loop_count, 5)),
        1.0 - abs(norm(f1.max_loop_nesting, 3) - norm(f2.max_loop_nesting, 3)),
        1.0 if f1.has_recursion == f2.has_recursion else 0.0,
        1.0 if f1.has_swap_pattern == f2.has_swap_pattern else 0.0,
        1.0 if f1.has_mult_accumulate == f2.has_mult_accumulate else 0.0,
        1.0 if f1.has_add_accumulate == f2.has_add_accumulate else 0.0,
        1.0 if f1.uses_negative_slice == f2.uses_negative_slice else 0.0,
        1.0 if bool(f1.builtin_calls) == bool(f2.builtin_calls) else 0.0,
        1.0 - abs(norm(f1.branch_count, 5) - norm(f2.branch_count, 5)),
        (len(f1.comparison_ops & f2.comparison_ops) / max(len(f1.comparison_ops | f2.comparison_ops), 1)
         if (f1.comparison_ops or f2.comparison_ops) else 1.0),
    ]
    return sum(dims) / len(dims)


def _label_similarity(score: float) -> str:
    if score >= 0.85:
        return "Very High"
    if score >= 0.65:
        return "High"
    if score >= 0.45:
        return "Moderate"
    if score >= 0.25:
        return "Low"
    return "Very Low"


def compare_approaches(code1: str, code2: str) -> ApproachComparison:
    """
    Feature 2 — approach detection + different-approach flagging.

    CodeXray never claims certainty about intent from code alone; results
    are framed as "Likely"/"Uncertain"/"Unlikely" rather than absolute.
    """
    m1, m2 = detect_approach(code1), detect_approach(code2)
    score = _feature_distance_score(m1.features, m2.features)
    label = _label_similarity(score)

    both_recognized = m1.label not in ("Unrecognized / Unparsable",) and m2.label not in ("Unrecognized / Unparsable",)
    same_approach = both_recognized and m1.label == m2.label and m1.label != "Generic / Unclassified Approach"

    if not both_recognized:
        same_objective = "Uncertain"
    elif same_approach or score >= 0.55:
        same_objective = "Likely"
    elif score >= 0.3:
        same_objective = "Uncertain"
    else:
        same_objective = "Unlikely"

    different_approaches_detected = both_recognized and not same_approach and m1.label != m2.label

    if same_approach:
        notes = (f"Both snippets match the same recognized approach ('{m1.label}') with comparable "
                 "structure. Worth a closer look, but pattern similarity alone is not proof of copying.")
    elif different_approaches_detected:
        notes = (f"Different implementation strategies detected ('{m1.label}' vs '{m2.label}') "
                 f"despite a {'likely' if same_objective == 'Likely' else 'possibly'} shared objective "
                 "-- consistent with independent problem-solving.")
    else:
        notes = "Limited structural overlap; likely different tasks or approaches."

    return ApproachComparison(
        approach_1=m1, approach_2=m2,
        same_objective=same_objective, same_approach=same_approach,
        implementation_similarity=label, implementation_similarity_score=score,
        different_approaches_detected=different_approaches_detected, notes=notes,
    )
