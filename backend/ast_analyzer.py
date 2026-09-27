"""
ast_analyzer.py
================
Core AST (Abstract Syntax Tree) analysis module for CodeXray.

Responsibilities
-----------------
- Parse Python source into an AST
- Walk the tree and extract structured facts about the program
  (functions, loops, conditionals, assignments, operators, calls, returns,
  classes, nesting depth, etc.)
- Produce a *normalized* representation of the code where identifier
  names (variables, function params, etc.) are replaced with positional
  placeholders (VAR_1, VAR_2, FUNC_1, ...) so that two programs which are
  identical except for renamed identifiers can be recognized as
  structurally identical.
- Compute a structural (AST-based) similarity score between two code
  snippets.

Public API
----------
analyze_ast(code: str) -> dict
    Full structural report for a single snippet.

normalize_code(code: str) -> str
    Deterministic "normalized" pseudo-code string (renamed identifiers)
    used both for display and as an input to text-similarity diffing.

ast_similarity(code1: str, code2: str) -> float
    Similarity score in [0, 1] between two snippets, based on comparing
    normalized token sequences and structural node-type multisets.

Design notes
------------
This module intentionally has ZERO dependency on any ML library — it is
pure `ast` + stdlib, so it works even if torch/transformers are not
installed. That keeps CodeXray degrading gracefully: if CodeBERT is
unavailable, AST + text similarity can still carry the app.
"""

from __future__ import annotations

import ast
import difflib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


class CodeParseError(Exception):
    """Raised when the given source cannot be parsed as valid Python."""

    def __init__(self, message: str, lineno: int | None = None):
        super().__init__(message)
        self.lineno = lineno


# ---------------------------------------------------------------------------
# Low-level parsing
# ---------------------------------------------------------------------------

def parse_code(code: str) -> ast.AST:
    """Parse source into an AST, raising a CodeParseError with a friendly
    message (instead of a raw SyntaxError) on failure."""
    try:
        return ast.parse(code)
    except SyntaxError as exc:
        raise CodeParseError(f"Syntax error: {exc.msg}", lineno=exc.lineno) from exc


# ---------------------------------------------------------------------------
# Node extraction
# ---------------------------------------------------------------------------

# Node types we bucket explicitly; everything else falls into "other_nodes"
_INTERESTING_NODES = {
    ast.FunctionDef: "functions",
    ast.AsyncFunctionDef: "functions",
    ast.ClassDef: "classes",
    ast.For: "for_loops",
    ast.AsyncFor: "for_loops",
    ast.While: "while_loops",
    ast.If: "conditionals",
    ast.Assign: "assignments",
    ast.AugAssign: "aug_assignments",
    ast.AnnAssign: "assignments",
    ast.Return: "returns",
    ast.Call: "calls",
    ast.BinOp: "binary_ops",
    ast.BoolOp: "bool_ops",
    ast.Compare: "comparisons",
    ast.Break: "breaks",
    ast.Continue: "continues",
    ast.Try: "try_blocks",
    ast.With: "with_blocks",
    ast.Lambda: "lambdas",
    ast.ListComp: "comprehensions",
    ast.DictComp: "comprehensions",
    ast.SetComp: "comprehensions",
    ast.GeneratorExp: "comprehensions",
}


def _max_depth(node: ast.AST, depth: int = 0) -> int:
    """Deepest nesting level of compound statements (for/while/if/try/with/
    function/class) — a rough proxy for structural complexity."""
    compound = (
        ast.For, ast.AsyncFor, ast.While, ast.If, ast.Try, ast.With,
        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
    )
    best = depth
    for child in ast.iter_child_nodes(node):
        nxt = depth + 1 if isinstance(child, compound) else depth
        best = max(best, _max_depth(child, nxt))
    return best


def _detect_recursion(tree: ast.AST) -> list[str]:
    """Return names of functions that call themselves (direct recursion)."""
    recursive = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fname = node.name
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                    if inner.func.id == fname:
                        recursive.append(fname)
                        break
    return recursive


def _operator_name(op: ast.AST) -> str:
    return type(op).__name__


def extract_nodes(tree: ast.AST) -> dict[str, Any]:
    """Walk the tree once and bucket structural facts."""
    counts: Counter[str] = Counter()
    functions: list[dict[str, Any]] = []
    operators: Counter[str] = Counter()
    calls: Counter[str] = Counter()
    loop_count = 0
    nested_loops = 0

    for node in ast.walk(tree):
        for node_type, bucket in _INTERESTING_NODES.items():
            if isinstance(node, node_type):
                counts[bucket] += 1

        if isinstance(node, (ast.BinOp, ast.BoolOp, ast.UnaryOp)):
            operators[_operator_name(node.op)] += 1
        if isinstance(node, ast.Compare):
            for op in node.ops:
                operators[_operator_name(op)] += 1

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls[node.func.id] += 1
            elif isinstance(node.func, ast.Attribute):
                calls[node.func.attr] += 1

        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            loop_count += 1
            for inner in ast.walk(node):
                if inner is not node and isinstance(inner, (ast.For, ast.AsyncFor, ast.While)):
                    nested_loops += 1
                    break

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append({
                "name": node.name,
                "args": [a.arg for a in node.args.args],
                "num_args": len(node.args.args),
                "has_return": any(isinstance(n, ast.Return) and n.value is not None
                                   for n in ast.walk(node)),
                "lineno": node.lineno,
            })

    recursive_functions = _detect_recursion(tree)

    return {
        "counts": dict(counts),
        "functions": functions,
        "operators": dict(operators),
        "top_calls": calls.most_common(10),
        "loop_count": loop_count,
        "nested_loop_count": nested_loops,
        "recursive_functions": recursive_functions,
        "max_nesting_depth": _max_depth(tree),
    }


# ---------------------------------------------------------------------------
# Normalization (variable renaming resistance)
# ---------------------------------------------------------------------------

@dataclass
class _NameMapper:
    var_map: dict[str, str] = field(default_factory=dict)
    func_map: dict[str, str] = field(default_factory=dict)
    class_map: dict[str, str] = field(default_factory=dict)
    _var_n: int = 0
    _func_n: int = 0
    _class_n: int = 0

    def var(self, name: str) -> str:
        if name not in self.var_map:
            self._var_n += 1
            self.var_map[name] = f"VAR_{self._var_n}"
        return self.var_map[name]

    def func(self, name: str) -> str:
        if name in ("__init__", "main") or name.startswith("__"):
            return name
        if name not in self.func_map:
            self._func_n += 1
            self.func_map[name] = f"FUNC_{self._func_n}"
        return self.func_map[name]

    def cls(self, name: str) -> str:
        if name not in self.class_map:
            self._class_n += 1
            self.class_map[name] = f"CLASS_{self._class_n}"
        return self.class_map[name]


class _Renamer(ast.NodeTransformer):
    """Rewrites identifier names to positional placeholders in visitation
    order, so `a, b` and `x, y` normalize to the same VAR_1, VAR_2 stream
    regardless of the original spelling. Built-in names and dunders are
    left untouched."""

    _BUILTINS = set(dir(__builtins__)) if isinstance(__builtins__, dict) is False else set(__builtins__.keys())

    def __init__(self):
        self.mapper = _NameMapper()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node.name = self.mapper.func(node.name)
        for a in node.args.args:
            a.arg = self.mapper.var(a.arg)
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef):
        node.name = self.mapper.cls(node.name)
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name):
        if node.id in self._BUILTINS or node.id in ("self", "cls"):
            return node
        node.id = self.mapper.var(node.id)
        return node

    def visit_Attribute(self, node: ast.Attribute):
        # Keep attribute access names as-is (e.g. `.append`, `.sort`) —
        # renaming these would destroy useful semantic signal such as
        # "uses list built-ins" vs "hand-rolled loop".
        self.generic_visit(node)
        return node


def normalize_code(code: str) -> str:
    """Return a normalized pseudo-source string with identifiers replaced
    by positional placeholders. Falls back to returning stripped source
    unchanged if unparse fails (Python < 3.9 has no ast.unparse)."""
    tree = parse_code(code)
    renamer = _Renamer()
    normalized_tree = renamer.visit(tree)
    ast.fix_missing_locations(normalized_tree)
    try:
        return ast.unparse(normalized_tree)
    except AttributeError:
        # Extremely old Python without ast.unparse — degrade to a token
        # dump instead of failing outright.
        return " ".join(type(n).__name__ for n in ast.walk(normalized_tree))


# ---------------------------------------------------------------------------
# Logic-only token stream (used by logic_fingerprint.py too)
# ---------------------------------------------------------------------------

def structural_token_stream(tree: ast.AST) -> list[str]:
    """A flat sequence of node-type tokens in tree-walk order, ignoring all
    literal/identifier content entirely. This is the backbone that
    logic_fingerprint.py builds on."""
    return [type(node).__name__ for node in ast.walk(tree)]


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def _node_type_histogram(tree: ast.AST) -> Counter[str]:
    return Counter(type(node).__name__ for node in ast.walk(tree))


def _histogram_cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    norm_a = sum(v * v for v in a.values()) ** 0.5
    norm_b = sum(v * v for v in b.values()) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def ast_similarity(code1: str, code2: str) -> dict[str, Any]:
    """Structural similarity between two snippets.

    Combines two signals:
      1. Node-type histogram cosine similarity (bag-of-node-types — robust
         to reordering, catches "same ingredients, different order").
      2. Sequence similarity (difflib ratio) over the normalized token
         stream — sensitive to actual structural ordering, catches
         "same shape".

    Returns both sub-scores plus a blended `score` in [0, 1].
    """
    tree1 = parse_code(code1)
    tree2 = parse_code(code2)

    hist1, hist2 = _node_type_histogram(tree1), _node_type_histogram(tree2)
    histogram_score = _histogram_cosine(hist1, hist2)

    seq1 = structural_token_stream(tree1)
    seq2 = structural_token_stream(tree2)
    sequence_score = difflib.SequenceMatcher(a=seq1, b=seq2).ratio()

    # Weighted blend: sequence order matters slightly more than raw
    # ingredient counts, since two programs can share a node-type bag but
    # combine it very differently. Tune against evaluation/results.csv.
    blended = 0.45 * histogram_score + 0.55 * sequence_score

    return {
        "score": round(blended, 4),
        "histogram_similarity": round(histogram_score, 4),
        "sequence_similarity": round(sequence_score, 4),
    }


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------

def analyze_ast(code: str) -> dict[str, Any]:
    """Full structural report for a single code snippet. This is the
    function similarity_engine.py and ai_detector.py both call into."""
    tree = parse_code(code)
    facts = extract_nodes(tree)
    normalized = normalize_code(code)

    return {
        "valid": True,
        "node_facts": facts,
        "normalized_code": normalized,
        "token_stream_length": len(structural_token_stream(tree)),
    }


if __name__ == "__main__":
    sample_a = "a = 10\nb = 20\nprint(a + b)\n"
    sample_b = "x = 10\ny = 20\nprint(x + y)\n"
    print("normalize(a) ->", normalize_code(sample_a))
    print("normalize(b) ->", normalize_code(sample_b))
    print("similarity   ->", ast_similarity(sample_a, sample_b))
