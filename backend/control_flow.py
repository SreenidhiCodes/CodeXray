"""
control_flow.py
=================
Deep control-flow analysis for Python source.

Where ast_analyzer.py gives raw node counts and logic_fingerprint.py gives
a linear "shape" of the program, this module builds an actual per-function
control-flow tree/graph: branch structure, loop nesting, early exits,
recursion, and reachability-adjacent facts (e.g. "this branch has no
return on some paths").

Public API
----------
analyze_control_flow(code: str) -> dict
    Module-level report: control_flow tree per function + aggregate stats.

branch_complexity(code: str) -> int
    Cyclomatic-complexity-style branch count (McCabe-inspired), summed
    across the module. Useful as one input to ai_detector.py (very low
    or very uniform complexity can be a weak AI-generated-code signal).

control_flow_similarity(code1: str, code2: str) -> dict
    Compares two snippets' control-flow trees.
"""

from __future__ import annotations

import ast
from typing import Any

from ast_analyzer import parse_code


# ---------------------------------------------------------------------------
# Per-function control-flow tree
# ---------------------------------------------------------------------------

def _branch_node(node: ast.AST) -> dict[str, Any] | None:
    """Recursively describe a single control-flow-relevant statement as a
    small tree node. Returns None for statements we don't track."""
    if isinstance(node, (ast.If,)):
        return {
            "type": "if",
            "has_elif": any(isinstance(n, ast.If) for n in node.orelse),
            "has_else": bool(node.orelse) and not any(isinstance(n, ast.If) for n in node.orelse),
            "body": _walk_body(node.body),
            "orelse": _walk_body(node.orelse),
        }
    if isinstance(node, (ast.For, ast.AsyncFor)):
        return {
            "type": "for",
            "has_else": bool(node.orelse),
            "body": _walk_body(node.body),
        }
    if isinstance(node, ast.While):
        return {
            "type": "while",
            "has_else": bool(node.orelse),
            "body": _walk_body(node.body),
        }
    if isinstance(node, ast.Try):
        return {
            "type": "try",
            "num_except": len(node.handlers),
            "has_finally": bool(node.finalbody),
            "has_else": bool(node.orelse),
            "body": _walk_body(node.body),
        }
    if isinstance(node, ast.With):
        return {"type": "with", "body": _walk_body(node.body)}
    if isinstance(node, ast.Break):
        return {"type": "break"}
    if isinstance(node, ast.Continue):
        return {"type": "continue"}
    if isinstance(node, ast.Return):
        return {"type": "return", "has_value": node.value is not None}
    return None


def _walk_body(stmts: list[ast.stmt]) -> list[dict[str, Any]]:
    out = []
    for s in stmts:
        node = _branch_node(s)
        if node is not None:
            out.append(node)
        # Statements containing nested compound structures we don't
        # explicitly branch on (e.g. a plain expression statement) are
        # skipped — only control-flow-relevant nodes populate the tree.
    return out


def _max_branch_depth(nodes: list[dict[str, Any]], depth: int = 0) -> int:
    if not nodes:
        return depth
    best = depth
    for n in nodes:
        for key in ("body", "orelse"):
            if key in n:
                best = max(best, _max_branch_depth(n[key], depth + 1))
    return best


def _count_all_returns(body: list[ast.stmt]) -> list[bool]:
    """Return list of `has_value` flags for every Return in a function,
    in source order (used to flag inconsistent return behavior)."""
    flags = []
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Return):
            flags.append(node.value is not None)
    return flags


def _function_cfg(func: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    tree_body = _walk_body(func.body)
    returns = _count_all_returns(func.body)

    # Direct recursion check
    is_recursive = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == func.name
        for n in ast.walk(func)
    )

    # Every code path guaranteed to hit a `return`? A conservative,
    # syntax-level approximation (not full reachability analysis): true
    # only if the function body's last statement is a return, or every
    # branch of a trailing if/else independently returns.
    def _always_returns(stmts: list[ast.stmt]) -> bool:
        if not stmts:
            return False
        last = stmts[-1]
        if isinstance(last, ast.Return):
            return True
        if isinstance(last, ast.If) and last.orelse:
            return _always_returns(last.body) and _always_returns(last.orelse)
        return False

    return {
        "name": func.name,
        "branch_tree": tree_body,
        "max_nesting_depth": _max_branch_depth(tree_body),
        "num_returns": len(returns),
        "inconsistent_returns": len(set(returns)) > 1,  # mixes `return x` and bare `return`
        "always_returns_on_all_paths": _always_returns(func.body),
        "is_recursive": is_recursive,
        "num_branches": sum(1 for n in ast.walk(func) if isinstance(n, (ast.If, ast.For, ast.While))),
    }


# ---------------------------------------------------------------------------
# Cyclomatic-style complexity
# ---------------------------------------------------------------------------

_COMPLEXITY_NODES = (
    ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.ExceptHandler,
    ast.BoolOp, ast.With,
)


def branch_complexity(code: str) -> int:
    """McCabe-inspired cyclomatic complexity: 1 + number of decision
    points in the whole module. Simplified (doesn't special-case boolean
    short-circuit chains beyond counting BoolOp nodes once each)."""
    tree = parse_code(code)
    decisions = sum(1 for n in ast.walk(tree) if isinstance(n, _COMPLEXITY_NODES))
    return 1 + decisions


# ---------------------------------------------------------------------------
# Module-level report
# ---------------------------------------------------------------------------

def analyze_control_flow(code: str) -> dict[str, Any]:
    tree = parse_code(code)
    functions = [
        _function_cfg(n) for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    module_level_branches = _walk_body(tree.body)

    return {
        "functions": functions,
        "module_level_branch_tree": module_level_branches,
        "cyclomatic_complexity": branch_complexity(code),
        "recursive_functions": [f["name"] for f in functions if f["is_recursive"]],
        "max_nesting_depth": max(
            [f["max_nesting_depth"] for f in functions] + [_max_branch_depth(module_level_branches)]
            or [0]
        ),
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _shape_signature(nodes: list[dict[str, Any]]) -> tuple:
    """Recursively reduce a branch tree to a hashable shape signature
    (type + child shapes), ignoring everything but structure."""
    sig = []
    for n in nodes:
        children = []
        for key in ("body", "orelse"):
            if key in n:
                children.append(_shape_signature(n[key]))
        sig.append((n["type"], tuple(children)))
    return tuple(sig)


def control_flow_similarity(code1: str, code2: str) -> dict[str, Any]:
    """Compare two snippets' control-flow shape. Uses a tree-edit-style
    approximation: compares flattened shape signatures per function with
    difflib, and compares aggregate stats (complexity, nesting, recursion)
    directly."""
    import difflib

    cf1 = analyze_control_flow(code1)
    cf2 = analyze_control_flow(code2)

    sig1 = [str(_shape_signature(f["branch_tree"])) for f in cf1["functions"]]
    sig2 = [str(_shape_signature(f["branch_tree"])) for f in cf2["functions"]]
    # Compare concatenated signatures as a fallback if function counts differ
    flat1 = "|".join(sig1) or str(_shape_signature(cf1["module_level_branch_tree"]))
    flat2 = "|".join(sig2) or str(_shape_signature(cf2["module_level_branch_tree"]))

    shape_ratio = difflib.SequenceMatcher(a=flat1, b=flat2).ratio()

    complexity_diff = abs(cf1["cyclomatic_complexity"] - cf2["cyclomatic_complexity"])
    complexity_similarity = 1.0 / (1.0 + complexity_diff)  # 1.0 when equal, decays smoothly

    recursion_match = bool(cf1["recursive_functions"]) == bool(cf2["recursive_functions"])

    blended = 0.6 * shape_ratio + 0.3 * complexity_similarity + 0.1 * (1.0 if recursion_match else 0.0)

    return {
        "score": round(blended, 4),
        "shape_similarity": round(shape_ratio, 4),
        "complexity_similarity": round(complexity_similarity, 4),
        "recursion_match": recursion_match,
        "code1_complexity": cf1["cyclomatic_complexity"],
        "code2_complexity": cf2["cyclomatic_complexity"],
    }


if __name__ == "__main__":
    sample = """
def classify(n):
    if n < 0:
        return "negative"
    elif n == 0:
        return "zero"
    else:
        return "positive"
"""
    report = analyze_control_flow(sample)
    print("Cyclomatic complexity:", report["cyclomatic_complexity"])
    print("Functions:", [f["name"] for f in report["functions"]])
    print("Always returns on all paths:", report["functions"][0]["always_returns_on_all_paths"])
