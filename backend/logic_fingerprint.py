"""
logic_fingerprint.py
=====================
Builds a compact, human-readable "logic fingerprint" for a Python
program: a linear sequence of high-level control/logic tokens
(FUNCTION, FOR, IF, ASSIGNMENT, RETURN, ...) that captures *shape*
while ignoring names, literals, and formatting.

This sits one level above ast_analyzer.structural_token_stream(): that
stream includes every AST node type verbatim (Load, Store, arg, Module...)
which is too noisy to show a user. This module collapses that stream down
to a small, curated vocabulary and folds nested structure into a readable
arrow-chain, e.g.:

    FUNCTION -> ASSIGNMENT -> FOR -> IF -> ASSIGNMENT -> RETURN

Public API
----------
build_fingerprint(code: str) -> dict
    Returns the token list, the arrow-chain string, and per-function
    fingerprints.

fingerprint_similarity(code1: str, code2: str) -> dict
    Compares two fingerprints via sequence alignment (Levenshtein-style
    edit distance turned into a similarity ratio) plus n-gram overlap.

compare_fingerprints(fp1: list[str], fp2: list[str]) -> dict
    Lower-level comparator, useful for comparing two already-built
    fingerprints (e.g. one function fingerprint against another) without
    re-parsing source.
"""

from __future__ import annotations

import ast
import difflib
from typing import Any

from ast_analyzer import parse_code

# Curated vocabulary: AST node type -> fingerprint token.
# Anything not in this map is simply skipped (e.g. Load/Store/Name context
# nodes, which are structurally uninteresting for a logic-level view).
_TOKEN_MAP: dict[type, str] = {
    ast.FunctionDef: "FUNCTION",
    ast.AsyncFunctionDef: "FUNCTION",
    ast.ClassDef: "CLASS",
    ast.arguments: "PARAMETER",
    ast.For: "FOR",
    ast.AsyncFor: "FOR",
    ast.While: "WHILE",
    ast.If: "IF",
    ast.Assign: "ASSIGNMENT",
    ast.AugAssign: "ASSIGNMENT",
    ast.AnnAssign: "ASSIGNMENT",
    ast.Return: "RETURN",
    ast.Call: "FUNCTION_CALL",
    ast.Break: "BREAK",
    ast.Continue: "CONTINUE",
    ast.Try: "TRY",
    ast.ExceptHandler: "EXCEPT",
    ast.With: "WITH",
    ast.Lambda: "LAMBDA",
    ast.ListComp: "COMPREHENSION",
    ast.DictComp: "COMPREHENSION",
    ast.SetComp: "COMPREHENSION",
    ast.GeneratorExp: "COMPREHENSION",
    ast.Yield: "YIELD",
    ast.YieldFrom: "YIELD",
    ast.Raise: "RAISE",
    ast.Import: "IMPORT",
    ast.ImportFrom: "IMPORT",
}

# When walking with ast.walk (BFS-ish order that doesn't preserve strict
# source order for siblings deep in the tree), we instead do our own
# depth-first, source-ordered walk so the fingerprint reads top-to-bottom
# the way a human would read the file.


def _ordered_walk(node: ast.AST):
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _ordered_walk(child)


def _tokenize(tree: ast.AST, dedupe_consecutive_calls: bool = True) -> list[str]:
    tokens: list[str] = []
    for node in _ordered_walk(tree):
        for node_type, tok in _TOKEN_MAP.items():
            if isinstance(node, node_type):
                # PARAMETER only makes sense directly under a FUNCTION and
                # only if there actually are args — skip empty arg lists.
                if node_type is ast.arguments and not node.args:
                    continue
                tokens.append(tok)
                break

    if dedupe_consecutive_calls:
        # Collapse runs of consecutive FUNCTION_CALL tokens (e.g. chained
        # `.append()` calls inside one expression) into one, so the
        # fingerprint reflects logic shape rather than call-chain noise.
        collapsed: list[str] = []
        for tok in tokens:
            if tok == "FUNCTION_CALL" and collapsed and collapsed[-1] == "FUNCTION_CALL":
                continue
            collapsed.append(tok)
        tokens = collapsed

    return tokens


def _function_fingerprints(tree: ast.AST) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = _tokenize(node)
    return result


def build_fingerprint(code: str) -> dict[str, Any]:
    """Build the full logic fingerprint for a code snippet."""
    tree = parse_code(code)
    tokens = _tokenize(tree)
    return {
        "tokens": tokens,
        "chain": " -> ".join(tokens) if tokens else "(no logic detected)",
        "length": len(tokens),
        "per_function": {
            name: {"tokens": toks, "chain": " -> ".join(toks)}
            for name, toks in _function_fingerprints(tree).items()
        },
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _ngram_overlap(a: list[str], b: list[str], n: int = 2) -> float:
    ga, gb = _ngrams(a, n), _ngrams(b, n)
    if not ga or not gb:
        return 0.0
    set_a, set_b = set(ga), set(gb)
    return len(set_a & set_b) / len(set_a | set_b)


def compare_fingerprints(fp1: list[str], fp2: list[str]) -> dict[str, Any]:
    """Compare two already-tokenized fingerprints.

    - `sequence_ratio`: difflib ratio over the raw token sequence (order
      sensitive — this is effectively a normalized edit distance).
    - `bigram_overlap` / `trigram_overlap`: Jaccard overlap of n-grams,
      which tolerates small local reorderings better than a strict
      sequence match.
    """
    sequence_ratio = difflib.SequenceMatcher(a=fp1, b=fp2).ratio()
    bigram = _ngram_overlap(fp1, fp2, 2)
    trigram = _ngram_overlap(fp1, fp2, 3)

    blended = 0.5 * sequence_ratio + 0.3 * bigram + 0.2 * trigram

    return {
        "score": round(blended, 4),
        "sequence_ratio": round(sequence_ratio, 4),
        "bigram_overlap": round(bigram, 4),
        "trigram_overlap": round(trigram, 4),
    }


def fingerprint_similarity(code1: str, code2: str) -> dict[str, Any]:
    """Full pipeline: build fingerprints for two snippets and compare
    them, also returning the fingerprints themselves for display."""
    fp1 = build_fingerprint(code1)
    fp2 = build_fingerprint(code2)
    comparison = compare_fingerprints(fp1["tokens"], fp2["tokens"])
    return {
        "fingerprint_1": fp1,
        "fingerprint_2": fp2,
        **comparison,
    }


if __name__ == "__main__":
    sample_a = """
def maximum(arr):
    m = arr[0]
    for x in arr:
        if x > m:
            m = x
    return m
"""
    sample_b = """
def find_max(values):
    best = values[0]
    for v in values:
        if v > best:
            best = v
    return best
"""
    fp = build_fingerprint(sample_a)
    print("Fingerprint:", fp["chain"])
    print("Similarity:", fingerprint_similarity(sample_a, sample_b))
