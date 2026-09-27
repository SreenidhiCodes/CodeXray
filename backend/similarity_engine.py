"""
similarity_engine.py
======================
The core "brain" of CodeXray: combines every similarity signal produced
elsewhere in the backend into one hybrid, explainable similarity report.

    Text Similarity
          +
    AST Similarity
          +
    Logic-Fingerprint Similarity
          +
    Control-Flow Similarity
          +
    Semantic Similarity (CodeBERT, optional)
          v
    Hybrid / Overall Similarity + Evidence

Public API
----------
compare_codes(code1: str, code2: str, weights: dict | None = None) -> dict
    The main entry point. Returns the full comparison payload the
    Streamlit UI renders directly: per-signal scores, an overall blended
    score, human-readable "detected patterns" evidence, and an
    approach-analysis summary (same objective, different implementation,
    etc).

DEFAULT_WEIGHTS
    The starting weights for the blend. These are NOT presented as
    universally correct — see `evaluation/` for how they should be
    tuned against the labeled dataset (exact_copy / variable_renamed /
    reformatted / restructured / different_algorithm / unrelated).

Dependency notes
-----------------
This module owns the *combination* logic only. It imports:
  - ast_analyzer.ast_similarity            (this team's own module)
  - logic_fingerprint.fingerprint_similarity (this team's own module)
  - control_flow.control_flow_similarity    (this team's own module)
  - codebert_analyzer.semantic_similarity   (this team's own module,
    optional at runtime — degrades gracefully if torch/transformers
    aren't installed)
  - backend.text_similarity.text_similarity  (owned by a teammate —
    plain-text / token-level similarity, e.g. difflib or TF-IDF over raw
    source). If that module isn't present yet, this file falls back to a
    local difflib-based text similarity so the engine still runs
    end-to-end during development.
"""

from __future__ import annotations

from typing import Any

from ast_analyzer import ast_similarity, normalize_code, parse_code, CodeParseError
from logic_fingerprint import fingerprint_similarity
from control_flow import control_flow_similarity
import codebert_analyzer

try:
    # Owned by a teammate. Expected signature: text_similarity(a, b) -> float in [0, 1]
    from text_similarity import text_similarity as _external_text_similarity
except ImportError:
    _external_text_similarity = None


def _fallback_text_similarity(code1: str, code2: str) -> float:
    """Local fallback used only if backend/text_similarity.py doesn't
    exist yet (development convenience). A real implementation should
    live in text_similarity.py, e.g. normalized token overlap or TF-IDF
    cosine similarity."""
    import difflib
    return difflib.SequenceMatcher(a=code1, b=code2).ratio()


def _text_similarity(code1: str, code2: str) -> float:
    fn = _external_text_similarity or _fallback_text_similarity
    return float(fn(code1, code2))


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

# NOTE: these are starting points, not final values. `evaluation/evaluation.py`
# should sweep weight combinations against the labeled dataset folders and
# report the combination that maximizes F1 / correlation with ground truth,
# then this default should be updated to match.
DEFAULT_WEIGHTS = {
    "text": 0.15,
    "ast": 0.30,
    "fingerprint": 0.15,
    "control_flow": 0.10,
    "semantic": 0.30,
}

# Used automatically when CodeBERT is unavailable, so the four remaining
# signals still sum to 1.0 instead of silently underweighting the result.
_WEIGHTS_WITHOUT_SEMANTIC = {
    "text": 0.25,
    "ast": 0.40,
    "fingerprint": 0.20,
    "control_flow": 0.15,
}


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total == 0:
        raise ValueError("Weights must not sum to zero.")
    return {k: v / total for k, v in weights.items()}


# ---------------------------------------------------------------------------
# Evidence / explanation generation
# ---------------------------------------------------------------------------

def _generate_evidence(
    code1: str, code2: str,
    ast_result: dict, fp_result: dict, cf_result: dict,
    text_score: float, semantic_result: dict | None,
) -> list[str]:
    """Turn raw sub-scores into short, human-readable bullet points —
    this is what powers the "DETECTED PATTERNS" panel in the UI."""
    evidence: list[str] = []

    exact_copy = code1.strip() == code2.strip()
    norm1, norm2 = normalize_code(code1), normalize_code(code2)
    if exact_copy:
        pass  # covered by the "Exact copy" bullet below instead
    elif norm1 == norm2:
        evidence.append("Variable renaming detected (identical after normalization)")
    elif ast_result["histogram_similarity"] > 0.9:
        evidence.append("Variable names changed, structure preserved")

    if ast_result["sequence_similarity"] > 0.85:
        evidence.append("Similar loop structure detected")
    if cf_result["shape_similarity"] > 0.85:
        evidence.append("Similar conditional structure detected")
    if fp_result["score"] > 0.85:
        evidence.append("Similar function structure (logic fingerprint match)")

    from control_flow import analyze_control_flow
    both_recursive = (
        bool(analyze_control_flow(code1)["recursive_functions"])
        and bool(analyze_control_flow(code2)["recursive_functions"])
    )
    if both_recursive:
        evidence.append("Both snippets use recursion")
    if semantic_result and semantic_result["score"] > 0.85:
        evidence.append("High semantic similarity (CodeBERT)")
    if text_score < 0.4 and ast_result["score"] > 0.8:
        evidence.append("Low text overlap but high structural similarity (likely paraphrased/renamed copy)")
    if code1.strip() == code2.strip():
        evidence.append("Exact copy (byte-for-byte identical)")

    if not evidence:
        evidence.append("No strong similarity patterns detected")

    return evidence


def _approach_label(code: str) -> str:
    """Very light heuristic classification of *how* a snippet solves its
    problem, used for the "APPROACH ANALYSIS" panel (Code 1 -> Iterative,
    Code 2 -> Built-in function, etc). This intentionally stays simple;
    task_analyzer.py (a teammate's module) is the right place for deeper
    same-problem/different-approach reasoning — this is a lightweight
    local signal similarity_engine can use even if that module isn't
    wired up yet."""
    try:
        tree = parse_code(code)
    except CodeParseError:
        return "Unknown"

    import ast as _ast
    has_recursion = False
    has_loop = False
    builtin_heavy_calls = {"sorted", "sum", "max", "min", "map", "filter", "reduce", "any", "all"}
    uses_builtin = False
    has_comprehension = False

    for node in _ast.walk(tree):
        if isinstance(node, (_ast.For, _ast.While, _ast.AsyncFor)):
            has_loop = True
        if isinstance(node, (_ast.ListComp, _ast.DictComp, _ast.SetComp, _ast.GeneratorExp)):
            has_comprehension = True
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name):
            if node.func.id in builtin_heavy_calls:
                uses_builtin = True
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            fname = node.name
            for inner in _ast.walk(node):
                if isinstance(inner, _ast.Call) and isinstance(inner.func, _ast.Name) and inner.func.id == fname:
                    has_recursion = True

    if has_recursion:
        return "Recursive"
    if uses_builtin and not has_loop:
        return "Built-in function"
    if has_comprehension and not has_loop:
        return "Comprehension"
    if has_loop:
        return "Iterative"
    return "Direct/declarative"


def _approach_analysis(code1: str, code2: str, overall_score: float, ast_score: float) -> dict[str, Any]:
    approach1, approach2 = _approach_label(code1), _approach_label(code2)
    same_approach = approach1 == approach2

    notes = []
    notes.append("Same objective" if overall_score > 0.5 else "Objective unclear / low overall similarity")
    notes.append("Same implementation approach" if same_approach else "Different implementation")
    notes.append("High structural similarity" if ast_score > 0.7 else "Low structural similarity")

    return {
        "code1_approach": approach1,
        "code2_approach": approach2,
        "same_approach": same_approach,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compare_codes(code1: str, code2: str, weights: dict[str, float] | None = None) -> dict[str, Any]:
    """Compare two Python snippets across every available signal and
    return the full, structured, UI-ready result."""

    text_score = _text_similarity(code1, code2)
    ast_result = ast_similarity(code1, code2)
    fp_result = fingerprint_similarity(code1, code2)
    cf_result = control_flow_similarity(code1, code2)

    semantic_result = None
    semantic_available = codebert_analyzer.is_available()
    if semantic_available:
        try:
            semantic_result = codebert_analyzer.semantic_similarity(code1, code2)
        except codebert_analyzer.ModelUnavailableError:
            semantic_available = False

    active_weights = dict(weights) if weights else (
        DEFAULT_WEIGHTS if semantic_available else _WEIGHTS_WITHOUT_SEMANTIC
    )
    active_weights = _normalize_weights(active_weights)

    scores = {
        "text": text_score,
        "ast": ast_result["score"],
        "fingerprint": fp_result["score"],
        "control_flow": cf_result["score"],
    }
    if semantic_available:
        scores["semantic"] = semantic_result["score"]

    overall = sum(scores[k] * active_weights.get(k, 0.0) for k in scores)

    evidence = _generate_evidence(code1, code2, ast_result, fp_result, cf_result, text_score, semantic_result)
    approach = _approach_analysis(code1, code2, overall, ast_result["score"])

    return {
        "overall_similarity": round(overall, 4),
        "overall_similarity_pct": round(overall * 100),
        "component_scores": {
            "text_similarity": round(text_score, 4),
            "ast_similarity": ast_result["score"],
            "fingerprint_similarity": fp_result["score"],
            "control_flow_similarity": cf_result["score"],
            "semantic_similarity": (semantic_result["score"] if semantic_result else None),
        },
        "weights_used": active_weights,
        "semantic_available": semantic_available,
        "evidence": evidence,
        "approach_analysis": approach,
        "detail": {
            "ast": ast_result,
            "fingerprint": {k: v for k, v in fp_result.items() if k != "fingerprint_1" and k != "fingerprint_2"},
            "control_flow": cf_result,
            "semantic": semantic_result,
        },
    }


if __name__ == "__main__":
    a = """
def maximum(arr):
    m = arr[0]
    for x in arr:
        if x > m:
            m = x
    return m
"""
    b = "def maximum(arr):\n    return max(arr)\n"

    result = compare_codes(a, b)
    print(f"Overall similarity: {result['overall_similarity_pct']}%")
    for k, v in result["component_scores"].items():
        print(f"  {k}: {v}")
    print("Evidence:")
    for e in result["evidence"]:
        print(f"  - {e}")
    print("Approach:", result["approach_analysis"])
