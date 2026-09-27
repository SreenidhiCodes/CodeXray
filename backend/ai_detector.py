"""
ai_detector.py
================
Estimates how strongly a single Python snippet's *characteristics*
resemble typical AI-generated code, as an indicator — NOT a proof, and
NOT the same computation as CodeBERT similarity in codebert_analyzer.py.

Two tiers, used in this priority order:

  1. Trained classifier (preferred): if a scikit-learn model has been
     fit and saved to models/ai_detector/classifier.joblib (via
     `train_classifier()` below, using datasets/human_code/ and
     datasets/ai_generated_code/), `predict_ai_likelihood()` uses it.

  2. Heuristic fallback (always available): a hand-built, explainable
     scoring function over stylistic features — variable-naming
     regularity, docstring/comment density, formatting consistency,
     structural uniformity, error-handling presence, line-length
     variance, etc. This is what ships before a labeled dataset exists,
     and it's also what makes every score explainable (see
     `explain_indicators()`).

Public API
----------
extract_features(code: str) -> dict
    Numeric feature vector used by both the heuristic and the trained
    classifier.

predict_ai_likelihood(code: str) -> dict
    {"score": 0.78, "display": "AI-generation likelihood: 78%",
     "method": "classifier" | "heuristic", "confidence": "low"|"medium"|"high"}

explain_indicators(code: str, features: dict | None = None) -> list[str]
    Human-readable bullet points behind the score, e.g.
    "Highly regular structure", "Repetitive naming pattern".

train_classifier(human_dir, ai_dir, out_path=...) -> dict
    Fits a simple logistic-regression classifier on extracted features
    from the two labeled dataset folders and saves it. Returns basic
    train-set metrics. (Real evaluation belongs in evaluation/evaluation.py
    against a held-out split — this is just the fit step.)

IMPORTANT: every public score returned here is explicitly phrased as a
likelihood/indicator (see PHRASING NOTE below) and callers (report_generator.py,
app.py) should preserve that phrasing rather than asserting certainty.
"""

from __future__ import annotations

import ast
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ast_analyzer import parse_code, CodeParseError

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "ai_detector" / "classifier.joblib"

# PHRASING NOTE: never render this as "this code IS N% AI-generated".
# Always "AI-generation likelihood/indicator: N%" — see spec section 7.
DISPLAY_TEMPLATE = "AI-generation likelihood: {pct}%"


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _identifier_names(tree: ast.AST) -> list[str]:
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store,)):
            names.append(node.id)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(node.name)
            names.extend(a.arg for a in node.args.args)
    return names


def _naming_regularity(names: list[str]) -> float:
    """0..1 — how uniform the naming *style* is (all snake_case, all
    similar length, low reuse of generic single-letter names mixed with
    descriptive ones). AI output tends to be very consistently styled;
    human code is messier (mixes i/j/tmp with descriptive names)."""
    if not names:
        return 0.0
    snake = sum(1 for n in names if re.fullmatch(r"[a-z][a-z0-9_]*", n))
    style_consistency = snake / len(names)

    lengths = [len(n) for n in names]
    mean_len = sum(lengths) / len(lengths)
    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
    length_uniformity = 1 / (1 + variance / 10)  # low variance -> close to 1

    return round(0.5 * style_consistency + 0.5 * length_uniformity, 4)


def _docstring_density(tree: ast.AST) -> float:
    """Fraction of functions/classes with a docstring. AI-generated code
    very often documents every function; human code (especially student
    code, scripts, competitive-programming code) usually doesn't."""
    defs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    if not defs:
        return 0.0
    documented = sum(1 for n in defs if ast.get_docstring(n))
    return round(documented / len(defs), 4)


def _comment_density(code: str) -> float:
    lines = code.splitlines()
    code_lines = [l for l in lines if l.strip()]
    if not code_lines:
        return 0.0
    comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
    return round(comment_lines / len(code_lines), 4)


def _formatting_consistency(code: str) -> float:
    """Checks indentation consistency and blank-line spacing regularity —
    AI output is typically PEP8-perfect and evenly spaced."""
    lines = code.splitlines()
    indents = [len(l) - len(l.lstrip(" ")) for l in lines if l.strip()]
    if not indents:
        return 0.0
    # Consistent multiples of 4 -> high score
    mult4 = sum(1 for i in indents if i % 4 == 0)
    indent_consistency = mult4 / len(indents)

    blank_runs = [len(g) for g in re.findall(r"(?:\n[ \t]*){2,}", code + "\n")]
    blank_consistency = 1.0 if len(set(blank_runs)) <= 1 else 0.6

    return round(0.7 * indent_consistency + 0.3 * blank_consistency, 4)


def _error_handling_presence(tree: ast.AST) -> float:
    """AI-generated solutions to typical prompts frequently include
    defensive try/except and input validation even when not asked for."""
    has_try = any(isinstance(n, ast.Try) for n in ast.walk(tree))
    has_type_hints = any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            n.returns is not None or any(a.annotation is not None for a in n.args.args)
        )
        for n in ast.walk(tree)
    )
    return round(0.5 * float(has_try) + 0.5 * float(has_type_hints), 4)


def _structural_uniformity(tree: ast.AST) -> float:
    """How similar in size/shape all top-level functions are. AI
    solutions to multi-part prompts often produce suspiciously
    same-shaped functions; humans vary more per function."""
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(funcs) < 2:
        return 0.0
    sizes = [len(list(ast.walk(f))) for f in funcs]
    mean_size = sum(sizes) / len(sizes)
    variance = sum((s - mean_size) ** 2 for s in sizes) / len(sizes)
    return round(1 / (1 + variance / 25), 4)


def extract_features(code: str) -> dict[str, float]:
    """Return the full named feature vector for a snippet. All values
    are normalized to [0, 1] so they can be combined or fed to a linear
    classifier without additional scaling."""
    tree = parse_code(code)
    names = _identifier_names(tree)

    return {
        "naming_regularity": _naming_regularity(names),
        "docstring_density": _docstring_density(tree),
        "comment_density": min(_comment_density(code), 1.0),
        "formatting_consistency": _formatting_consistency(code),
        "error_handling_presence": _error_handling_presence(tree),
        "structural_uniformity": _structural_uniformity(tree),
    }


# ---------------------------------------------------------------------------
# Heuristic scoring (always available, fully explainable)
# ---------------------------------------------------------------------------

_HEURISTIC_WEIGHTS = {
    "naming_regularity": 0.20,
    "docstring_density": 0.20,
    "comment_density": 0.10,
    "formatting_consistency": 0.20,
    "error_handling_presence": 0.15,
    "structural_uniformity": 0.15,
}

_INDICATOR_LABELS = {
    "naming_regularity": "Highly regular, consistent identifier naming",
    "docstring_density": "Every function documented with a docstring",
    "comment_density": "Dense, evenly-distributed comments",
    "formatting_consistency": "Very consistent formatting/indentation",
    "error_handling_presence": "Defensive error handling / type hints present",
    "structural_uniformity": "Suspiciously uniform function sizes",
}

_THRESHOLD = 0.55  # feature value above which it counts as a "signal" in the explanation


def _heuristic_score(features: dict[str, float]) -> float:
    return sum(features[k] * w for k, w in _HEURISTIC_WEIGHTS.items())


def explain_indicators(code: str, features: dict[str, float] | None = None) -> list[str]:
    """Explainable-AI-style bullet list: which features pushed the score
    up, phrased as indicators rather than proof (spec section 8)."""
    features = features or extract_features(code)
    indicators = [
        _INDICATOR_LABELS[k] for k, v in features.items() if v >= _THRESHOLD
    ]
    if not indicators:
        indicators.append("No strong AI-style indicators detected — code reads as organically written")
    return indicators


# ---------------------------------------------------------------------------
# Optional trained classifier
# ---------------------------------------------------------------------------

def _feature_vector(features: dict[str, float]) -> list[float]:
    # Fixed key order so classifier training/inference always agree.
    keys = list(_HEURISTIC_WEIGHTS.keys())
    return [features[k] for k in keys]


def train_classifier(human_dir: str, ai_dir: str, out_path: Path | None = None) -> dict[str, Any]:
    """Fit a simple logistic-regression classifier on labeled examples.

    Expects `human_dir` and `ai_dir` to each contain `.py` files. Saves
    the fitted model with joblib and returns basic training metrics.
    Requires scikit-learn + joblib (optional dependencies, same pattern
    as codebert_analyzer's torch/transformers).
    """
    try:
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn and joblib are required to train the classifier. "
            "Run: pip install scikit-learn joblib"
        ) from exc

    X, y = [], []
    for label, directory in ((0, human_dir), (1, ai_dir)):
        for path in Path(directory).glob("*.py"):
            try:
                code = path.read_text(encoding="utf-8")
                feats = extract_features(code)
            except (CodeParseError, UnicodeDecodeError):
                continue
            X.append(_feature_vector(feats))
            y.append(label)

    if len(set(y)) < 2:
        raise RuntimeError("Need labeled examples from BOTH human_dir and ai_dir to train.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    out_path = out_path or MODEL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_path)

    return metrics


def _load_classifier():
    if not MODEL_PATH.exists():
        return None
    try:
        import joblib
        return joblib.load(MODEL_PATH)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Public prediction entry point
# ---------------------------------------------------------------------------

def predict_ai_likelihood(code: str) -> dict[str, Any]:
    """Main entry point. Uses the trained classifier if one has been
    fitted and saved; otherwise falls back to the explainable heuristic.
    Always phrased as a likelihood/indicator, never a certainty claim."""
    features = extract_features(code)
    clf = _load_classifier()

    if clf is not None:
        try:
            proba = clf.predict_proba([_feature_vector(features)])[0][1]
            score = float(proba)
            method = "classifier"
        except Exception:  # noqa: BLE001 - fall back safely on any inference error
            score = _heuristic_score(features)
            method = "heuristic"
    else:
        score = _heuristic_score(features)
        method = "heuristic"

    score = max(0.0, min(1.0, score))
    pct = round(score * 100)

    if 0.35 <= score <= 0.65:
        confidence = "low"
    elif 0.2 <= score < 0.35 or 0.65 < score <= 0.8:
        confidence = "medium"
    else:
        confidence = "high"

    return {
        "score": round(score, 4),
        "display": DISPLAY_TEMPLATE.format(pct=pct),
        "method": method,
        "confidence": confidence,
        "indicators": explain_indicators(code, features),
        "features": features,
        "disclaimer": (
            "This is an estimated indicator based on code style patterns, "
            "not proof of AI generation. Treat it as one signal among many."
        ),
    }


if __name__ == "__main__":
    sample = '''
def calculate_sum(numbers: list[int]) -> int:
    """Calculate the sum of a list of numbers.

    Args:
        numbers: List of integers to sum.

    Returns:
        The total sum.
    """
    try:
        total = 0
        for number in numbers:
            total += number
        return total
    except TypeError as error:
        raise ValueError("Input must contain only integers") from error


def calculate_average(numbers: list[int]) -> float:
    """Calculate the average of a list of numbers.

    Args:
        numbers: List of integers to average.

    Returns:
        The average value.
    """
    try:
        return calculate_sum(numbers) / len(numbers)
    except ZeroDivisionError as error:
        raise ValueError("Cannot average an empty list") from error
'''
    result = predict_ai_likelihood(sample)
    print(result["display"], f"(method={result['method']}, confidence={result['confidence']})")
    for i in result["indicators"]:
        print(" -", i)
