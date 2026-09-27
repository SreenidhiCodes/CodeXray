"""
codebert_analyzer.py
======================
AI semantic-similarity engine for CodeXray, built on Microsoft's
CodeBERT (microsoft/codebert-base).

Pipeline
--------
    Python code
        |
        v
    CodeBERT tokenizer (max 512 tokens, truncated)
        |
        v
    CodeBERT encoder
        |
        v
    Mean-pooled embedding (768-dim)
        |
        v
    Cosine similarity between two embeddings

Public API
----------
load_model(model_name: str = "microsoft/codebert-base") -> None
    Eagerly loads (and caches into models/codebert/) the tokenizer +
    model. Optional — get_embedding() will lazy-load on first call.

get_embedding(code: str) -> np.ndarray
    Returns a single 768-dim embedding vector for a code snippet.

generate_embedding(code: str) -> np.ndarray
    Alias for get_embedding(), matching the naming used in the spec.

semantic_similarity(code1: str, code2: str) -> dict
    Cosine similarity between the two snippets' embeddings, returned as
    both a raw float and a display-ready "Semantic Similarity: NN%".

batch_embeddings(snippets: list[str]) -> np.ndarray
    Embeds many snippets in one batched forward pass — used by batch
    analysis mode so we don't reload/re-run the model per pair.

Notes
-----
- Requires `torch` and `transformers`. Both are optional dependencies:
  if they are not installed, every function raises a clear
  ModelUnavailableError rather than crashing the whole app, so the rest
  of CodeXray (AST/logic/control-flow/text similarity) still works.
- The model is downloaded once from the Hugging Face Hub on first use
  and cached under `models/codebert/` (set via `HF_HOME` /
  `local_dir` below) so subsequent runs are offline-friendly.
- CodeBERT's max sequence length is 512 tokens. Longer files are
  truncated; `codebert_analyzer` never silently drops a whole file, it
  just flags `"truncated": True` in the result so the UI can warn the
  user that similarity was computed on a partial view of long inputs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

MODEL_NAME = "microsoft/codebert-base"
MODEL_CACHE_DIR = Path(__file__).resolve().parent.parent / "models" / "codebert"
MAX_TOKENS = 512


class ModelUnavailableError(RuntimeError):
    """Raised when torch/transformers aren't installed or the model
    can't be loaded (e.g. no network on first run)."""


# ---------------------------------------------------------------------------
# Lazy, cached model/tokenizer loading
# ---------------------------------------------------------------------------

_tokenizer = None
_model = None
_torch = None


def load_model(model_name: str = MODEL_NAME) -> None:
    """Load (or reuse) the CodeBERT tokenizer + model. Safe to call
    repeatedly; only loads once per process."""
    global _tokenizer, _model, _torch

    if _model is not None and _tokenizer is not None:
        return

    try:
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ModelUnavailableError(
            "torch and/or transformers are not installed. "
            "Run: pip install torch transformers"
        ) from exc

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))

    try:
        _tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
        _model = AutoModel.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
        _model.eval()
        _torch = torch
    except Exception as exc:  # noqa: BLE001 - surface any download/load failure uniformly
        raise ModelUnavailableError(
            f"Could not load '{model_name}'. Check your network connection "
            f"on first run (the model must download once). Original error: {exc}"
        ) from exc


def is_available() -> bool:
    """Check whether the CodeBERT backend can be used, without raising."""
    try:
        load_model()
        return True
    except ModelUnavailableError:
        return False


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _mean_pool(last_hidden_state, attention_mask):
    """Attention-mask-weighted mean pooling over token embeddings —
    standard, robust way to turn CodeBERT's per-token output into one
    fixed-size vector per snippet (better than just taking [CLS] for
    similarity tasks)."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def get_embedding(code: str) -> dict[str, Any]:
    """Embed a single code snippet.

    Returns a dict (not a bare vector) so callers get truncation info
    alongside the embedding:
        {"vector": np.ndarray[768], "truncated": bool, "num_tokens": int}
    """
    load_model()
    tokens = _tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
        padding=True,
    )
    full_length = len(_tokenizer.encode(code))
    truncated = full_length > MAX_TOKENS

    with _torch.no_grad():
        output = _model(**tokens)
        pooled = _mean_pool(output.last_hidden_state, tokens["attention_mask"])

    return {
        "vector": pooled[0].numpy(),
        "truncated": truncated,
        "num_tokens": min(full_length, MAX_TOKENS),
    }


def generate_embedding(code: str) -> dict[str, Any]:
    """Alias matching the naming convention requested in the project spec."""
    return get_embedding(code)


def batch_embeddings(snippets: list[str]) -> list[dict[str, Any]]:
    """Embed many snippets. Uses one batched forward pass per model call
    for efficiency in batch-analysis mode, while still returning one
    result dict per input snippet."""
    load_model()
    results = []
    # Batch in chunks of 8 to keep memory bounded on CPU-only machines.
    chunk_size = 8
    for start in range(0, len(snippets), chunk_size):
        chunk = snippets[start:start + chunk_size]
        tokens = _tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKENS,
            padding=True,
        )
        with _torch.no_grad():
            output = _model(**tokens)
            pooled = _mean_pool(output.last_hidden_state, tokens["attention_mask"])
        for i, code in enumerate(chunk):
            full_length = len(_tokenizer.encode(code))
            results.append({
                "vector": pooled[i].numpy(),
                "truncated": full_length > MAX_TOKENS,
                "num_tokens": min(full_length, MAX_TOKENS),
            })
    return results


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def _cosine(a, b) -> float:
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def semantic_similarity(code1: str, code2: str) -> dict[str, Any]:
    """Cosine similarity between two snippets' CodeBERT embeddings.

    Returns:
        {
          "score": 0.89,                       # raw cosine sim, in [-1, 1] clipped to [0, 1]
          "display": "Semantic Similarity: 89%",
          "truncated_1": False, "truncated_2": False,
        }
    """
    emb1 = get_embedding(code1)
    emb2 = get_embedding(code2)
    raw = _cosine(emb1["vector"], emb2["vector"])
    score = max(0.0, min(1.0, raw))  # clip - cosine can dip slightly negative on noise

    return {
        "score": round(score, 4),
        "display": f"Semantic Similarity: {round(score * 100)}%",
        "truncated_1": emb1["truncated"],
        "truncated_2": emb2["truncated"],
    }


if __name__ == "__main__":
    if not is_available():
        print(
            "CodeBERT backend unavailable in this environment "
            "(install torch + transformers, and ensure network access "
            "to huggingface.co on first run)."
        )
    else:
        a = "def add(a, b):\n    return a + b\n"
        b = "def sum_two(x, y):\n    total = x + y\n    return total\n"
        print(semantic_similarity(a, b))
