"""Local cross-encoder NLI grader.

Scores factual claims using an NLI cross-encoder model.
NOTE: The resulting grounding score is an NLI entailment probability,
not calibrated ground truth.
"""
import os
import re
import time
from typing import Tuple

import numpy as np
from sentence_transformers import CrossEncoder

NLI_MODEL = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
NLI_CLAIM_THRESHOLD = float(os.getenv("NLI_CLAIM_THRESHOLD", "0.5"))

_cross_encoder: CrossEncoder | None = None
_entailment_idx: int = 1
_is_initialized: bool = False

HEDGE_PATTERNS = [
    r"docs?\s+do(?:es)?\s*(?:n['’]?t|not)\s+say",
    r"docs?\s+(?:do(?:es)?\s*(?:n['’]?t|not)|lack)",
    r"context\s+does\s+(?:not|n['’]?t)",
    r"not\s+mentioned\s+in",
    r"not\s+found\s+in",
    r"no\s+information\s+(?:found|provided|available)",
    r"it\s+is\s+unclear",
    r"i\s+could\s+not\s+find",
    r"missing\s+from\s+the\s+documentation",
    r"refer\s+to\s+the\s+official\s+docs",
    r"does\s+not\s+contain",
]
_HEDGE_RE = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)


def get_nli_model() -> Tuple[CrossEncoder, int]:
    """Load the CrossEncoder model once at startup and determine the entailment index."""
    global _cross_encoder, _entailment_idx, _is_initialized
    if _is_initialized and _cross_encoder is not None:
        return _cross_encoder, _entailment_idx

    model = CrossEncoder(NLI_MODEL)
    cfg = getattr(model, "config", None) or getattr(model.model, "config", None)
    id2label = getattr(cfg, "id2label", {0: "contradiction", 1: "entailment", 2: "neutral"})

    entail_idx = 1
    for k, v in id2label.items():
        if "entail" in str(v).lower():
            entail_idx = int(k)
            break

    _cross_encoder = model
    _entailment_idx = entail_idx
    _is_initialized = True
    return _cross_encoder, _entailment_idx


def clean_markdown(text: str) -> str:
    """Remove Markdown syntax, citations, and formatting from a claim."""
    # Strip [Source: ...] citations
    s = re.sub(r"\[Source:\s*[^\]]+\]", "", text, flags=re.IGNORECASE)
    # Strip markdown links [text](url) -> text
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    # Strip bold, italic, code markers, headers
    s = re.sub(r"[*_`#]", "", s)
    # Strip leading bullets or numbers
    s = re.sub(r"^\s*[-*•\d\.\)]+\s*", "", s)
    # Normalize whitespace before punctuation (e.g. 'word .' -> 'word.')
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)
    # Normalize internal multi-spaces
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def extract_claims(draft: str) -> list[str]:
    """Split draft answer into atomic claims (sentences and table rows), skipping hedges."""
    claims: list[str] = []
    lines = draft.splitlines()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check if line is a table row (starts and ends with |)
        if stripped.startswith("|") and stripped.endswith("|"):
            inner = stripped[1:-1].strip()
            # Skip separator rows like |---|---|
            if set(inner.replace("|", "").replace(":", "").strip()) <= {"-"}:
                continue
            cells = [clean_markdown(c) for c in inner.split("|")]
            # Skip header rows that are generic labels
            if any(c.lower() in ("feature", "pricing", "competitor", "aspect", "category") for c in cells):
                continue
            row_claim = " - ".join(c for c in cells if c)
            if len(row_claim) >= 10 and not _HEDGE_RE.search(row_claim):
                claims.append(row_claim)
            continue

        # Regular prose lines: split by sentence
        cleaned_line = clean_markdown(stripped)
        if not cleaned_line:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", cleaned_line)
        for sent in sentences:
            s = sent.strip()
            if len(s) < 10:
                continue
            if _HEDGE_RE.search(s):
                continue
            # Skip pure URL lines or punctuation
            if not any(c.isalnum() for c in s):
                continue
            claims.append(s)

    return claims


def grade_with_nli(
    draft: str,
    chunks: list[str],
    threshold: float = NLI_CLAIM_THRESHOLD,
) -> dict:
    """Grade a draft answer against retrieved chunks using CrossEncoder NLI.
    
    Computes entailment probability for all (chunk, claim) pairs in one batched predict call.
    Returns:
        {
            "grounding_score": float,
            "unsupported_claims": list[str],
            "claims": list[str],
            "claim_scores": list[float],
            "min_claim_score": float,
            "mean_entail": float,
            "supported_count": int,
            "total_claims": int,
            "elapsed_ms": int,
        }
    """
    t0 = time.perf_counter()
    claims = extract_claims(draft)

    if not claims or not chunks:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "grounding_score": 0.0,
            "unsupported_claims": claims,
            "claims": claims,
            "claim_scores": [0.0] * len(claims),
            "min_claim_score": 0.0,
            "mean_entail": 0.0,
            "supported_count": 0,
            "total_claims": len(claims),
            "elapsed_ms": elapsed_ms,
        }

    model, entail_idx = get_nli_model()

    # Create (premise=chunk, hypothesis=claim) pairs
    # Model max length handles truncation automatically
    pairs: list[list[str]] = []
    for claim in claims:
        for chunk in chunks:
            pairs.append([chunk, claim])

    # Single batched predict call with softmax
    # CrossEncoder.predict(..., apply_softmax=True) returns array of shape (N, num_labels)
    preds = model.predict(pairs, apply_softmax=True, show_progress_bar=False)
    entail_probs = preds[:, entail_idx]

    # Map back to each claim: claim_score = max over chunks
    num_chunks = len(chunks)
    claim_scores: list[float] = []
    unsupported: list[str] = []

    for i, claim in enumerate(claims):
        claim_entails = entail_probs[i * num_chunks : (i + 1) * num_chunks]
        best_score = float(np.max(claim_entails))
        claim_scores.append(best_score)
        if best_score < threshold:
            unsupported.append(claim)

    supported_count = len(claims) - len(unsupported)
    grounding_score = round(supported_count / len(claims), 4) if claims else 0.0
    min_score = round(float(np.min(claim_scores)), 4) if claim_scores else 0.0
    mean_entail = round(float(np.mean(claim_scores)), 4) if claim_scores else 0.0
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "grounding_score": grounding_score,
        "unsupported_claims": unsupported,
        "claims": claims,
        "claim_scores": claim_scores,
        "min_claim_score": min_score,
        "mean_entail": mean_entail,
        "supported_count": supported_count,
        "total_claims": len(claims),
        "elapsed_ms": elapsed_ms,
    }
