"""Supervised ML intent router using SentenceTransformer + LogisticRegression."""
import json
import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

ROUTER_EMBED_MODEL = os.getenv("ROUTER_EMBED_MODEL", "all-MiniLM-L6-v2")
ROUTER_MIN_CONF = float(os.getenv("ROUTER_MIN_CONF", "0.6"))

_embedder: SentenceTransformer | None = None
_classifier: LogisticRegression | None = None
_is_initialized: bool = False


def get_router_models(intents_path: str = "data/intents.jsonl") -> Tuple[SentenceTransformer, LogisticRegression]:
    """Load embedder and train classifier once at startup."""
    global _embedder, _classifier, _is_initialized
    if _is_initialized and _embedder is not None and _classifier is not None:
        return _embedder, _classifier

    if _embedder is None:
        _embedder = SentenceTransformer(ROUTER_EMBED_MODEL)

    texts: list[str] = []
    labels: list[str] = []
    p = Path(intents_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    texts.append(item["text"])
                    labels.append(item["label"])

    if not texts:
        raise ValueError(f"No training data found in {intents_path}")

    # Embed training set and train LogisticRegression
    embeddings = _embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(embeddings, labels)

    _classifier = clf
    _is_initialized = True
    return _embedder, _classifier


def classify_intent(query: str, intents_path: str = "data/intents.jsonl") -> Tuple[str, float, int, int]:
    """Classify user query intent using ML.
    
    Returns:
        (intent, probability, embed_ms, clf_ms)
    """
    embedder, classifier = get_router_models(intents_path)

    t0_emb = time.perf_counter()
    q_emb = embedder.encode([query], show_progress_bar=False, normalize_embeddings=True)
    embed_ms = int((time.perf_counter() - t0_emb) * 1000)

    t0_clf = time.perf_counter()
    probs = classifier.predict_proba(q_emb)[0]
    clf_ms = int((time.perf_counter() - t0_clf) * 1000)

    idx = int(np.argmax(probs))
    intent = str(classifier.classes_[idx])
    prob = float(probs[idx])

    return intent, prob, embed_ms, clf_ms
