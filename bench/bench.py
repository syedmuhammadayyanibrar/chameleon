"""Chameleon v3 Performance & Quality Benchmark.

Measures:
1. ML Router latency (embed vs classify, 200 iterations, p50 and p95)
2. ML Router quality (5-fold cross validation: accuracy and macro F1)
3. Cross-Encoder NLI Grader latency (5 claims x 5 chunks, 200 iterations, p50 and p95)
4. System metadata: CPU model, torch thread count

Writes results to bench/results.json.
"""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from app.grader_nli import get_nli_model
from app.router_ml import get_router_models


def get_cpu_info() -> str:
    """Detect CPU model on Windows, Linux, or macOS."""
    if platform.system() == "Windows":
        try:
            out = subprocess.check_output("wmic cpu get name", shell=True, text=True)
            lines = [l.strip() for l in out.splitlines() if l.strip() and "Name" not in l]
            if lines:
                return lines[0]
        except Exception:
            pass
        return os.environ.get("PROCESSOR_IDENTIFIER", platform.processor())
    elif platform.system() == "Darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


def bench_router_quality(intents_path: str = "data/intents.jsonl", n_splits: int = 5) -> dict:
    """Evaluate router classification quality using stratified 5-fold cross-validation."""
    embedder, _ = get_router_models(intents_path)
    texts, labels = [], []
    with open(intents_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                texts.append(item["text"])
                labels.append(item["label"])

    X = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    y = np.array(labels)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies, f1s = [], []

    from sklearn.linear_model import LogisticRegression

    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        accuracies.append(accuracy_score(y[test_idx], preds))
        f1s.append(f1_score(y[test_idx], preds, average="macro"))

    return {
        "n_samples": len(texts),
        "n_splits": n_splits,
        "mean_accuracy": round(float(np.mean(accuracies)), 4),
        "mean_macro_f1": round(float(np.mean(f1s)), 4),
        "fold_accuracies": [round(float(a), 4) for a in accuracies],
        "fold_f1s": [round(float(f), 4) for f in f1s],
    }


def bench_router_latency(n_iter: int = 200) -> dict:
    """Benchmark router embedding vs logistic regression classification over n_iter runs."""
    embedder, classifier = get_router_models()
    sample_queries = [
        "How do I configure row level security policies?",
        "What are the best alternatives to Supabase for Postgres?",
        "Can I use JWT tokens to authenticate API calls?",
        "How does Clerk compare to Auth0 for pricing?",
        "Where can I find the webhook signing secret?",
    ]

    embed_times = []
    clf_times = []
    total_times = []

    # Warm-up
    for q in sample_queries:
        emb = embedder.encode([q], show_progress_bar=False, normalize_embeddings=True)
        classifier.predict_proba(emb)

    for i in range(n_iter):
        q = sample_queries[i % len(sample_queries)]

        t0 = time.perf_counter()
        emb = embedder.encode([q], show_progress_bar=False, normalize_embeddings=True)
        t_emb = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        classifier.predict_proba(emb)
        t_clf = (time.perf_counter() - t1) * 1000

        embed_times.append(t_emb)
        clf_times.append(t_clf)
        total_times.append(t_emb + t_clf)

    return {
        "iterations": n_iter,
        "embed_p50_ms": round(float(np.percentile(embed_times, 50)), 2),
        "embed_p95_ms": round(float(np.percentile(embed_times, 95)), 2),
        "clf_p50_ms": round(float(np.percentile(clf_times, 50)), 2),
        "clf_p95_ms": round(float(np.percentile(clf_times, 95)), 2),
        "total_p50_ms": round(float(np.percentile(total_times, 50)), 2),
        "total_p95_ms": round(float(np.percentile(total_times, 95)), 2),
    }


def bench_grader_latency(n_iter: int = 200) -> dict:
    """Benchmark CrossEncoder NLI grader with 5 claims x 5 chunks (25 pairs) over n_iter runs."""
    model, entail_idx = get_nli_model()

    sample_claims = [
        "Row Level Security is enabled with ALTER TABLE name ENABLE ROW LEVEL SECURITY.",
        "Policies restrict access using SQL expressions returning boolean values.",
        "API keys are generated in the project settings dashboard under API credentials.",
        "Webhooks deliver JSON payloads with retry backoff up to 24 hours.",
        "PostgreSQL 15 supports foreign data wrappers for remote querying.",
    ]
    sample_chunks = [
        "Supabase Postgres allows enabling RLS on any user table via standard SQL commands.",
        "Security policies specify whether rows are visible or modifiable by role and session claims.",
        "Project settings expose both publishable and service role secret keys.",
        "Webhook triggers automatically send event data to designated HTTPS endpoints.",
        "Extensions including pg_stat_statements and wrappers can be activated via UI or SQL.",
    ]

    pairs = [[chunk, claim] for claim in sample_claims for chunk in sample_chunks]

    # Warm-up
    model.predict(pairs, apply_softmax=True, show_progress_bar=False)

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        preds = model.predict(pairs, apply_softmax=True, show_progress_bar=False)
        _ = preds[:, entail_idx]
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "iterations": n_iter,
        "pairs_per_call": len(pairs),
        "grader_p50_ms": round(float(np.percentile(times, 50)), 2),
        "grader_p95_ms": round(float(np.percentile(times, 95)), 2),
    }


def run_all_benchmarks(output_file: str = "bench/results.json"):
    print("=" * 60)
    print("CHAMELEON v3 PERFORMANCE BENCHMARKS")
    print("=" * 60)

    cpu = get_cpu_info()
    threads = torch.get_num_threads()
    print(f"CPU Model: {cpu}")
    print(f"PyTorch Thread Count: {threads}")

    print("\n[1/3] Benchmarking ML Router Quality (5-fold CV)...")
    quality = bench_router_quality()
    print(f"  Accuracy: {quality['mean_accuracy'] * 100:.2f}% | Macro F1: {quality['mean_macro_f1']:.4f}")

    print("\n[2/3] Benchmarking ML Router Latency (200 iterations)...")
    router_lat = bench_router_latency(200)
    print(f"  Embed: p50={router_lat['embed_p50_ms']}ms, p95={router_lat['embed_p95_ms']}ms")
    print(f"  Clf:   p50={router_lat['clf_p50_ms']}ms, p95={router_lat['clf_p95_ms']}ms")
    print(f"  Total: p50={router_lat['total_p50_ms']}ms, p95={router_lat['total_p95_ms']}ms")

    print("\n[3/3] Benchmarking NLI Grader Latency (25 pairs, 200 iterations)...")
    grader_lat = bench_grader_latency(200)
    print(f"  Grader: p50={grader_lat['grader_p50_ms']}ms, p95={grader_lat['grader_p95_ms']}ms")

    results = {
        "system": {
            "cpu": cpu,
            "torch_threads": threads,
            "platform": platform.platform(),
        },
        "router_quality": quality,
        "router_latency": router_lat,
        "grader_latency": grader_lat,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved benchmark results to {output_file}")
    return results


if __name__ == "__main__":
    run_all_benchmarks()
