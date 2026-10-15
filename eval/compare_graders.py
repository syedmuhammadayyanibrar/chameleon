"""Compare LLM Grader vs Local NLI Grader on eval/qa.jsonl.

Measures:
- Agreement rate (grounded vs low-confidence)
- Disagreement analysis (differences in claims & scores)
- Latency comparison (ms)
- Cost comparison (USD)
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from app.agents import Claims, _cost, _ctx, _llm, _structured, _text
from app.grader_nli import grade_with_nli
from app.ingester import company_exists, query_company
from app.logic import GROUNDING_THRESHOLD, compute_grounding

HERE = Path(__file__).parent
QA_FILE = HERE / "qa.jsonl"
OUT_FILE = HERE / "grader_comparison.json"

ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gemini-3.5-flash-lite")
GRADER_MODEL = os.getenv("GRADER_MODEL", "gemini-3.1-flash-lite")


async def run_comparison():
    print("=" * 60)
    print("CHAMELEON GRADER COMPARISON: LLM vs NLI")
    print("=" * 60)

    if not QA_FILE.exists():
        print(f"Error: {QA_FILE} not found")
        return

    rows = [json.loads(line) for line in QA_FILE.read_text().splitlines() if line.strip()]
    comparisons = []

    llm_latencies = []
    nli_latencies = []
    llm_costs = []
    nli_costs = []

    for r in rows:
        domain = r["domain"]
        q = r["question"]

        if not company_exists(domain):
            print(f"Skipping {domain} (not ingested)")
            continue

        print(f"\nEvaluating: [{domain}] {q}")
        docs = query_company(domain, q, 5)
        if not docs:
            print("  No docs retrieved, skipping.")
            continue

        # Generate draft answer using ANSWER_MODEL
        draft, tin, tout = await _text(
            ANSWER_MODEL,
            "You answer questions about a company using ONLY the provided context. Cite sources inline as "
            "[Source: url] after each claim. If the context does not contain the answer, say exactly what is "
            "missing instead of guessing.",
            f"Context:\n{_ctx(docs)}\n\nQuestion: {q}",
        )

        # 1. LLM Grader
        t0_llm = time.perf_counter()
        parsed, g_tin, g_tout = await _structured(
            GRADER_MODEL,
            Claims,
            "Split the draft answer into atomic factual claims. For each, set supported=true only if the "
            "context directly supports it.",
            f"Context:\n{_ctx(docs)}\n\nDraft answer:\n{draft}",
        )
        llm_ms = int((time.perf_counter() - t0_llm) * 1000)
        llm_score = compute_grounding([c.supported for c in parsed.claims])
        llm_cost = _cost(GRADER_MODEL, g_tin, g_tout)
        llm_grounded = llm_score >= GROUNDING_THRESHOLD

        # 2. Local NLI Grader
        t0_nli = time.perf_counter()
        nli_res = grade_with_nli(draft, [d["text"] for d in docs])
        nli_ms = nli_res["elapsed_ms"]
        nli_score = nli_res["grounding_score"]
        nli_cost = 0.0  # Local CPU model runs at 0 API cost
        nli_grounded = nli_score >= GROUNDING_THRESHOLD

        agree = llm_grounded == nli_grounded
        llm_latencies.append(llm_ms)
        nli_latencies.append(nli_ms)
        llm_costs.append(llm_cost)
        nli_costs.append(nli_cost)

        status = "AGREE" if agree else "DISAGREE"
        print(f"  {status}: LLM score={llm_score:.2f} ({llm_ms}ms, ${llm_cost:.6f}) vs NLI score={nli_score:.2f} ({nli_ms}ms, $0.0)")

        comparisons.append({
            "domain": domain,
            "question": q,
            "should_refuse": r.get("should_refuse", False),
            "draft_answer": draft,
            "llm": {
                "score": llm_score,
                "grounded": llm_grounded,
                "latency_ms": llm_ms,
                "cost_usd": llm_cost,
                "unsupported": [c.claim for c in parsed.claims if not c.supported],
            },
            "nli": {
                "score": nli_score,
                "grounded": nli_grounded,
                "latency_ms": nli_ms,
                "cost_usd": 0.0,
                "min_claim_score": nli_res["min_claim_score"],
                "mean_entail": nli_res["mean_entail"],
                "unsupported": nli_res["unsupported_claims"],
            },
            "agreement": agree,
        })

    if not comparisons:
        print("No comparisons run.")
        return

    n = len(comparisons)
    agreed = sum(1 for c in comparisons if c["agreement"])
    summary = {
        "samples_evaluated": n,
        "agreement_count": agreed,
        "agreement_rate": round(agreed / n, 4),
        "llm_mean_latency_ms": round(sum(llm_latencies) / n, 2),
        "nli_mean_latency_ms": round(sum(nli_latencies) / n, 2),
        "speedup_factor": round((sum(llm_latencies) / n) / max(1, (sum(nli_latencies) / n)), 2),
        "llm_total_cost_usd": round(sum(llm_costs), 6),
        "nli_total_cost_usd": 0.0,
        "disagreements": [c for c in comparisons if not c["agreement"]],
    }

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Agreement Rate: {summary['agreement_rate'] * 100:.1f}% ({agreed}/{n})")
    print(f"Avg Latency:    LLM={summary['llm_mean_latency_ms']}ms | NLI={summary['nli_mean_latency_ms']}ms")
    print(f"Total Cost:     LLM=${summary['llm_total_cost_usd']:.6f} | NLI=$0.000000")

    OUT_FILE.write_text(json.dumps({"summary": summary, "details": comparisons}, indent=2))
    print(f"Full details written to {OUT_FILE}")
    return summary


if __name__ == "__main__":
    asyncio.run(run_comparison())
