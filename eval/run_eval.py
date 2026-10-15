"""Run the graph over eval/qa.jsonl. Usage: python -m eval.run_eval"""
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from app.graph import app_graph  # noqa: E402
from app.ingester import company_exists  # noqa: E402

HERE = Path(__file__).parent


async def main() -> None:
    rows = [json.loads(line) for line in (HERE / "qa.jsonl").read_text().splitlines() if line.strip()]
    results = []
    for r in rows:
        if not company_exists(r["domain"]):
            print(f"skip (not ingested): {r['domain']}")
            continue
        try:
            state = await app_graph.ainvoke(
                {"company_domain": r["domain"], "query": r["question"], "telemetry": []},
                config={"recursion_limit": 25},
            )
            answer = state.get("final_answer", "")
            low = answer.startswith("Confidence low")
            says_missing = any(p in answer.lower() for p in ("does not contain", "doesn't contain", "not found", "no information", "missing"))
            refused = low or says_missing
            if r["should_refuse"]:
                ok = refused
            else:  # keyword check, independent of the LLM judge
                ok = (not low) and any(k.lower() in answer.lower() for k in r["expect_any"])
            results.append({**r, "ok": ok, "retries": state.get("retries", 0), "score": state.get("grounding_score", 0.0), "answer": answer})
            print(("PASS" if ok else "FAIL"), r["question"])
        except Exception as e:
            print("ERROR", r["question"], type(e).__name__, str(e)[:100])
            results.append({**r, "ok": False, "error": f"{type(e).__name__}: {str(e)[:100]}"})
        await asyncio.sleep(2)

    ans = [x for x in results if not x["should_refuse"]]
    ref = [x for x in results if x["should_refuse"]]
    pct = lambda xs: round(100 * sum(x["ok"] for x in xs) / len(xs), 1) if xs else None  # noqa: E731
    summary = {
        "answerable_correct_pct": pct(ans),
        "unanswerable_refused_pct": pct(ref),
        "avg_retries": round(sum(x["retries"] for x in results) / len(results), 2) if results else None,
    }
    print(json.dumps(summary, indent=2))
    (HERE / "results.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
