"""LangGraph compilation for Chameleon."""
from langgraph.graph import END, START, StateGraph

from app.agents import competitor_intel, finalize, grader, rewrite, router, support_rag
from app.jit import jit_ingest
from app.logic import after_grade
from app.state import GraphState

_ROUTER_BY_INTENT = {
    "technical_support": "support_rag",
    "competitor_battlecard": "competitor_intel",
}

_REWRITE_BY_INTENT = {
    "technical_support": "jit_ingest",
    "competitor_battlecard": "competitor_intel",
}


def _build():
    g = StateGraph(GraphState)
    for name, fn in [
        ("router", router),
        ("support_rag", support_rag),
        ("competitor_intel", competitor_intel),
        ("grader", grader),
        ("rewrite", rewrite),
        ("jit_ingest", jit_ingest),
        ("finalize", finalize),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", lambda s: _ROUTER_BY_INTENT[s["intent"]], list(_ROUTER_BY_INTENT.values()))
    g.add_edge("support_rag", "grader")
    g.add_edge("competitor_intel", "grader")
    g.add_conditional_edges("grader", after_grade, {"finalize": "finalize", "rewrite": "rewrite"})

    # On rewrite: technical support goes through JIT ingestion before re-retrieval; competitor skips it
    g.add_conditional_edges("rewrite", lambda s: _REWRITE_BY_INTENT[s["intent"]], list(_REWRITE_BY_INTENT.values()))
    g.add_edge("jit_ingest", "support_rag")

    g.add_edge("finalize", END)
    return g.compile()


app_graph = _build()
