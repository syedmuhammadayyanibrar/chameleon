import operator
from typing import Annotated, TypedDict


class GraphState(TypedDict, total=False):
    company_domain: str
    query: str
    search_query: str
    intent: str
    retrieved_docs: list[dict]
    sources: list[str]
    draft_answer: str
    grounding_score: float
    unsupported_claims: list[str]
    retries: int
    final_answer: str
    telemetry: Annotated[list[dict], operator.add]
