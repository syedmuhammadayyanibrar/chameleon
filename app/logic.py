"""Pure decision logic (no I/O)."""

GROUNDING_THRESHOLD = 0.75
MAX_RETRIES = 2


def compute_grounding(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def after_grade(state: dict) -> str:
    if (
        state.get("grounding_score", 0.0) >= GROUNDING_THRESHOLD
        or not state.get("retrieved_docs")
        or state.get("retries", 0) >= MAX_RETRIES
    ):
        return "finalize"
    return "rewrite"
