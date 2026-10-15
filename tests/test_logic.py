from app.ingester import collection_name, normalize_domain
from app.logic import after_grade, compute_grounding


def test_grounding_fraction():
    assert compute_grounding([True, True, False, True]) == 0.75
    assert compute_grounding([False, False]) == 0.0
    assert compute_grounding([True]) == 1.0


def test_grounding_zero_claims():
    assert compute_grounding([]) == 0.0


def test_after_grade_pass():
    assert after_grade({"grounding_score": 0.8, "retrieved_docs": [{"x": 1}], "retries": 0}) == "finalize"


def test_after_grade_retry():
    assert after_grade({"grounding_score": 0.4, "retrieved_docs": [{"x": 1}], "retries": 1}) == "rewrite"


def test_after_grade_max_retries():
    assert after_grade({"grounding_score": 0.4, "retrieved_docs": [{"x": 1}], "retries": 2}) == "finalize"


def test_after_grade_no_docs():
    assert after_grade({"grounding_score": 0.0, "retrieved_docs": [], "retries": 0}) == "finalize"


def test_normalize_domain():
    assert normalize_domain("https://www.Stripe.com/docs/api") == "stripe.com"
    assert normalize_domain("docs.example.io") == "docs.example.io"
    assert normalize_domain("  http://example.com?x=1 ") == "example.com"


def test_collection_naming():
    assert collection_name("https://www.stripe.com") == "stripe_com"
    assert collection_name("a.b") == "a_b"
    assert collection_name("x.y") == "x_y"
    name = collection_name("a" * 100 + ".com")
    assert 3 <= len(name) <= 63
    assert collection_name("ab") == "ab0"
