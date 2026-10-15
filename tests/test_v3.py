"""Comprehensive unit tests for Chameleon v3 features:
- JIT hard safety URL validation
- ML intent router classification & fallback
- NLI grader claim extraction & formatting
- Diverse bucketing & leaf-page prioritization
- Code block and table chunking preservation
"""
from app.grader_nli import clean_markdown, extract_claims
from app.ingester import (
    _chunk,
    _content_hash_id,
    _split_into_semantic_blocks,
    classify_url_bucket,
    is_marketing_url,
)
from app.jit import is_safe_jit_url
from app.logic import after_grade, compute_grounding
from app.router_ml import classify_intent


def test_jit_safe_urls_accepted():
    target = "supabase.com"
    assert is_safe_jit_url("https://supabase.com/docs/guides/auth", target) is True
    assert is_safe_jit_url("https://docs.supabase.com/api/v1", target) is True
    assert is_safe_jit_url("https://supabase.com/docs/guides/database/row-level-security", target) is True
    assert is_safe_jit_url("https://auth.internal.supabase.com/docs", target) is True


def test_jit_safety_rejections():
    target = "supabase.com"

    # Reject non-HTTPS
    assert is_safe_jit_url("http://supabase.com/docs", target) is False
    assert is_safe_jit_url("ftp://supabase.com/docs", target) is False
    assert is_safe_jit_url("file:///etc/passwd", target) is False

    # Reject external or lookalike domains
    assert is_safe_jit_url("https://evil-supabase.com/docs", target) is False
    assert is_safe_jit_url("https://not-supabase.com", target) is False
    assert is_safe_jit_url("https://google.com", target) is False
    assert is_safe_jit_url("https://supabase.com.attacker.com/page", target) is False

    # Reject IP addresses (IPv4, IPv6, private, loopback, link-local)
    assert is_safe_jit_url("https://127.0.0.1/docs", target) is False
    assert is_safe_jit_url("https://192.168.1.1/docs", target) is False
    assert is_safe_jit_url("https://10.0.0.1/admin", target) is False
    assert is_safe_jit_url("https://169.254.169.254/latest/meta-data", target) is False
    assert is_safe_jit_url("https://[::1]/docs", target) is False
    assert is_safe_jit_url("https://0.0.0.0/test", target) is False

    # Reject localhost
    assert is_safe_jit_url("https://localhost/admin", target) is False
    assert is_safe_jit_url("https://api.localhost:3000", target) is False

    # Reject embedded credentials
    assert is_safe_jit_url("https://admin:secret@supabase.com/docs", target) is False

    # Reject non-443 ports
    assert is_safe_jit_url("https://supabase.com:8080/docs", target) is False
    assert is_safe_jit_url("https://supabase.com:9000/api", target) is False


def test_router_ml_classification():
    tech_query = "How do I configure row level security policies in postgres?"
    intent, prob, embed_ms, clf_ms = classify_intent(tech_query)
    assert intent == "technical_support"
    assert 0.0 <= prob <= 1.0
    assert embed_ms >= 0
    assert clf_ms >= 0

    comp_query = "How does Supabase compare to Firebase or Appwrite for pricing?"
    intent_comp, prob_comp, _, _ = classify_intent(comp_query)
    assert intent_comp == "competitor_battlecard"
    assert prob_comp >= 0.6


def test_claim_extraction_and_cleaning():
    draft = """
Here is how to set up authentication in Resend:
First, generate an API key in the API Keys settings [Source: https://resend.com/docs/api-keys].
Then, install the resend package with npm install resend.
The docs do not say anything about email limits.

| Feature | Resend | SendGrid |
|---|---|---|
| SDK Support | Native Node, Python | Node, Python, Ruby |
| Rate Limit | 100 emails/sec | 50 emails/sec |

I could not find information on custom SMTP servers.
"""
    claims = extract_claims(draft)
    assert len(claims) >= 3

    # Ensure citations are stripped
    assert not any("[Source:" in c for c in claims)
    # Ensure hedges are skipped
    assert not any("docs do not say" in c.lower() for c in claims)
    assert not any("could not find" in c.lower() for c in claims)
    # Ensure table rows were converted to claims
    assert any("SDK Support" in c and "Native Node" in c for c in claims)


def test_clean_markdown():
    text = "Enable **PostgreSQL** RLS via `ALTER TABLE` [Source: https://supabase.com/docs]."
    cleaned = clean_markdown(text)
    assert cleaned == "Enable PostgreSQL RLS via ALTER TABLE."


def test_bucketing_and_depth():
    b_auth, d_auth, s_auth = classify_url_bucket("https://supabase.com/docs/guides/auth/row-level-security")
    assert b_auth in ("auth", "database")
    assert d_auth == 4  # ["docs", "guides", "auth", "row-level-security"]
    assert s_auth >= 1

    b_db, d_db, s_db = classify_url_bucket("https://supabase.com/docs/postgres/migrations")
    assert b_db == "database"
    assert d_db == 3

    b_api, d_api, _ = classify_url_bucket("https://supabase.com/docs/api/rest")
    assert b_api == "api"
    assert d_api == 3


def test_marketing_url_detection():
    assert is_marketing_url("https://supabase.com/") is True
    assert is_marketing_url("https://supabase.com/about") is True
    assert is_marketing_url("https://supabase.com/blog") is True
    assert is_marketing_url("https://supabase.com/careers") is True

    # Tech pages must NOT be considered marketing
    assert is_marketing_url("https://supabase.com/docs") is False
    assert is_marketing_url("https://supabase.com/docs/guides/database") is False
    assert is_marketing_url("https://supabase.com/api/v1") is False


def test_code_block_chunking_preservation():
    code_block = "```python\ndef connect_db():\n    db = Database()\n    return db\n```"
    text = f"Here is the database connection guide:\n\n{code_block}\n\nCall connect_db() to start."
    blocks = _split_into_semantic_blocks(text)
    assert any(code_block in b for b in blocks)

    chunks = _chunk(text, "https://supabase.com/docs/db", "supabase.com", jit=False)
    assert len(chunks) >= 1
    # Check that code block was kept intact within a chunk
    assert any("def connect_db():" in c[0] and "return db" in c[0] for c in chunks)
    # Check metadata
    assert chunks[0][1]["source"] == "https://supabase.com/docs/db"
    assert chunks[0][1]["jit"] == "false"


def test_content_hash_id():
    id1 = _content_hash_id("col", "https://example.com/doc", "Same text")
    id2 = _content_hash_id("col", "https://example.com/doc", "Same text")
    id3 = _content_hash_id("col", "https://example.com/doc", "Different text")
    assert id1 == id2
    assert id1 != id3
    assert id1.startswith("col-")
