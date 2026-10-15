"""Just-In-Time (JIT) targeted ingestion for the retry loop."""
import asyncio
import ipaddress
import logging
import os
import time
from urllib.parse import urlparse

from ddgs import DDGS

from app.ingester import (
    _chroma,
    _fetch,
    add_chunks,
    collection_name,
    fetch_and_index_url,
)
from app.state import GraphState

logger = logging.getLogger("chameleon.jit")

JIT_MAX_PAGES_PER_DOMAIN = int(os.getenv("JIT_MAX_PAGES_PER_DOMAIN", "50"))

# In-memory tracking of JIT-indexed URLs per domain
_DOMAIN_JIT_URLS: dict[str, set[str]] = {}


def is_safe_jit_url(url: str, target_domain: str) -> bool:
    """Validate that a URL satisfies hard safety rules for JIT ingestion:
    - Must use HTTPS scheme
    - Hostname must match target_domain or be a direct subdomain of it
    - Reject IP addresses (IPv4, IPv6, private, link-local, loopback)
    - Reject localhost
    - Reject ports other than 443
    - Reject embedded credentials
    """
    if not url or not isinstance(url, str):
        return False

    url = url.strip()
    if not url.startswith("https://"):
        return False

    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.scheme != "https":
        return False

    if parsed.username or parsed.password:
        return False

    if parsed.port and parsed.port != 443:
        return False

    host = (parsed.hostname or "").strip().lower()
    if not host:
        return False

    # Reject localhost
    if host == "localhost" or host.endswith(".localhost"):
        return False

    # Reject IP addresses
    try:
        ipaddress.ip_address(host)
        return False
    except ValueError:
        pass  # Not a plain IP, continue

    # Check for hex/octal/integer/bracketed IP representations
    if host.startswith("[") and host.endswith("]"):
        return False
    parts = host.split(".")
    if len(parts) == 4 and all(p.isdigit() for p in parts):
        return False

    target = target_domain.strip().lower().removeprefix("www.")
    # Must equal target or end with .target
    if host == target or host.endswith("." + target):
        return True

    return False


def get_indexed_sources(domain: str) -> set[str]:
    """Retrieve all distinct source URLs currently stored for this company."""
    sources: set[str] = set()
    try:
        name = collection_name(domain)
        col = _chroma().get_collection(name)
        data = col.get(include=["metadatas"])
        for meta in data.get("metadatas") or []:
            if meta and "source" in meta:
                sources.add(meta["source"].rstrip("/"))
    except Exception:
        pass
    return sources


def get_domain_jit_count(domain: str) -> int:
    """Get the total number of JIT pages ingested for this domain."""
    return len(_DOMAIN_JIT_URLS.get(domain, set()))


async def search_domain_web(domain: str, query: str, max_results: int = 5) -> list[str]:
    """Query duckduckgo for official domain pages matching search query."""
    def _search():
        try:
            ddgs = DDGS()
            q = f"site:{domain} {query}"
            results = ddgs.text(q, max_results=max_results) or []
            return [r.get("href", "") for r in results if r.get("href")]
        except Exception as e:
            logger.warning(f"DDGS JIT search error: {e}")
            return []

    return await asyncio.to_thread(_search)


async def jit_ingest(state: GraphState) -> dict:
    """JIT ingestion node executed during technical support retries.
    
    Wired as: grader (low score, retries left) -> rewrite -> jit_ingest -> support_rag.
    """
    t0 = time.perf_counter()
    domain = state["company_domain"]
    query = state.get("search_query") or state.get("query", "")
    intent = state.get("intent", "technical_support")

    # Only run for technical support
    if intent != "technical_support":
        return {}

    # Budget: at most 1 JIT round per request
    telemetry = state.get("telemetry") or []
    if any(e.get("node") == "jit_ingest" for e in telemetry):
        return {}

    try:
        # Check domain budget
        current_jit_pages = _DOMAIN_JIT_URLS.setdefault(domain, set())
        remaining_domain_budget = max(0, JIT_MAX_PAGES_PER_DOMAIN - len(current_jit_pages))
        if remaining_domain_budget <= 0:
            return {
                "telemetry": [
                    {
                        "node": "jit_ingest",
                        "ms": int((time.perf_counter() - t0) * 1000),
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "cost_usd": 0.0,
                        "detail": f"jit skipped: domain cap reached ({JIT_MAX_PAGES_PER_DOMAIN} pages)",
                    }
                ]
            }

        existing_sources = await asyncio.to_thread(get_indexed_sources, domain)
        raw_urls = await search_domain_web(domain, query, max_results=5)

        # Filter safe and unindexed URLs
        candidate_urls: list[str] = []
        for u in raw_urls:
            if is_safe_jit_url(u, domain):
                norm_u = u.rstrip("/")
                if norm_u not in existing_sources and norm_u not in current_jit_pages:
                    candidate_urls.append(u)

        # Budget: at most 3 new pages per round
        urls_to_fetch = candidate_urls[: min(3, remaining_domain_budget)]
        if not urls_to_fetch:
            return {
                "telemetry": [
                    {
                        "node": "jit_ingest",
                        "ms": int((time.perf_counter() - t0) * 1000),
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "cost_usd": 0.0,
                        "detail": "jit +0 pages, +0 chunks 0ms (no new candidate URLs)",
                    }
                ]
            }

        # Fetch and index using Jina
        total_chunks = 0
        indexed_urls = []
        for url in urls_to_fetch:
            new_chunks = await fetch_and_index_url(domain, url, jit=True)
            if new_chunks:
                total_chunks += len(new_chunks)
                indexed_urls.append(url)
                current_jit_pages.add(url.rstrip("/"))

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        detail = f"jit +{len(indexed_urls)} pages, +{total_chunks} chunks {elapsed_ms}ms"
        if indexed_urls:
            detail += f" ({', '.join(indexed_urls)})"

        return {
            "telemetry": [
                {
                    "node": "jit_ingest",
                    "ms": elapsed_ms,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "cost_usd": 0.0,
                    "detail": detail,
                }
            ]
        }

    except Exception as e:
        logger.warning(f"JIT ingest encountered non-fatal error: {e}", exc_info=True)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "telemetry": [
                {
                    "node": "jit_ingest",
                    "ms": elapsed_ms,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "cost_usd": 0.0,
                    "detail": f"jit failed: {str(e)[:50]}",
                }
            ]
        }
