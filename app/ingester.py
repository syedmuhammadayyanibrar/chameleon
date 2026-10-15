"""Ingestion: discover pages with diverse bucketing & leaf prioritization, fetch via Jina, chunk safely, store in Chroma."""
import asyncio
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import chromadb
import httpx
from langchain_text_splitters import MarkdownHeaderTextSplitter

logger = logging.getLogger("chameleon.ingester")

MAX_PAGES = 15
MIN_PAGE_CHARS = 200
MIN_TECH_PAGES = int(os.getenv("MIN_TECH_PAGES", "5"))

_client = None

# Load bucket keywords from config/keywords.json
KEYWORDS_FILE = Path("config/keywords.json")


def _load_keywords() -> dict[str, list[str]]:
    if KEYWORDS_FILE.exists():
        try:
            with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {KEYWORDS_FILE}: {e}")
    return {
        "auth": ["auth", "login", "sso", "oauth", "jwt", "password", "session", "user", "mfa", "saml"],
        "database": ["postgres", "rls", "sql", "migrations", "schema", "table", "query", "database", "orm", "index"],
        "api": ["api", "rest", "graphql", "endpoints", "sdk", "methods", "client", "request", "routes"],
        "events": ["webhooks", "realtime", "events", "pubsub", "channels", "broadcast", "listener"],
        "deploy": ["deploy", "hosting", "cli", "config", "env", "docker", "ci", "cd", "cluster", "production"],
        "billing": ["pricing", "limits", "quotas", "plans", "tier", "subscription", "cost", "enterprise"],
        "quickstart": ["quickstart", "getting-started", "guide", "tutorial", "overview", "introduction", "setup"],
    }


BUCKET_KEYWORDS = _load_keywords()
MARKETING_SEGMENTS = {"", "about", "careers", "blog", "customers", "press", "contact", "legal", "terms", "privacy"}


def _chroma():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "./chroma_db"))
    return _client


def normalize_domain(value: str) -> str:
    v = value.strip().lower()
    if "://" not in v:
        v = "http://" + v
    host = urlparse(v).hostname or ""
    return host.removeprefix("www.")


def collection_name(domain: str) -> str:
    name = re.sub(r"[^a-z0-9]", "_", normalize_domain(domain)).strip("_")
    name = name[:63].strip("_")
    return name.ljust(3, "0")


def classify_url_bucket(url: str) -> tuple[str, int, int]:
    """Score each bucket by keyword matches in path and compute path depth.
    
    Returns (assigned_bucket, depth, keyword_matches_count).
    """
    p = urlparse(url).path.lower()
    segments = [s for s in p.strip("/").split("/") if s]
    depth = len(segments)

    best_bucket = "other"
    best_score = 0
    for bucket, kws in BUCKET_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in p)
        if score > best_score:
            best_score = score
            best_bucket = bucket

    return best_bucket, depth, best_score


def is_marketing_url(url: str) -> bool:
    """Check if URL points to a marketing / non-technical page."""
    p = urlparse(url).path.strip("/").lower()
    segments = [s for s in p.split("/") if s]
    if not segments:
        return True  # Root homepage is marketing
    # If explicitly in docs or api, it is not marketing
    if any(s in ("docs", "documentation", "api", "guides", "reference", "tutorial") for s in segments):
        return False
    return segments[0] in MARKETING_SEGMENTS


async def _get_text(client: httpx.AsyncClient, url: str) -> str:
    try:
        r = await client.get(url, timeout=15, follow_redirects=True)
        return r.text if r.status_code == 200 else ""
    except httpx.HTTPError:
        return ""


async def _llms_txt_urls(client: httpx.AsyncClient, domain: str) -> list[str]:
    urls: list[str] = []
    candidates = [
        f"https://{domain}/llms.txt",
        f"https://docs.{domain}/llms.txt",
        f"https://{domain}/docs/llms.txt",
    ]
    for target in candidates:
        text = await _get_text(client, target)
        if text:
            md_links = re.findall(r"\[.*?\]\((https?://[^\)\s]+)\)", text)
            plain_links = re.findall(r"^(https?://\S+)$", text, flags=re.MULTILINE)
            for u in md_links + plain_links:
                u_host = urlparse(u).hostname or ""
                if u_host.endswith(domain) and not u.endswith("/llms-full.txt"):
                    urls.append(u)
    return urls


async def _sitemap_urls(client: httpx.AsyncClient, sitemap_url: str, follow_index: bool = True) -> list[str]:
    xml = await _get_text(client, sitemap_url)
    locs = re.findall(r"<loc>\s*(.*?)\s*</loc>", xml, flags=re.S)
    if follow_index and "<sitemapindex" in xml:
        urls: list[str] = []
        for child in locs[:5]:
            urls += await _sitemap_urls(client, child, follow_index=False)
        return urls
    return locs


async def _one_hop_links(client: httpx.AsyncClient, base_url: str, domain: str) -> list[str]:
    """Fallback: extract one hop of links from docs.<domain> or <domain>/docs."""
    html = await _get_text(client, base_url)
    if not html:
        return []
    # Match href links
    found = re.findall(r'href=["\'](https?://[^"\']+|/[^"\']+)["\']', html)
    urls: list[str] = []
    for href in found:
        if href.startswith("/"):
            href = f"https://{domain}{href}"
        host = urlparse(href).hostname or ""
        if host.endswith(domain):
            urls.append(href.split("#")[0].split("?")[0])
    return list(dict.fromkeys(urls))


async def discover_pages(domain: str, extra_urls: list[str]) -> list[str]:
    """Discover pages using diverse bucketing and leaf-page prioritization."""
    home = f"https://{domain}"
    async with httpx.AsyncClient() as client:
        llms_urls = await _llms_txt_urls(client, domain)
        sitemap_found: list[str] = []
        for base in (domain, f"docs.{domain}"):
            sitemap_found += await _sitemap_urls(client, f"https://{base}/sitemap.xml")

        # Fallback if sitemap yielded few URLs
        if len(sitemap_found) < MIN_TECH_PAGES:
            for fallback_base in (f"https://docs.{domain}", f"https://{domain}/docs"):
                one_hop = await _one_hop_links(client, fallback_base, domain)
                sitemap_found.extend(one_hop)

    valid_candidates = list(dict.fromkeys(
        u for u in sitemap_found if (urlparse(u).hostname or "").endswith(domain)
    ))

    # Separate technical candidates from marketing pages
    tech_candidates: list[str] = []
    marketing_candidates: list[str] = []

    for u in valid_candidates:
        if is_marketing_url(u):
            marketing_candidates.append(u)
        else:
            tech_candidates.append(u)

    # Exclude marketing pages unless total tech pages < MIN_TECH_PAGES
    allowed_candidates = tech_candidates.copy()
    if len(allowed_candidates) < MIN_TECH_PAGES:
        allowed_candidates.extend(marketing_candidates)

    # Bucket candidates and sort by depth (leaf pages first)
    buckets: dict[str, list[tuple[int, int, str]]] = {}
    for u in allowed_candidates:
        bucket, depth, kw_score = classify_url_bucket(u)
        buckets.setdefault(bucket, []).append((depth, kw_score, u))

    # Sort each bucket by depth descending, then keyword score descending
    for b in buckets:
        buckets[b].sort(key=lambda item: (item[0], item[1]), reverse=True)

    # Selection: pick round-robin from non-empty buckets, highest depth first, at most 4 per bucket, cap 15 total
    selected_from_buckets: list[str] = []
    bucket_counts: dict[str, int] = {b: 0 for b in buckets}
    active_bucket_keys = [b for b in BUCKET_KEYWORDS.keys() if b in buckets and buckets[b]]
    if "other" in buckets and buckets["other"]:
        active_bucket_keys.append("other")

    while len(selected_from_buckets) < MAX_PAGES and active_bucket_keys:
        for b in list(active_bucket_keys):
            if buckets[b] and bucket_counts[b] < 4:
                item = buckets[b].pop(0)
                selected_from_buckets.append(item[2])
                bucket_counts[b] += 1
                if len(selected_from_buckets) >= MAX_PAGES:
                    break
            else:
                active_bucket_keys.remove(b)

    # Combine extra_urls, llms.txt URLs, and bucket-selected URLs
    ordered = list(extra_urls) + llms_urls + selected_from_buckets
    seen, result = set(), []
    for u in ordered:
        key = u.rstrip("/")
        if key not in seen:
            seen.add(key)
            result.append(u)
        if len(result) >= MAX_PAGES:
            break

    # If still under MAX_PAGES and home not seen and needed
    if len(result) < MIN_TECH_PAGES and home.rstrip("/") not in seen:
        result.append(home)

    return result[:MAX_PAGES]


async def _fetch(client: httpx.AsyncClient, sem: asyncio.Semaphore, url: str) -> str | None:
    headers = {}
    if os.getenv("JINA_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['JINA_API_KEY']}"
    async with sem:
        for attempt in range(3):
            try:
                r = await client.get(f"https://r.jina.ai/{url}", headers=headers, timeout=60)
                if r.status_code == 429 or r.status_code >= 500:
                    raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                if r.status_code != 200:
                    return None
                text = r.text.strip()
                return text if len(text) >= MIN_PAGE_CHARS else None
            except httpx.HTTPError:
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
    return None


def _split_into_semantic_blocks(content: str) -> list[str]:
    """Parse text into semantic blocks while preserving code blocks and table rows."""
    blocks: list[str] = []
    lines = content.splitlines(keepends=True)
    in_code = False
    in_table = False
    current_block: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Code fence detection
        if stripped.startswith("```"):
            if not in_code:
                # Flush previous non-code block
                if current_block:
                    blocks.append("".join(current_block).strip())
                    current_block = []
                in_code = True
                current_block.append(line)
            else:
                current_block.append(line)
                in_code = False
                blocks.append("".join(current_block).strip())
                current_block = []
            continue

        if in_code:
            current_block.append(line)
            continue

        # Table detection (starts and ends with |)
        is_table_line = stripped.startswith("|") and stripped.endswith("|")
        if is_table_line:
            if not in_table:
                if current_block:
                    blocks.append("".join(current_block).strip())
                    current_block = []
                in_table = True
            current_block.append(line)
            continue
        elif in_table:
            in_table = False
            blocks.append("".join(current_block).strip())
            current_block = []

        # Regular text: split by double newlines into paragraphs
        if stripped == "":
            if current_block:
                current_block.append(line)
                blocks.append("".join(current_block).strip())
                current_block = []
        else:
            current_block.append(line)

    if current_block:
        blocks.append("".join(current_block).strip())

    return [b for b in blocks if b]


def _chunk(markdown: str, url: str, domain: str, jit: bool = False) -> list[tuple[str, dict]]:
    """Chunk markdown document into semantically coherent pieces.
    
    Preserves code blocks up to 1200 chars and tables intact.
    Adds bucket, heading path, and jit metadata.
    """
    headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    sections = MarkdownHeaderTextSplitter(headers, strip_headers=False).split_text(markdown)

    bucket, _, _ = classify_url_bucket(url)
    chunks: list[tuple[str, dict]] = []

    for sec in sections:
        sec_text = sec.page_content
        h1 = sec.metadata.get("h1", "")
        h2 = sec.metadata.get("h2", "")
        h3 = sec.metadata.get("h3", "")
        heading_path = "/".join(filter(None, [h1, h2, h3])) or "root"

        blocks = _split_into_semantic_blocks(sec_text)
        current_chunk = ""

        for block in blocks:
            # If block is code block or table and large (> 1200), split at line boundaries
            if len(block) > 1200 and ("```" in block or block.startswith("|")):
                lines = block.splitlines(keepends=True)
                sub_chunk = ""
                for l in lines:
                    if len(sub_chunk) + len(l) > 800:
                        if sub_chunk:
                            chunks.append((
                                sub_chunk.strip(),
                                {
                                    "source": url,
                                    "domain": domain,
                                    "bucket": bucket,
                                    "heading": heading_path,
                                    "jit": str(jit).lower(),
                                },
                            ))
                        sub_chunk = l
                    else:
                        sub_chunk += l
                if sub_chunk:
                    chunks.append((
                        sub_chunk.strip(),
                        {
                            "source": url,
                            "domain": domain,
                            "bucket": bucket,
                            "heading": heading_path,
                            "jit": str(jit).lower(),
                        },
                    ))
                continue

            # Standard block accumulation up to 800 chars
            if len(current_chunk) + len(block) + 2 <= 800:
                current_chunk = (current_chunk + "\n\n" + block).strip() if current_chunk else block
            else:
                if current_chunk:
                    chunks.append((
                        current_chunk,
                        {
                            "source": url,
                            "domain": domain,
                            "bucket": bucket,
                            "heading": heading_path,
                            "jit": str(jit).lower(),
                        },
                    ))
                current_chunk = block

        if current_chunk:
            chunks.append((
                current_chunk,
                {
                    "source": url,
                    "domain": domain,
                    "bucket": bucket,
                    "heading": heading_path,
                    "jit": str(jit).lower(),
                },
            ))

    return chunks


def _content_hash_id(col_name: str, url: str, text: str) -> str:
    """Generate deterministic content hash ID to avoid duplicates."""
    digest = hashlib.sha256(f"{url}::{text}".encode("utf-8")).hexdigest()[:16]
    return f"{col_name}-{digest}"


def _store(domain: str, chunks: list[tuple[str, dict]]) -> None:
    name = collection_name(domain)
    try:
        _chroma().delete_collection(name)
    except Exception:
        pass
    col = _chroma().get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    # Deduplicate chunks by ID within batch
    seen_ids = set()
    unique_chunks = []
    for c in chunks:
        cid = _content_hash_id(name, c[1]["source"], c[0])
        if cid not in seen_ids:
            seen_ids.add(cid)
            unique_chunks.append((cid, c[0], c[1]))

    for i in range(0, len(unique_chunks), 100):
        batch = unique_chunks[i : i + 100]
        col.add(
            ids=[b[0] for b in batch],
            documents=[b[1] for b in batch],
            metadatas=[b[2] for b in batch],
        )


def add_chunks(domain: str, chunks: list[tuple[str, dict]], jit: bool = True) -> None:
    if not chunks:
        return
    name = collection_name(domain)
    col = _chroma().get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    seen_ids = set()
    unique_chunks = []
    for c in chunks:
        cid = _content_hash_id(name, c[1]["source"], c[0])
        if cid not in seen_ids:
            seen_ids.add(cid)
            unique_chunks.append((cid, c[0], c[1]))

    for i in range(0, len(unique_chunks), 100):
        batch = unique_chunks[i : i + 100]
        col.upsert(
            ids=[b[0] for b in batch],
            documents=[b[1] for b in batch],
            metadatas=[b[2] for b in batch],
        )


async def fetch_and_index_url(domain: str, url: str, jit: bool = False) -> list[dict]:
    sem = asyncio.Semaphore(1)
    async with httpx.AsyncClient() as client:
        text = await _fetch(client, sem, url)
    if not text:
        return []
    chunks = _chunk(text, url, domain, jit=jit)
    if chunks:
        await asyncio.to_thread(add_chunks, domain, chunks, jit)
    return [{"text": c[0], "source": c[1]["source"]} for c in chunks]


async def ingest_company(domain: str, urls: list[str] | None = None) -> dict:
    domain = normalize_domain(domain)
    pages = await discover_pages(domain, urls or [])
    sem = asyncio.Semaphore(3)
    async with httpx.AsyncClient() as client:
        texts = await asyncio.gather(*(_fetch(client, sem, u) for u in pages))
    chunks: list[tuple[str, dict]] = []
    indexed = 0
    for url, text in zip(pages, texts):
        if text:
            indexed += 1
            chunks += _chunk(text, url, domain, jit=False)
    await asyncio.to_thread(_store, domain, chunks)
    return {"domain": domain, "pages_indexed": indexed, "pages_failed": len(pages) - indexed, "chunks": len(chunks)}


def company_exists(domain: str) -> bool:
    try:
        _chroma().get_collection(collection_name(domain))
        return True
    except Exception:
        return False


def query_company(domain: str, query: str, k: int = 5) -> list[dict]:
    if not company_exists(domain):
        return []
    try:
        col = _chroma().get_collection(collection_name(domain))
        res = col.query(query_texts=[query], n_results=k)
        if not res or not res.get("documents") or not res["documents"][0]:
            return []
        return [{"text": t, "source": m.get("source", "")} for t, m in zip(res["documents"][0], res["metadatas"][0])]
    except Exception:
        return []


def list_companies() -> list[dict]:
    out = []
    try:
        cols = _chroma().list_collections()
    except Exception:
        return []
    for c in cols:
        name = c if isinstance(c, str) else c.name
        try:
            col = _chroma().get_collection(name)
            data = col.get(include=["metadatas"])
            metas = data.get("metadatas") or []
            if not metas:
                continue
            out.append(
                {
                    "domain": metas[0].get("domain", name),
                    "chunks": len(metas),
                    "pages": len({m.get("source") for m in metas if m and "source" in m}),
                }
            )
        except Exception:
            continue
    return out
