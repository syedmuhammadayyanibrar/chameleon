"""Ingestion: discover pages, fetch via Jina Reader, chunk, store in Chroma."""
import asyncio
import os
import re
from urllib.parse import urlparse

import chromadb
import httpx
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

MAX_PAGES = 15
MIN_PAGE_CHARS = 200
PRIORITY_WORDS = ["docs", "api", "pricing", "about", "guide", "features", "integrations", "quickstart", "faq"]

_client = None


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


def _score(url: str) -> int:
    u = url.lower()
    return sum(w in u for w in PRIORITY_WORDS)


async def _get_text(client: httpx.AsyncClient, url: str) -> str:
    try:
        r = await client.get(url, timeout=15, follow_redirects=True)
        return r.text if r.status_code == 200 else ""
    except httpx.HTTPError:
        return ""


async def _sitemap_urls(client: httpx.AsyncClient, sitemap_url: str, follow_index: bool = True) -> list[str]:
    xml = await _get_text(client, sitemap_url)
    locs = re.findall(r"<loc>\s*(.*?)\s*</loc>", xml, flags=re.S)
    if follow_index and "<sitemapindex" in xml:
        urls: list[str] = []
        for child in locs[:5]:
            urls += await _sitemap_urls(client, child, follow_index=False)
        return urls
    return locs


async def discover_pages(domain: str, extra_urls: list[str]) -> list[str]:
    home = f"https://{domain}"
    async with httpx.AsyncClient() as client:
        found: list[str] = []
        for base in (domain, f"docs.{domain}"):
            found += await _sitemap_urls(client, f"https://{base}/sitemap.xml")
    found = [u for u in found if (urlparse(u).hostname or "").endswith(domain)]
    found.sort(key=_score, reverse=True)
    ordered = [home] + list(extra_urls) + found
    seen, result = set(), []
    for u in ordered:
        key = u.rstrip("/")
        if key not in seen:
            seen.add(key)
            result.append(u)
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


def _chunk(markdown: str, url: str, domain: str) -> list[tuple[str, dict]]:
    headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    sections = MarkdownHeaderTextSplitter(headers, strip_headers=False).split_text(markdown)
    docs = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(sections)
    return [(d.page_content, {"source": url, "domain": domain}) for d in docs]


def _store(domain: str, chunks: list[tuple[str, dict]]) -> None:
    name = collection_name(domain)
    try:
        _chroma().delete_collection(name)
    except Exception:  # collection did not exist
        pass
    col = _chroma().create_collection(name, metadata={"hnsw:space": "cosine"})
    for i in range(0, len(chunks), 100):
        batch = chunks[i : i + 100]
        col.add(
            ids=[f"{name}-{i + j}" for j in range(len(batch))],
            documents=[c[0] for c in batch],
            metadatas=[c[1] for c in batch],
        )


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
            chunks += _chunk(text, url, domain)
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
    col = _chroma().get_collection(collection_name(domain))
    res = col.query(query_texts=[query], n_results=k)
    return [{"text": t, "source": m["source"]} for t, m in zip(res["documents"][0], res["metadatas"][0])]


def list_companies() -> list[dict]:
    out = []
    for c in _chroma().list_collections():
        col = _chroma().get_collection(c if isinstance(c, str) else c.name)
        metas = col.get(include=["metadatas"])["metadatas"]
        if not metas:
            continue
        out.append(
            {
                "domain": metas[0]["domain"],
                "chunks": len(metas),
                "pages": len({m["source"] for m in metas}),
            }
        )
    return out
