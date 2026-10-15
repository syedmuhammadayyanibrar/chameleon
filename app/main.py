from dotenv import load_dotenv

load_dotenv()  # must run before importing modules that create LLM clients

import asyncio  # noqa: E402
import hmac  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

from fastapi import FastAPI, Header, HTTPException, Query, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from slowapi import Limiter, _rate_limit_exceeded_handler  # noqa: E402
from slowapi.errors import RateLimitExceeded  # noqa: E402
from slowapi.util import get_remote_address  # noqa: E402

from app.graph import app_graph  # noqa: E402
from app.ingester import company_exists, ingest_company, list_companies, normalize_domain  # noqa: E402

log = logging.getLogger("chameleon")
ROOT = Path(__file__).resolve().parent.parent
DAILY_QUERY_CAP = int(os.getenv("DAILY_QUERY_CAP", "200"))
_daily = {"day": None, "n": 0}
limiter = Limiter(key_func=get_remote_address)


async def _seed() -> None:
    path = ROOT / "seed" / "companies.json"
    if not path.exists():
        return
    for c in json.loads(path.read_text()):
        if company_exists(c["domain"]):
            continue
        try:
            log.info("seeding %s: %s", c["domain"], await ingest_company(c["domain"], c.get("urls")))
        except Exception as e:
            log.error("seed failed for %s: %s", c["domain"], type(e).__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Pre-warm router & grader models once at startup
    if os.getenv("ROUTER_MODE", "ml").lower() == "ml":
        try:
            from app.router_ml import get_router_models
            await asyncio.to_thread(get_router_models)
            log.info("ML router model pre-warmed successfully")
        except Exception as e:
            log.warning(f"Router pre-warm warning: {e}")
    if os.getenv("GRADER_MODE", "nli").lower() == "nli":
        try:
            from app.grader_nli import get_nli_model
            await asyncio.to_thread(get_nli_model)
            log.info("NLI grader model pre-warmed successfully")
        except Exception as e:
            log.warning(f"NLI grader pre-warm warning: {e}")

    task = asyncio.create_task(_seed())  # keep a reference so it isn't garbage-collected
    yield
    task.cancel()


app = FastAPI(title="Chameleon", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:7860")],
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


class IngestRequest(BaseModel):
    domain: str
    urls: list[str] = []


@app.post("/ingest")
@limiter.limit("5/hour")
async def ingest(request: Request, body: IngestRequest, x_api_key: str | None = Header(default=None)):
    admin = os.getenv("ADMIN_API_KEY")
    if not admin or not x_api_key or not hmac.compare_digest(x_api_key.encode(), admin.encode()):
        raise HTTPException(401, "Invalid or missing API key")
    if not normalize_domain(body.domain):
        raise HTTPException(422, "Invalid domain")
    return await ingest_company(body.domain, body.urls)


@app.get("/companies")
async def companies():
    examples = {}
    path = ROOT / "seed" / "companies.json"
    if path.exists():
        examples = {c["domain"]: c.get("example_question", "") for c in json.loads(path.read_text())}
    return [{**c, "example": examples.get(c["domain"], "")} for c in list_companies()]


def _take_daily() -> None:
    today = date.today()
    if _daily["day"] != today:
        _daily.update(day=today, n=0)
    if _daily["n"] >= DAILY_QUERY_CAP:
        raise HTTPException(429, "Daily query limit reached. Please try again tomorrow.")
    _daily["n"] += 1


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


@app.get("/stream")
@limiter.limit("10/minute")
async def stream(request: Request, domain: str, query: str = Query(min_length=3, max_length=500)):
    domain = normalize_domain(domain)
    if not domain or not company_exists(domain):
        raise HTTPException(404, "Company not ingested")
    _take_daily()

    async def events():
        t0 = time.perf_counter()
        ms = lambda: int((time.perf_counter() - t0) * 1000)  # noqa: E731
        try:
            init = {"company_domain": domain, "query": query, "telemetry": []}
            async for chunk in app_graph.astream(init, stream_mode="updates", config={"recursion_limit": 25}):
                for node, update in chunk.items():
                    update = dict(update or {})
                    if "retrieved_docs" in update:
                        update["n_chunks"] = len(update.pop("retrieved_docs"))
                    yield _sse({"node": node, "update": update, "elapsed_ms": ms()})
            yield _sse({"node": "__done__", "elapsed_ms": ms()})
        except Exception as e:
            yield _sse({"node": "__error__", "message": type(e).__name__})

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# Serve the static UI from the same origin (must be mounted last).
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
