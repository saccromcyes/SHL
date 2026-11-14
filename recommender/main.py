"""FastAPI app: recommendation API, a SHL-assignment-compatible endpoint, and the web UI."""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter, defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from recommender import __version__
from recommender.catalog import TEST_TYPES
from recommender.config import ROOT, Settings, get_settings
from recommender.fetch import FetchError, fetch_job_text
from recommender.service import Recommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("recommender")
WEB = ROOT / "web"
EVAL_RESULTS = ROOT / "eval" / "results" / "latest.json"
_URL = re.compile(r"^https?://\S+$")


class Overrides(BaseModel):
    max_minutes: int | None = Field(default=None, ge=1, le=600)
    remote_testing: bool | None = None
    job_level: str | None = None


class RecommendRequest(BaseModel):
    query: str = Field(min_length=3, max_length=20000)
    overrides: Overrides | None = None


class AssignmentRequest(BaseModel):
    query: str = Field(min_length=3, max_length=20000)


class RateLimiter:
    def __init__(self, limit: int, window_s: int = 600):
        self.limit, self.window, self.hits = limit, window_s, defaultdict(deque)

    def check(self, key: str) -> None:
        now, hits = time.monotonic(), self.hits[key]
        while hits and now - hits[0] > self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            retry = int(self.window - (now - hits[0])) + 1
            raise HTTPException(429, f"Rate limit reached. Try again in {retry} seconds.", headers={"Retry-After": str(retry)})
        hits.append(now)


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.recommender = Recommender(settings)
        app.state.limiter = RateLimiter(settings.rate_limit_per_10min)
        log.info("Catalog loaded: %d assessments, mode=%s", len(app.state.recommender.items), app.state.recommender.mode)
        yield

    app = FastAPI(
        title="SHL Assessment Recommender", version=__version__, lifespan=lifespan, docs_url="/api/docs", redoc_url=None
    )

    async def resolve_text(query: str) -> tuple[str, str | None]:
        query = query.strip()
        if _URL.match(query):
            try:
                text = await run_in_threadpool(fetch_job_text, query, settings.fetch_timeout_s, settings.fetch_max_bytes)
            except FetchError as exc:
                raise HTTPException(400, str(exc)) from exc
            except Exception as exc:
                raise HTTPException(400, "Couldn't read that page. Paste the job description instead.") from exc
            return text[: settings.max_query_chars], query
        return query[: settings.max_query_chars], None

    @app.get("/api/health")
    @app.get("/health", include_in_schema=False)
    def health():
        return {"status": "healthy"}

    @app.get("/api/config")
    def config(request: Request):
        rec: Recommender = request.app.state.recommender
        return {
            "version": __version__,
            "mode": rec.mode,
            "llm_model": settings.llm_model if rec.llm else None,
            "embedder": getattr(rec.embedder, "name", "hashing"),
            "catalog_size": len(rec.items),
        }

    @app.get("/api/catalog/stats")
    def catalog_stats(request: Request):
        items = request.app.state.recommender.items
        tests = [a for a in items if not a.is_report]
        return {
            "assessments": len(tests),
            "reports_excluded": len(items) - len(tests),
            "by_type": {
                TEST_TYPES[k]: v for k, v in Counter(t for a in tests for t in a.test_types).most_common() if k in TEST_TYPES
            },
            "remote": sum(a.remote for a in tests),
            "adaptive": sum(a.adaptive for a in tests),
        }

    @app.post("/api/recommend")
    async def recommend(body: RecommendRequest, request: Request):
        request.app.state.limiter.check(client_ip(request))
        text, source = await resolve_text(body.query)
        overrides = {k: v for k, v in (body.overrides.model_dump() if body.overrides else {}).items() if v is not None}
        result = await run_in_threadpool(request.app.state.recommender.recommend, text, None, overrides)
        result["fetched_from"] = source
        result["input_preview"] = text[:400] if source else None
        return result

    # The response shape specified in the original SHL assignment.
    @app.post("/recommend")
    async def recommend_assignment(body: AssignmentRequest, request: Request):
        request.app.state.limiter.check(client_ip(request))
        text, _ = await resolve_text(body.query)
        result = await run_in_threadpool(request.app.state.recommender.recommend, text)
        return {
            "recommended_assessments": [
                {
                    "url": r["url"],
                    "name": r["name"],
                    "adaptive_support": "Yes" if r["adaptive"] else "No",
                    "description": r["description"],
                    "duration": r["minutes"],
                    "remote_support": "Yes" if r["remote"] else "No",
                    "test_type": r["test_type_names"],
                }
                for r in result["results"]
            ]
        }

    @app.get("/api/eval")
    def eval_results():
        if not EVAL_RESULTS.exists():
            return JSONResponse({"available": False})
        return {"available": True, **json.loads(EVAL_RESULTS.read_text())}

    if WEB.exists():
        app.mount("/static", StaticFiles(directory=WEB), name="static")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(WEB / "index.html")

    return app


app = create_app()
