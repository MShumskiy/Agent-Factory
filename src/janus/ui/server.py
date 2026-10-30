"""The UI's HTTP side: a JSON API over scenarios, trees and jobs, and the static app.

It is meant for this machine. `janus ui` binds to 127.0.0.1, every response carries a content security
policy that blocks loading anything from elsewhere, the app ships its own font and logo, and the API
calls models only through jobs, which pass the same privacy checks as the command line.
"""

from __future__ import annotations

from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from janus.llm.client import LLMError, LLMPSettings
from janus.llm.privacy import PrivacyError
from janus.sim.engine import ENGINE_VERSION
from janus.sim.expr import ExprError
from janus.sim.scenario import Scenario, ScenarioError
from janus.sim.tree import ExploreSettings
from janus.ui.data import Catalog, scenario_summary, tree_payload
from janus.ui.jobs import Busy, JobRunner, LLMFactory, default_llm_factory

STATIC = Path(__file__).parent / "static"
CSP = "; ".join([
    "default-src 'self'", "script-src 'self'", "style-src 'self'", "img-src 'self'", "font-src 'self'",
    "connect-src 'self'", "object-src 'none'", "base-uri 'none'", "frame-ancestors 'none'", "form-action 'self'",
])


class ExploreRequest(BaseModel):
    options: list[int] | None = None
    depth: int = Field(default=2, ge=1, le=24)
    branches: int = Field(default=3, ge=1, le=10)
    candidates: int = Field(default=1, ge=1, le=4)
    rollouts: int = Field(default=0, ge=0, le=10)
    budget: int = Field(default=40, ge=1, le=5000)
    seed: int | None = None


class RunRequest(BaseModel):
    option: int = Field(default=0, ge=0)
    seed: int | None = None
    steps: int | None = Field(default=None, ge=1)


def create_app(scenarios: Path, out: Path, llm_factory: LLMFactory | None = None) -> FastAPI:
    catalog = Catalog(scenarios, out)
    jobs = JobRunner(catalog, llm_factory or default_llm_factory(catalog))
    app = FastAPI(title="JANUS", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.catalog = catalog
    app.state.jobs = jobs

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        # API answers are never stored; the app's files are revalidated on every load, so an update shows at once.
        response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api/") else "no-cache"
        return response

    def scenario(name: str) -> Scenario:
        try:
            return catalog.get(name)
        except KeyError:
            raise HTTPException(404, f"no scenario named {name!r}") from None

    @app.get("/api/status")
    def status() -> dict:
        running = jobs.running()
        result = {"engine": ENGINE_VERSION, "llmp": None, "llmp_ok": False,
                  "running": running.as_dict() if running else None}
        try:
            settings = LLMPSettings.from_env()
        except LLMError as e:
            return {**result, "error": str(e)}
        result["llmp"] = settings.base_url
        try:
            reply = httpx.get(f"{settings.base_url}/health", headers={"Authorization": settings.api_key}, timeout=2)
            result["llmp_ok"] = reply.status_code == 200
        except httpx.HTTPError:
            pass
        return result

    @app.get("/api/scenarios")
    def scenarios_list() -> list[dict]:
        return [scenario_summary(catalog, path, s) for path, s in catalog.entries()]

    @app.get("/api/scenarios/{name}/tree")
    def tree(name: str, opponents: str = "expected", focal: str | None = None) -> dict:
        payload = tree_payload(catalog, scenario(name), opponents, focal)
        if payload is None:
            raise HTTPException(404, f"{name} has not been explored yet")
        return payload

    @app.post("/api/scenarios/{name}/explore", status_code=202)
    def explore(name: str, body: ExploreRequest) -> dict:
        scenario(name)
        settings = ExploreSettings(depth=body.depth, branches=body.branches, candidates=body.candidates,
                                   rollouts=body.rollouts, budget=body.budget)
        return _start(lambda: jobs.explore(name, settings, body.options, body.seed))

    @app.post("/api/scenarios/{name}/run", status_code=202)
    def run(name: str, body: RunRequest) -> dict:
        scenario(name)
        return _start(lambda: jobs.run(name, body.option, body.seed, body.steps))

    @app.get("/api/jobs")
    def jobs_list() -> list[dict]:
        return [job.as_dict() for job in reversed(list(jobs.jobs.values()))]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict:
        if (found := jobs.jobs.get(job_id)) is None:
            raise HTTPException(404, f"no job {job_id}")
        return found.as_dict()

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app


def _start(start) -> dict:
    try:
        return start().as_dict()
    except Busy as e:
        raise HTTPException(409, str(e)) from e
    except (PrivacyError, LLMError, ScenarioError, ExprError) as e:
        raise HTTPException(400, str(e)) from e
