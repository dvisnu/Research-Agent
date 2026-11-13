"""The web UI's server: a JSON API over the store, and the built React app.

It listens on 127.0.0.1 only. Requests must name a local host (against DNS rebinding),
and requests that change something must carry the ``X-Research-Agent`` header, which a
page on another site cannot add without a CORS preflight this server never grants, so
no other site open in the browser can start runs or spend quota.
"""

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import ValidationError

from research_agent.config import Settings
from research_agent.costs import run_cost
from research_agent.pipeline import STAGES
from research_agent.pipeline.stored_report import stored_report
from research_agent.report import RunReport
from research_agent.storage import Database, new_id
from research_agent.web import views
from research_agent.web.jobs import CONSOLE_LOG, Busy, Jobs

STATIC = Path(__file__).parent / "static"
HEADER = "x-research-agent"
LOCAL_HOSTS = ("127.0.0.1", "localhost", "[::1]")
FILES = {"report.md": "text/markdown", "report.json": "application/json",
         "provenance.json": "application/json", CONSOLE_LOG: "text/plain"}
_RUN_ID = re.compile(r"^run_[0-9a-f]+$")
MAX_QUESTION = 4000
CONSOLE_TAIL = 40
_EMPTY_REPORT = RunReport(run_id="", question="", generated_at="").model_dump(mode="json")


def _with_defaults(value: Any, default: Any) -> Any:
    """The stored report in the current shape: a part it lacks or holds as null takes the empty default."""
    if value is None:
        return default
    if isinstance(value, dict) and isinstance(default, dict):
        return {**value, **{key: _with_defaults(value.get(key), d) for key, d in default.items()}}
    return value

_NOT_BUILT = """<!doctype html><meta charset="utf-8"><title>Research Agent</title>
<body style="font-family:system-ui;max-width:40rem;margin:4rem auto;line-height:1.5">
<h1>The web UI isn't built yet</h1>
<p>Build it once (Node 20+), then reload:</p>
<pre>cd frontend
npm install
npm run build</pre>
<p>The API is up: <a href="/api/docs">/api/docs</a>.</p></body>"""


def create_app(settings: Settings, jobs: Jobs | None = None) -> FastAPI:
    jobs = jobs or Jobs(settings)
    app = FastAPI(title="Research Agent", docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)

    @app.middleware("http")
    async def local_only(request: Request, call_next: Any) -> Response:
        if _host(request.headers.get("host") or "") not in LOCAL_HOSTS:
            return JSONResponse({"detail": "This server only answers on localhost."}, status_code=403)
        if request.method not in ("GET", "HEAD", "OPTIONS") and request.headers.get(HEADER) != "1":
            return JSONResponse({"detail": f"Missing the {HEADER} header."}, status_code=403)
        return await call_next(request)

    def db() -> Iterator[Database]:
        with Database(settings.db_path) as database:
            yield database

    Db = Annotated[Database, Depends(db)]

    def run_or_404(database: Database, run_id: str) -> dict[str, Any]:
        rows = views.run_rows(database, [run_id]) if _RUN_ID.match(run_id) else []
        if not rows:
            raise HTTPException(404, f"No run {run_id}.")
        return _with_state(rows[0])

    def _with_state(run: dict[str, Any]) -> dict[str, Any]:
        job = jobs.get(run["run_id"])
        run["state"] = views.state(run, job.alive if job else None)
        run["started_here"] = job is not None
        return run

    @app.get("/api/status")
    def status(database: Db) -> dict[str, Any]:
        """The run going now (here or elsewhere), and how the agent is set up (never the keys)."""
        active = jobs.active()
        running = [_with_state(r) for r in views.run_rows(database, limit=50) if r["status"] == "running"]
        try:
            current = Settings()  # read .env again: a key may have been switched since
            setup: dict[str, Any] = {
                "models": {"extraction": current.llm_extraction, "reasoning": current.llm_reasoning,
                           "verifier": current.llm_verifier, "embedding": current.embedding},
                "keys": {name: bool(getattr(current, field)) for name, field in (
                    ("RA_GOOGLE_API_KEY", "google_api_key"), ("RA_OPENROUTER_API_KEY", "openrouter_api_key"),
                    ("RA_GROQ_API_KEY", "groq_api_key"), ("RA_S2_API_KEY", "s2_api_key"),
                    ("RA_OPENALEX_API_KEY", "openalex_api_key"))},
                "corpus": {"full_read": current.full_read, "abstract_only": current.abstract_only},
                "error": None,
            }
        except ValidationError as exc:
            setup = {"error": "; ".join(f"RA_{str(e['loc'][0]).upper() if e['loc'] else 'settings'}: {e['msg']}"
                                        for e in exc.errors(include_input=False, include_url=False))}
        return {
            "active": active.run_id if active else None,
            "running": [r for r in running if r["state"] == "running"],
            "stages": [{"name": s.name, "phase": s.phase, "summary": s.summary} for s in STAGES],
            "setup": setup,
        }

    @app.get("/api/runs")
    def list_runs(database: Db, limit: int = 100) -> list[dict[str, Any]]:
        return [_with_state(r) for r in views.run_rows(database, limit=min(limit, 500))]

    @app.post("/api/runs")
    def start(
        question: Annotated[str, Form()],
        refresh: Annotated[bool, Form()] = False,
        uploads: Annotated[list[UploadFile] | None, File()] = None,
    ) -> dict[str, Any]:
        question = " ".join(question.split())
        if not question:
            raise HTTPException(422, "Write a research question or idea.")
        if len(question) > MAX_QUESTION:
            raise HTTPException(422, f"Keep the question under {MAX_QUESTION} characters.")
        if jobs.active() is not None:
            raise HTTPException(409, f"Run {jobs.active().run_id} is still going; one run at a time.")
        run_id = new_id("run")
        paths = []
        for upload in uploads or []:
            if not upload.filename:
                continue
            name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(upload.filename).name)
            if not name.lower().endswith(".pdf"):
                raise HTTPException(422, f"{upload.filename}: only PDFs can be uploaded.")
            path = settings.data_dir / "uploads" / run_id / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(upload.file.read())
            paths.append(path.resolve())
        try:
            jobs.start(run_id, question, paths, refresh=refresh)
        except Busy as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"run_id": run_id}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str, database: Db) -> dict[str, Any]:
        run = run_or_404(database, run_id)
        log = settings.runs_dir / run_id / CONSOLE_LOG
        if run["state"] in ("failed", "interrupted") and log.exists():
            run["console_tail"] = log.read_text(encoding="utf-8", errors="replace").splitlines()[-CONSOLE_TAIL:]
        run["files"] = [name for name in FILES if (settings.runs_dir / run_id / name).exists()]
        return run

    @app.post("/api/runs/{run_id}/resume")
    def resume(run_id: str, database: Db) -> dict[str, Any]:
        run = run_or_404(database, run_id)
        if run["state"] == "completed":
            raise HTTPException(409, "This run has completed.")
        if run["state"] == "running":
            raise HTTPException(409, "This run is still going.")
        try:
            jobs.resume(run_id)
        except Busy as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"run_id": run_id}

    @app.post("/api/runs/{run_id}/stop")
    def stop(run_id: str, database: Db) -> dict[str, Any]:
        run_or_404(database, run_id)
        if not jobs.stop(run_id):
            raise HTTPException(409, "Only a run started from this window, and still going, can be stopped here.")
        return {"run_id": run_id}

    @app.get("/api/runs/{run_id}/progress")
    def progress(run_id: str, database: Db) -> dict[str, Any]:
        run = run_or_404(database, run_id)
        return {"run": run, **views.progress(database, run_id)}

    @app.get("/api/runs/{run_id}/report")
    def report(run_id: str, database: Db) -> Any:
        run_or_404(database, run_id)
        path = settings.runs_dir / run_id / "report.json"
        if path.exists():  # an older run's report lacks later phases' parts, or holds null for them
            return {"final": True, **_with_defaults(json.loads(path.read_text(encoding="utf-8")), _EMPTY_REPORT)}
        try:  # a run still going, or stopped: what it has found so far
            return {"final": False, **stored_report(database, run_id).model_dump(mode="json")}
        except Exception as exc:  # noqa: BLE001 - partial runs can lack any part
            raise HTTPException(404, f"No report yet: {type(exc).__name__}: {exc}") from exc

    @app.get("/api/runs/{run_id}/papers")
    def papers(run_id: str, database: Db) -> list[dict[str, Any]]:
        run_or_404(database, run_id)
        return views.papers(database, run_id)

    @app.get("/api/runs/{run_id}/cost")
    def cost(run_id: str, database: Db) -> dict[str, Any]:
        run_or_404(database, run_id)
        return run_cost(database, run_id).model_dump(mode="json")

    @app.get("/api/runs/{run_id}/graph")
    def graph(run_id: str, database: Db) -> dict[str, Any]:
        run_or_404(database, run_id)
        return views.graph(database, run_id)

    @app.get("/api/graph/{kind}/{item_id}")
    def graph_item(kind: str, item_id: str, database: Db) -> dict[str, Any]:
        if kind not in ("node", "edge"):
            raise HTTPException(404, "Ask for a node or an edge.")
        found = views.graph_item(database, kind, item_id)
        if found is None:
            raise HTTPException(404, f"No {kind} {item_id}.")
        return found

    @app.get("/api/runs/{run_id}/plan/elements/{element_id}")
    def element(run_id: str, element_id: str, database: Db) -> dict[str, Any]:
        run_or_404(database, run_id)
        found = views.element_chain(database, run_id, element_id)
        if found is None:
            raise HTTPException(404, f"No plan element {element_id}.")
        return found

    @app.get("/api/sentences/{sentence_id}")
    def sentence(sentence_id: str, database: Db, context: int = 2) -> dict[str, Any]:
        found = views.sentence(database, sentence_id, context=max(0, min(context, 10)))
        if found is None:
            raise HTTPException(404, f"No sentence {sentence_id}: it is no longer stored.")
        return found

    @app.get("/api/runs/{run_id}/files/{name}")
    def file(run_id: str, name: str, database: Db) -> FileResponse:
        run_or_404(database, run_id)
        path = settings.runs_dir / run_id / name
        if name not in FILES or not path.exists():
            raise HTTPException(404, f"No {name} for this run.")
        return FileResponse(path, media_type=FILES[name], filename=name if name != CONSOLE_LOG else None)

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str) -> Response:
        if path.startswith("api/"):
            raise HTTPException(404, "No such API route.")
        if not (STATIC / "index.html").exists():
            return HTMLResponse(_NOT_BUILT)
        target = (STATIC / path).resolve()
        if path and target.is_file() and target.is_relative_to(STATIC.resolve()):
            return FileResponse(target)
        return FileResponse(STATIC / "index.html")  # the app's own routes

    return app


def _host(header: str) -> str:
    """The host a Host header names, without its port."""
    if header.startswith("["):
        return header.split("]")[0] + "]"
    return header.rsplit(":", 1)[0]
