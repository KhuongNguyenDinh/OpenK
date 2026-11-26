"""FastAPI backend for OpenK Data Studio."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from openk.core.clustering import ALL_METHODS, DEFAULT_PARAMETERS
from openk.core.engine import (
    DatasetSession,
    load_dataframe_from_bytes,
    load_dataframe_from_path,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_SET_DIR = ROOT_DIR / "data_set"
BENCHMARK_HISTORY_PATH = ROOT_DIR / "benchmarks" / "history.jsonl"
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "web" / "templates"


class SampleLoadRequest(BaseModel):
    filename: str


class ClusterRequest(BaseModel):
    column: str
    method: str = "fingerprint"
    threshold: float | None = None
    min_cluster_size: int = 2


class MergeItem(BaseModel):
    from_values: list[str]
    to_value: str


class MergeRequest(BaseModel):
    column: str
    merges: list[MergeItem]


class TransformRequest(BaseModel):
    column: str | None = None
    operation: str
    params: dict[str, Any] = Field(default_factory=dict)


class RecipeApplyRequest(BaseModel):
    recipe: list[dict[str, Any]]


def _get_session(sessions: dict[str, DatasetSession], session_id: str) -> DatasetSession:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session


def _serialize_session(
    session_id: str,
    session: DatasetSession,
    offset: int = 0,
    limit: int = 100,
    search: str = "",
) -> dict[str, Any]:
    df = session.df
    if search.strip():
        q = search.strip().lower()
        mask = df.astype(str).apply(lambda col: col.str.lower().str.contains(q, regex=False)).any(axis=1)
        filtered = df[mask]
    else:
        filtered = df

    total_filtered = len(filtered)
    page_df = filtered.iloc[offset : offset + limit]
    # Replace NaN with None for valid JSON
    rows = json.loads(page_df.to_json(orient="records", date_format="iso"))
    profile = session.profile()

    return {
        "session_id": session_id,
        "name": session.name,
        "columns": [str(c) for c in df.columns],
        "total_rows": len(df),
        "filtered_rows": total_filtered,
        "offset": offset,
        "limit": limit,
        "rows": rows,
        "profile": profile.to_dict(),
        "can_undo": session.can_undo,
        "can_redo": session.can_redo,
        "history": session.get_recipe(),
    }


def create_app() -> FastAPI:
    app = FastAPI(
        title="OpenK Data Studio",
        description="Interactive Data Profiling, Fuzzy Clustering, and Reproducible Wrangling API",
        version="2.0.0",
    )
    sessions: dict[str, DatasetSession] = {}
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "methods": ALL_METHODS,
                "defaults": DEFAULT_PARAMETERS,
            },
        )

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": "2.0.0", "active_sessions": len(sessions)}

    @app.get("/api/samples")
    async def list_samples() -> dict[str, Any]:
        if not DATA_SET_DIR.exists():
            return {"samples": []}
        supported = {".csv", ".tsv", ".json", ".xlsx", ".xls", ".parquet"}
        files = sorted(
            [
                {
                    "filename": p.name,
                    "size_kb": round(p.stat().st_size / 1024, 1),
                }
                for p in DATA_SET_DIR.iterdir()
                if p.is_file() and p.suffix.lower() in supported
            ],
            key=lambda x: x["filename"],
        )
        return {"samples": files}

    @app.post("/api/sessions/sample")
    async def load_sample(req: SampleLoadRequest) -> dict[str, Any]:
        target = (DATA_SET_DIR / req.filename).resolve()
        if not target.is_file() or target.parent != DATA_SET_DIR.resolve():
            raise HTTPException(status_code=404, detail=f"Sample '{req.filename}' not found")
        df = load_dataframe_from_path(target)
        sid = uuid.uuid4().hex[:12]
        sessions[sid] = DatasetSession(df, name=req.filename)
        return _serialize_session(sid, sessions[sid])

    @app.post("/api/sessions/upload")
    async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing filename")
        content = await file.read()
        try:
            df = load_dataframe_from_bytes(content, file.filename)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse file: {exc}") from exc
        sid = uuid.uuid4().hex[:12]
        sessions[sid] = DatasetSession(df, name=file.filename)
        return _serialize_session(sid, sessions[sid])

    @app.get("/api/sessions/{session_id}")
    async def get_session_state(
        session_id: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=500),
        search: str = Query(""),
    ) -> dict[str, Any]:
        session = _get_session(sessions, session_id)
        return _serialize_session(session_id, session, offset=offset, limit=limit, search=search)

    @app.get("/api/sessions/{session_id}/recommend")
    async def recommend_for_column(session_id: str, column: str = Query(...)) -> dict[str, Any]:
        session = _get_session(sessions, session_id)
        try:
            return session.recommend(column)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/cluster")
    async def run_cluster(session_id: str, req: ClusterRequest) -> dict[str, Any]:
        session = _get_session(sessions, session_id)
        try:
            clusters = session.cluster(
                column=req.column,
                method=req.method,
                threshold=req.threshold,
                min_cluster_size=req.min_cluster_size,
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "column": req.column,
            "method": req.method,
            "cluster_count": len(clusters),
            "clusters": clusters,
        }

    @app.post("/api/sessions/{session_id}/merge")
    async def apply_merge(session_id: str, req: MergeRequest) -> dict[str, Any]:
        session = _get_session(sessions, session_id)
        try:
            record = session.apply_cluster_merges(
                column=req.column,
                merges=[m.model_dump() for m in req.merges],
            )
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"operation": record.to_dict(), "state": _serialize_session(session_id, session)}

    @app.post("/api/sessions/{session_id}/transform")
    async def apply_transform(session_id: str, req: TransformRequest) -> dict[str, Any]:
        session = _get_session(sessions, session_id)
        try:
            record = session.transform_column(req.column, req.operation, **req.params)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"operation": record.to_dict(), "state": _serialize_session(session_id, session)}

    @app.get("/api/sessions/{session_id}/facet")
    async def get_facet(
        session_id: str, column: str = Query(...), limit: int = Query(50, ge=1, le=200)
    ) -> dict[str, Any]:
        session = _get_session(sessions, session_id)
        try:
            return {"column": column, "items": session.text_facet(column, limit=limit)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/undo")
    async def undo_op(session_id: str) -> dict[str, Any]:
        session = _get_session(sessions, session_id)
        undone = session.undo()
        return {"undone": undone, "state": _serialize_session(session_id, session)}

    @app.post("/api/sessions/{session_id}/redo")
    async def redo_op(session_id: str) -> dict[str, Any]:
        session = _get_session(sessions, session_id)
        redone = session.redo()
        return {"redone": redone, "state": _serialize_session(session_id, session)}

    @app.get("/api/sessions/{session_id}/recipe")
    async def get_recipe(session_id: str) -> dict[str, Any]:
        session = _get_session(sessions, session_id)
        return {"name": session.name, "recipe": session.get_recipe()}

    @app.post("/api/sessions/{session_id}/recipe")
    async def apply_recipe(session_id: str, req: RecipeApplyRequest) -> dict[str, Any]:
        session = _get_session(sessions, session_id)
        try:
            applied = session.apply_recipe(req.recipe)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"applied": applied, "state": _serialize_session(session_id, session)}

    @app.get("/api/sessions/{session_id}/export")
    async def export_dataset(session_id: str, fmt: str = Query("csv")) -> Response:
        session = _get_session(sessions, session_id)
        try:
            data, media_type, ext = session.export_bytes(fmt)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        stem = Path(session.name).stem or "openk_export"
        return Response(
            content=data,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{stem}_cleaned.{ext}"'},
        )

    @app.get("/api/benchmarks")
    async def get_benchmarks() -> dict[str, Any]:
        if not BENCHMARK_HISTORY_PATH.exists():
            return {"runs": []}
        runs = []
        for line in BENCHMARK_HISTORY_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                runs.append(json.loads(line))
        return {"runs": runs[-20:]}

    return app


app = create_app()
