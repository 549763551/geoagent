"""GeoAgent Web 服务：把智能体引擎的事件流以 SSE 实时推送到前端。

接口一览：
- POST /api/task          提交任务，返回 SSE 事件流（理解 / 召回 / 规划 / 执行 / 交付）
- GET  /api/session/{id}  查询会话状态（对话、工具轨迹、产物）
- POST /api/session/reset 重置会话（开启新会话）
- GET  /api/meta          运行模式、数据目录、工具清单等元信息
- GET  /api/memory        长期记忆（跨会话经验）
- GET  /files/...         分析产物静态资源（图 / 表 / 栅格 / 报告）

启动：python -m app.server
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import threading
import zipfile
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import engine
from .catalog import DERIVED_INDICES, catalog
from .config import settings
from .memory import long_term_memory, session_store
from .tools import registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("geoagent.server")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
HEARTBEAT_SECONDS = 15.0

app = FastAPI(title="GeoAgent · 遥感栅格分析智能体", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================================
# 请求模型
# ======================================================================
class TaskRequest(BaseModel):
    query: str = Field(..., description="自然语言任务描述")
    session_id: str | None = Field(None, description="会话 ID，缺省则新建")


class ResetRequest(BaseModel):
    session_id: str | None = None


# ======================================================================
# SSE 工具
# ======================================================================
def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


# ======================================================================
# 任务接口（SSE）
# ======================================================================
@app.post("/api/task")
async def api_task(payload: TaskRequest) -> StreamingResponse:
    query = (payload.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="任务描述不能为空")

    session = session_store.get_or_create(payload.session_id)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    done_marker = {"type": "__done__"}

    def emit(event: dict[str, Any]) -> None:
        """引擎在工作线程中回调，转投到事件循环的队列。"""
        loop.call_soon_threadsafe(queue.put_nowait, dict(event))

    def worker() -> None:
        try:
            engine.run(query, session, emit)
        except Exception as exc:  # pragma: no cover - 引擎内部已兜底
            logger.exception("任务执行异常")
            emit({"type": "error", "text": f"任务执行失败：{exc}"})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, done_marker)

    threading.Thread(target=worker, name=f"geoagent-{session.id}", daemon=True).start()

    async def event_stream() -> AsyncIterator[str]:
        yield _sse({"type": "session", "session_id": session.id, "mode": engine.mode})
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"          # 心跳，防止代理断开长连接
                continue
            if event.get("type") == "__done__":
                break
            yield _sse(event)
        yield _sse({"type": "done", "session_id": session.id})

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=SSE_HEADERS)


# ======================================================================
# 会话接口
# ======================================================================
@app.get("/api/session/{session_id}")
async def api_session(session_id: str) -> JSONResponse:
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return JSONResponse({
        **session.digest(),
        "messages": session.messages,
        "understanding": session.understanding,
        "plan": session.plan,
        "steps": [r.to_dict() for r in session.records],
        "artifacts": [a.to_dict() for a in session.artifacts],
    })


@app.get("/api/session/{session_id}/artifacts.zip")
async def api_artifacts_zip(session_id: str) -> Response:
    """把一次会话产出的全部文件打包下载。"""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    files = [a for a in session.artifacts if Path(a.path).exists()]
    if not files:
        raise HTTPException(status_code=404, detail="本次会话暂无可导出的产物")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for art in files:
            src = Path(art.path)
            zf.write(src, arcname=f"{art.type}/{src.name}")
    filename = f"geoagent-{session_id[:8]}-artifacts.zip"
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/session/reset")
async def api_session_reset(payload: ResetRequest) -> JSONResponse:
    if payload.session_id:
        session_store.reset(payload.session_id)
    session = session_store.get_or_create(None)
    return JSONResponse({"session_id": session.id})


# ======================================================================
# 元信息接口
# ======================================================================
@app.get("/api/meta")
async def api_meta() -> JSONResponse:
    return JSONResponse({
        "agent": "GeoAgent",
        "mode": engine.mode,
        "llm_model": settings.llm_model if settings.llm_enabled else "",
        "data_root": str(settings.data_root),
        "datasets": [
            {"key": e.key, "title": e.title, "variable": e.variable, "unit": e.unit,
             "kind": e.kind, "tags": e.tags, "description": e.description,
             "years": e.available_years, "scenarios": sorted(e.scenario_files)}
            for e in catalog.all()
        ],
        "tools": [{"name": s.name, "category": s.category, "description": s.description}
                  for s in registry.specs()],
        "derived_indices": [{"name": name, "description": desc}
                            for name, desc in DERIVED_INDICES.items()],
        "stats": {"datasets": len(catalog.all()), "tools": len(registry.names())},
    })


@app.get("/api/memory")
async def api_memory(limit: int = 20) -> JSONResponse:
    items = long_term_memory.all()[-limit:]
    items.reverse()
    return JSONResponse({"count": len(items), "items": items})


@app.get("/api/health")
async def api_health() -> JSONResponse:
    return JSONResponse({"ok": True, "mode": engine.mode})


# ======================================================================
# 静态资源
# ======================================================================
settings.output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(settings.output_dir)), name="files")

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:  # 前端尚未构建时的占位页
    @app.get("/")
    async def index() -> JSONResponse:
        return JSONResponse({
            "agent": "GeoAgent",
            "message": "前端目录 frontend/ 不存在，接口已就绪。",
            "docs": "/docs",
        })


def main() -> None:
    import uvicorn

    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
