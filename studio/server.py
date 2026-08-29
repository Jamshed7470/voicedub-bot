"""FastAPI-сервер студии.

Запуск отдельным процессом:
    python -m studio

Либо вместе с ботом — переменная STUDIO_EMBEDDED=true (uvicorn поднимается
в том же event loop, что и aiogram).

Отдаёт собранный фронтенд из studio/static/ и API из studio/api.py.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core.config import load_config
from studio import auth
from studio.api import router
from studio.ws import hub

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    cfg = load_config()
    app = FastAPI(title="VoiceDub Studio", docs_url=None, redoc_url=None)

    # CORS только для собственного адреса: студия и API живут на одном
    # origin, чужим страницам обращаться к проектам незачем
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[cfg.studio_url],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
        allow_headers=["Authorization", "If-Match", "Content-Type"],
        expose_headers=["ETag"],
    )

    app.include_router(router)

    @app.get("/health")
    async def health():
        return {"ok": True, "studio": cfg.studio_on,
                "static": STATIC_DIR.exists()}

    @app.websocket("/api/projects/{job_id}/events")
    async def events(socket: WebSocket, job_id: str, t: str = Query(...)):
        try:
            auth.check(t, job_id, cfg.studio_secret, cfg.studio_link_ttl_h)
        except auth.AuthError:
            await socket.close(code=4401)
            return
        await socket.accept()
        await hub.join(job_id, socket)
        try:
            while True:
                # клиент шлёт ping'и; нам важно лишь держать соединение
                await socket.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            pass
        finally:
            await hub.leave(job_id, socket)

    if STATIC_DIR.exists():
        app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")),
                  name="assets")

    @app.get("/studio/{job_id}", response_class=HTMLResponse)
    async def studio_page(job_id: str, request: Request, t: str = Query(None)):
        index = STATIC_DIR / "index.html"
        if not index.exists():
            return HTMLResponse(_no_frontend_page(), status_code=200)
        return HTMLResponse(index.read_text(encoding="utf-8"))

    @app.get("/", response_class=HTMLResponse)
    async def root():
        return HTMLResponse(
            "<h1>VoiceDub Studio</h1><p>Откройте проект по ссылке из бота "
            "(команда /review).</p>")

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        detail = exc.detail
        body = detail if isinstance(detail, dict) else {"error": str(detail)}
        return JSONResponse(body, status_code=exc.status_code)

    return app


def _no_frontend_page() -> str:
    return """<!doctype html><meta charset="utf-8">
<title>VoiceDub Studio</title>
<style>body{font:16px/1.6 system-ui;max-width:44rem;margin:4rem auto;padding:0 1rem;
color:#1a1a1a}code{background:#f2f3f5;padding:.15em .4em;border-radius:4px}</style>
<h1>Интерфейс студии не собран</h1>
<p>API работает, но веб-страница отсутствует. Соберите фронтенд:</p>
<pre><code>cd studio/frontend
npm install
npm run build</code></pre>
<p>Сборка попадёт в <code>studio/static/</code>. Обычно она уже лежит в
репозитории — если её нет, значит проект склонирован без неё.</p>"""


def run() -> None:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    if not cfg.studio_secret:
        log.warning("STUDIO_SECRET не задан — ссылки на проекты выдаваться "
                    "не будут. Сгенерируйте секрет, см. README.")
    log.info("Студия: http://%s:%s (внешний адрес %s)",
             cfg.studio_host, cfg.studio_port, cfg.studio_url)
    uvicorn.run(create_app(), host=cfg.studio_host, port=cfg.studio_port,
                log_level="info")


app = None   # создаётся лениво, чтобы импорт модуля не читал конфиг
