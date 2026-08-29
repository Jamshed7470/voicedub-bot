"""WebSocket-уведомления студии.

Через него идут прогресс рендера, готовность превью и смена стадии.
Соединение может оборваться (вкладку свернули, сеть моргнула), поэтому
события не копятся: клиент при переподключении заново читает проект,
а WebSocket служит только для «что происходит прямо сейчас».
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

log = logging.getLogger(__name__)


class Hub:
    """Подписчики по job_id."""

    def __init__(self):
        self._clients: dict[str, set] = defaultdict(set)
        self._lock = asyncio.Lock()
        self._last: dict[str, dict] = {}

    async def join(self, job_id: str, socket) -> None:
        async with self._lock:
            self._clients[job_id].add(socket)
        last = self._last.get(job_id)
        if last:
            # только что подключившийся клиент должен сразу увидеть текущее
            # состояние, а не ждать следующего события
            try:
                await socket.send_json(last)
            except Exception:  # noqa: BLE001
                pass

    async def leave(self, job_id: str, socket) -> None:
        async with self._lock:
            self._clients[job_id].discard(socket)
            if not self._clients[job_id]:
                self._clients.pop(job_id, None)

    async def publish(self, job_id: str, event: dict) -> None:
        self._last[job_id] = event
        async with self._lock:
            targets = list(self._clients.get(job_id, ()))
        dead = []
        for socket in targets:
            try:
                await socket.send_json(event)
            except Exception:  # noqa: BLE001 — отвалившийся клиент не наша беда
                dead.append(socket)
        for socket in dead:
            await self.leave(job_id, socket)

    def publish_threadsafe(self, loop, job_id: str, event: dict) -> None:
        """Отправка из рабочего потока пайплайна."""
        if loop is None or loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self.publish(job_id, event), loop)

    # ---------- типовые события ----------

    async def progress(self, job_id: str, stage: int, label: str, pct: int) -> None:
        await self.publish(job_id, {"type": "progress", "stage": stage,
                                    "label": label, "pct": pct})

    async def stage_changed(self, job_id: str, stage: str) -> None:
        await self.publish(job_id, {"type": "stage_changed", "stage": stage})

    async def preview_ready(self, job_id: str, kind: str, ref: str,
                            url: str) -> None:
        await self.publish(job_id, {"type": "preview_ready", "kind": kind,
                                    "ref": ref, "url": url})

    async def preview_failed(self, job_id: str, ref: str, error: str) -> None:
        await self.publish(job_id, {"type": "preview_failed", "ref": ref,
                                    "error": error})

    async def qc_updated(self, job_id: str, warnings: int) -> None:
        await self.publish(job_id, {"type": "qc_updated", "warnings": warnings})


hub = Hub()
