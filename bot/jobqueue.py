"""Очередь задач: строго одна GPU-задача одновременно,
одна активная задача на пользователя."""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field

from aiogram import Bot
from aiogram.types import FSInputFile

from bot import texts
from bot.progress import ProgressReporter
from core.errors import JobCancelled, UserError
from core.pipeline import PipelineHooks, cleanup_job, run_job

log = logging.getLogger(__name__)


@dataclass
class Job:
    user_id: int
    chat_id: int
    kind: str                    # "tg" | "url"
    payload: object              # {"file_id","file_size"} или url
    target_lang: str
    settings: dict
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "queued"       # queued | running | done | failed | cancelled
    stage_label: str = ""
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    reporter: ProgressReporter | None = None
    same_lang_future: asyncio.Future | None = None


class JobQueue:
    def __init__(self, bot: Bot, cfg):
        self.bot = bot
        self.cfg = cfg
        self.queue: list[Job] = []
        self.current: Job | None = None
        self.jobs_by_id: dict[str, Job] = {}
        self._wakeup = asyncio.Event()
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------ API

    def start(self) -> None:
        self._task = asyncio.create_task(self._worker(), name="jobqueue-worker")

    def user_job(self, user_id: int) -> Job | None:
        if self.current and self.current.user_id == user_id:
            return self.current
        return next((j for j in self.queue if j.user_id == user_id), None)

    async def enqueue(self, job: Job) -> int:
        """Ставит задачу в очередь. Возвращает позицию (0 — сразу в работу)."""
        if self.user_job(job.user_id):
            raise UserError(texts.ALREADY_HAS_JOB)
        self.jobs_by_id[job.id] = job
        self.queue.append(job)
        position = len(self.queue) - (0 if self.current else 1)
        self._wakeup.set()
        return max(0, position)

    async def cancel(self, user_id: int) -> bool:
        job = self.user_job(user_id)
        if not job:
            return False
        job.cancel_event.set()
        if job.same_lang_future and not job.same_lang_future.done():
            job.same_lang_future.set_result(False)
        if job in self.queue:
            self.queue.remove(job)
            job.status = "cancelled"
            if job.reporter:
                await job.reporter.finish(texts.CANCELLED)
        return True

    def status_text(self, user_id: int) -> str | None:
        job = self.user_job(user_id)
        if not job:
            return None
        if job is self.current:
            return f"🎬 Обрабатываю: {job.stage_label or 'подготовка…'}"
        pos = self.queue.index(job) + (1 if self.current else 0)
        return texts.QUEUED.format(n=pos + (0 if self.current else 1))

    # --------------------------------------------------------------- worker

    async def _worker(self) -> None:
        while True:
            while not self.queue:
                self._wakeup.clear()
                await self._wakeup.wait()
            job = self.queue.pop(0)
            self.current = job
            try:
                await self._run_one(job)
            except Exception:  # noqa: BLE001 — воркер не должен умирать
                log.exception("Необработанная ошибка задачи %s", job.id)
            finally:
                self.current = None
                self.jobs_by_id.pop(job.id, None)
                # обновить позиции ожидающих
                for n, waiting in enumerate(self.queue, 1):
                    if waiting.reporter:
                        try:
                            await waiting.reporter.queue_position(n)
                        except Exception:  # noqa: BLE001
                            pass

    async def _run_one(self, job: Job) -> None:
        job.status = "running"
        reporter = job.reporter

        async def report(stage: int, label: str, pct: int) -> None:
            job.stage_label = label
            if reporter:
                await reporter.stage(stage, label, pct)

        async def confirm_same_lang(src_lang: str) -> bool:
            from bot.keyboards import same_lang_keyboard
            from core.config import TTS_LANGUAGES
            loop = asyncio.get_running_loop()
            job.same_lang_future = loop.create_future()
            lang_name = TTS_LANGUAGES.get(job.target_lang, job.target_lang)
            await self.bot.send_message(
                job.chat_id,
                texts.SAME_LANG_WARN.format(lang=lang_name),
                reply_markup=same_lang_keyboard(job.id),
            )
            try:
                return await asyncio.wait_for(job.same_lang_future, timeout=300)
            except asyncio.TimeoutError:
                return True  # нет ответа 5 минут — продолжаем

        hooks = PipelineHooks(report=report, confirm_same_lang=confirm_same_lang,
                              cancel_event=job.cancel_event)
        try:
            result = await run_job(job, self.bot, hooks, self.cfg)
            job.status = "done"
            if reporter:
                await reporter.finish(result.summary)
            await self._send_result(job, result)
        except JobCancelled:
            job.status = "cancelled"
            if reporter:
                await reporter.finish(texts.CANCELLED)
        except UserError as e:
            job.status = "failed"
            log.warning("Задача %s: %s", job.id, e.message_ru)
            if reporter:
                await reporter.finish(f"😔 {e.message_ru}")
        except Exception:  # noqa: BLE001
            job.status = "failed"
            log.exception("Задача %s упала", job.id)
            if reporter:
                await reporter.finish(texts.ERR_GENERIC)
        finally:
            cleanup_job(job.id)

    async def _send_result(self, job: Job, result) -> None:
        from core.config import TTS_LANGUAGES
        caption = texts.DONE_CAPTION.format(
            lang=TTS_LANGUAGES.get(result.tgt_lang, result.tgt_lang))
        file = FSInputFile(str(result.output_path))
        if result.kind == "video":
            await self.bot.send_video(job.chat_id, file, caption=caption,
                                      supports_streaming=True)
        else:
            await self.bot.send_audio(job.chat_id, file, caption=caption)
        await self.bot.send_document(job.chat_id, FSInputFile(str(result.srt_path)),
                                     caption=texts.SRT_CAPTION)
