"""Прогресс задачи: одно сообщение, редактируемое по этапам с процентами."""
from __future__ import annotations

import logging
import time

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter

from bot import texts

log = logging.getLogger(__name__)

TOTAL_STAGES = 10
MIN_EDIT_INTERVAL = 2.5  # секунды между редактированиями (лимиты Telegram)


class ProgressReporter:
    def __init__(self, bot: Bot, chat_id: int):
        self.bot = bot
        self.chat_id = chat_id
        self.message_id: int | None = None
        self._last_edit = 0.0
        self._last_text = ""

    async def start(self, text: str) -> None:
        msg = await self.bot.send_message(self.chat_id, text)
        self.message_id = msg.message_id
        self._last_text = text

    async def set_text(self, text: str, force: bool = False) -> None:
        if text == self._last_text:
            return
        now = time.monotonic()
        if not force and now - self._last_edit < MIN_EDIT_INTERVAL:
            return
        if self.message_id is None:
            await self.start(text)
            return
        try:
            await self.bot.edit_message_text(text, chat_id=self.chat_id,
                                             message_id=self.message_id)
            self._last_edit = now
            self._last_text = text
        except TelegramRetryAfter as e:
            log.debug("Flood limit на редактирование: %s c", e.retry_after)
        except TelegramBadRequest as e:
            if "message is not modified" not in str(e):
                log.warning("Не удалось отредактировать прогресс: %s", e)

    async def stage(self, stage: int, label: str, pct: int = 0) -> None:
        await self.set_text(texts.PROGRESS.format(
            stage=stage, total=TOTAL_STAGES, label=label, pct=max(0, min(100, pct))))

    async def queue_position(self, n: int) -> None:
        await self.set_text(texts.PROGRESS_QUEUE.format(n=n), force=True)

    async def finish(self, text: str) -> None:
        await self.set_text(text, force=True)
