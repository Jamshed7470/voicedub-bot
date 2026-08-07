"""Хендлеры бота (aiogram 3)."""
from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, Message

from bot import db, texts
from bot.jobqueue import Job, JobQueue
from bot.keyboards import lang_keyboard, settings_keyboard
from bot.progress import ProgressReporter
from core.config import TTS_LANGUAGES
from core.downloader import find_url
from core.errors import UserError

log = logging.getLogger(__name__)
router = Router()


# ------------------------------------------------------------------ команды

@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(texts.START)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(texts.HELP)


@router.message(Command("lang"))
async def cmd_lang(message: Message) -> None:
    await message.answer(texts.CHOOSE_LANG, reply_markup=lang_keyboard())


@router.message(Command("settings"))
async def cmd_settings(message: Message) -> None:
    user = db.get_user(message.from_user.id)
    await message.answer(
        texts.SETTINGS_TITLE,
        reply_markup=settings_keyboard(user["keep_background"], user["keep_original"]),
    )


@router.message(Command("status"))
async def cmd_status(message: Message, jobqueue: JobQueue) -> None:
    text = jobqueue.status_text(message.from_user.id)
    await message.answer(text or texts.NO_ACTIVE_JOB)


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, jobqueue: JobQueue) -> None:
    if await jobqueue.cancel(message.from_user.id):
        await message.answer(texts.CANCEL_OK)
    else:
        await message.answer(texts.NO_ACTIVE_JOB)


# ------------------------------------------------------------------ колбэки

@router.callback_query(F.data.startswith("lang:"))
async def cb_lang(query: CallbackQuery) -> None:
    code = query.data.split(":", 1)[1]
    if code not in TTS_LANGUAGES:
        await query.answer("Неподдерживаемый язык")
        return
    db.set_lang(query.from_user.id, code)
    await query.answer()
    await query.message.edit_text(texts.LANG_SAVED.format(lang=TTS_LANGUAGES[code]))


@router.callback_query(F.data == "set:bg")
async def cb_toggle_bg(query: CallbackQuery) -> None:
    db.toggle_background(query.from_user.id)
    user = db.get_user(query.from_user.id)
    await query.answer()
    await query.message.edit_reply_markup(
        reply_markup=settings_keyboard(user["keep_background"], user["keep_original"]))


@router.callback_query(F.data == "set:orig")
async def cb_toggle_orig(query: CallbackQuery) -> None:
    db.toggle_original(query.from_user.id)
    user = db.get_user(query.from_user.id)
    await query.answer()
    await query.message.edit_reply_markup(
        reply_markup=settings_keyboard(user["keep_background"], user["keep_original"]))


@router.callback_query(F.data.startswith("samelang:"))
async def cb_same_lang(query: CallbackQuery, jobqueue: JobQueue) -> None:
    _, answer, job_id = query.data.split(":", 2)
    job = jobqueue.jobs_by_id.get(job_id)
    await query.answer()
    try:
        await query.message.delete()
    except Exception:  # noqa: BLE001
        pass
    if job and job.same_lang_future and not job.same_lang_future.done():
        job.same_lang_future.set_result(answer == "yes")


# --------------------------------------------------------------- медиа/ссылки

def _extract_file(message: Message) -> tuple[str, int | None] | None:
    """Возвращает (file_id, file_size) для любого поддерживаемого типа вложения."""
    for attr in ("video", "audio", "voice", "video_note", "document"):
        obj = getattr(message, attr, None)
        if obj:
            return obj.file_id, getattr(obj, "file_size", None)
    return None


async def _enqueue(message: Message, jobqueue: JobQueue,
                   kind: str, payload) -> None:
    user = db.get_user(message.from_user.id)
    if not user["lang"]:
        await message.answer(texts.LANG_FIRST, reply_markup=lang_keyboard())
        return

    job = Job(
        user_id=message.from_user.id,
        chat_id=message.chat.id,
        kind=kind,
        payload=payload,
        target_lang=user["lang"],
        settings={
            "keep_background": user["keep_background"],
            "keep_original_track": user["keep_original"],
        },
    )
    reporter = ProgressReporter(message.bot, message.chat.id)
    job.reporter = reporter
    if jobqueue.user_job(message.from_user.id):
        await message.answer(texts.ALREADY_HAS_JOB)
        return
    # сообщение прогресса создаётся ДО постановки в очередь,
    # чтобы воркер не начал редактировать несуществующее сообщение
    await reporter.start("🎬 Принял! Готовлюсь…")
    try:
        position = await jobqueue.enqueue(job)
    except UserError as e:
        await reporter.finish(e.message_ru)
        return
    if position > 0:
        await reporter.queue_position(position)


@router.message(F.video | F.audio | F.voice | F.video_note | F.document)
async def on_media(message: Message, jobqueue: JobQueue) -> None:
    file = _extract_file(message)
    if not file:
        await message.answer(texts.UNKNOWN_INPUT)
        return
    file_id, file_size = file
    await _enqueue(message, jobqueue, "tg",
                   {"file_id": file_id, "file_size": file_size})


@router.message(F.text)
async def on_text(message: Message, jobqueue: JobQueue) -> None:
    url = find_url(message.text or "")
    if url:
        await _enqueue(message, jobqueue, "url", url)
    else:
        await message.answer(texts.UNKNOWN_INPUT)
