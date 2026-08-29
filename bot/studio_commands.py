"""Команды бота, относящиеся к студии.

Отдельный роутер, чтобы основной handlers.py не разрастался и чтобы при
STUDIO_ENABLED=false его можно было просто не подключать (INV-6).
"""
from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, Message

from bot import db, review
from core.config import JOBS_DIR, load_config
from project import store
from project.schema import Stage

log = logging.getLogger(__name__)
router = Router()


def _last_project(user_id: int):
    """Последняя задача пользователя, у которой есть project.json."""
    if not JOBS_DIR.exists():
        return None
    candidates = []
    for entry in JOBS_DIR.iterdir():
        if not (entry / "project.json").exists():
            continue
        proj = store.load_or_none(entry.name)
        if proj and proj.owner_telegram_id == user_id:
            candidates.append((entry.stat().st_mtime, proj))
    if not candidates:
        return None
    return max(candidates, key=lambda t: t[0])[1]


@router.message(Command("review"))
async def cmd_review(message: Message) -> None:
    """Выдаёт (или перевыпускает) ссылку на студию для последней задачи."""
    cfg = load_config()
    if not cfg.studio_on:
        await message.answer(
            "Студия проверки выключена. Чтобы включить, задайте в файле .env:\n"
            "<code>STUDIO_ENABLED=true</code>\nи запустите её командой "
            "<code>python -m studio</code>", parse_mode="HTML")
        return

    proj = _last_project(message.from_user.id)
    if proj is None:
        await message.answer("Пока нечего проверять — пришлите видео или ссылку.")
        return

    link = review.make_link(proj.job_id, message.from_user.id, cfg)
    if not link:
        await message.answer(
            "Не задан <code>STUDIO_SECRET</code> — ссылку выдать нельзя.\n"
            "Сгенерируйте секрет:\n"
            '<code>python -c "import secrets; print(secrets.token_urlsafe(48))"</code>\n'
            "и впишите его в .env.", parse_mode="HTML")
        return

    await message.answer(
        review.summary_text(proj), parse_mode="HTML",
        reply_markup=review.review_keyboard(proj.job_id, link))


@router.message(Command("approve"))
async def cmd_approve(message: Message) -> None:
    """Утверждает последнюю задачу без открытия студии."""
    proj = _last_project(message.from_user.id)
    if proj is None:
        await message.answer("Нет задачи, ожидающей утверждения.")
        return
    if proj.stage not in (Stage.REVIEW, Stage.TRANSLATED, Stage.PROFILED):
        await message.answer(
            f"Задача на стадии «{proj.stage.value}» — утверждать нечего.")
        return

    review.mark_approved(proj.job_id)
    await review.enqueue_approved(proj.job_id)
    await message.answer("✅ Утверждено. Начинаю озвучку.")


@router.message(Command("speakers"))
async def cmd_speakers(message: Message, command: CommandObject) -> None:
    """/speakers N — задать число голосов; /speakers auto — сбросить."""
    arg = (command.args or "").strip().lower()
    if not arg:
        current = db.get_user(message.from_user.id).get("speakers", "auto")
        await message.answer(
            f"Сейчас: <b>{current}</b>.\n"
            "Задать: <code>/speakers 4</code> (от 1 до 20) или "
            "<code>/speakers auto</code>.", parse_mode="HTML")
        return

    if arg in ("auto", "авто"):
        db.set_speakers(message.from_user.id, "auto")
        await message.answer("Число спикеров определяется автоматически.")
        return

    if not arg.isdigit() or not (1 <= int(arg) <= 20):
        await message.answer("Число спикеров — от 1 до 20, либо «auto».")
        return

    db.set_speakers(message.from_user.id, arg)
    await message.answer(
        f"Буду искать <b>{arg}</b> спикеров.\n"
        "Для текущей задачи перезапустите разбор кнопкой «Число спикеров» "
        "в студии — там это делается без потери правок.", parse_mode="HTML")


@router.message(Command("mode"))
async def cmd_mode(message: Message, command: CommandObject) -> None:
    """/mode clone|preset|auto — режим голосов по умолчанию."""
    arg = (command.args or "").strip().lower()
    modes = {
        "clone": "клон оригинального голоса каждого спикера",
        "preset": "голоса из банка",
        "auto": "клон, если чистой речи хватает; иначе голос из банка",
    }
    if arg not in modes:
        current = db.get_user(message.from_user.id).get("voice", "auto")
        await message.answer(
            f"Сейчас: <b>{current}</b>.\n\n"
            "<code>/mode clone</code> — " + modes["clone"] + "\n"
            "<code>/mode preset</code> — " + modes["preset"] + "\n"
            "<code>/mode auto</code> — " + modes["auto"], parse_mode="HTML")
        return

    db.set_voice_mode(message.from_user.id, arg)
    await message.answer(f"Режим голосов: <b>{arg}</b> — {modes[arg]}.",
                         parse_mode="HTML")


@router.message(Command("voices"))
async def cmd_voices(message: Message) -> None:
    """Список банка голосов."""
    from voices.bank import get_bank

    voices = get_bank().all()
    if not voices:
        await message.answer(
            "Банк голосов пуст. Соберите его одной командой:\n"
            "<code>python -m scripts.build_voice_bank --from-xtts</code>\n\n"
            "Или добавьте свои записи (30–60 с чистой речи) в папку "
            "<code>voice_db/</code> и выполните:\n"
            "<code>python -m scripts.build_voice_bank --from-dir voice_db</code>",
            parse_mode="HTML")
        return

    males = [v for v in voices if v.gender == "male"]
    females = [v for v in voices if v.gender == "female"]
    lines = [f"🎙 <b>Банк голосов: {len(voices)}</b>", ""]
    for title, group in (("Мужские", males), ("Женские", females)):
        if not group:
            continue
        lines.append(f"<b>{title} ({len(group)})</b>")
        for v in group[:20]:
            age = " · детский" if v.is_child else ""
            lines.append(f"  • {v.display_name}{age}")
        if len(group) > 20:
            lines.append(f"  …и ещё {len(group) - 20}")
        lines.append("")
    lines.append("Прослушать и назначить голоса можно в студии: /review")
    await message.answer("\n".join(lines), parse_mode="HTML")


@router.callback_query(F.data.startswith("approve:"))
async def cb_approve(callback: CallbackQuery) -> None:
    job_id = callback.data.split(":", 1)[1]
    proj = store.load_or_none(job_id)
    if proj is None:
        await callback.answer("Проект не найден", show_alert=True)
        return
    if proj.owner_telegram_id != callback.from_user.id:
        await callback.answer("Это чужая задача", show_alert=True)
        return

    review.mark_approved(job_id)
    await review.enqueue_approved(job_id)
    await callback.answer("Утверждено")
    try:
        await callback.message.edit_text(
            "✅ <b>Утверждено как есть.</b> Начинаю озвучку.", parse_mode="HTML")
    except Exception:  # noqa: BLE001 — сообщение могли удалить
        pass


@router.callback_query(F.data.startswith("spkhint:"))
async def cb_speaker_hint(callback: CallbackQuery) -> None:
    await callback.answer()
    await callback.message.answer(
        "Задайте число голосов командой <code>/speakers N</code> "
        "(от 1 до 20), затем в студии нажмите «Перезапустить разбор».\n\n"
        "Ваши правки при этом сохранятся: спикеры узнаются по голосу.",
        parse_mode="HTML")
