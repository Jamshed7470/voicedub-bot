"""Точка проверки: пауза между переводом и озвучкой.

Пайплайн доходит до готового перевода и останавливается. Пользователь
получает сообщение с кнопками и ссылку на студию, сверяет спикеров с
видео, правит — и только потом нажимает «Утвердить». Рендер полутора тысяч
реплик идёт часами, поэтому ошибку дешевле поймать здесь.

Работает только при STUDIO_ENABLED=true. Иначе задача идёт как раньше
(INV-6) — но уже с зафиксированными профилями голоса.
"""
from __future__ import annotations

import asyncio
import re
import logging

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from project import store
from project.schema import Project, Stage
from studio import auth

log = logging.getLogger(__name__)

# ожидающие утверждения задачи: job_id → событие, которого ждёт пайплайн
_waiting: dict[str, asyncio.Event] = {}
# как часто перечитывать отметку об утверждении, если студия в другом процессе
POLL_SEC = 5.0


def review_keyboard(job_id: str, link: str | None) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    if link:
        rows.append([InlineKeyboardButton(text="🎬 Открыть студию", url=link)])
    rows.append([
        InlineKeyboardButton(text="✅ Утвердить как есть",
                             callback_data=f"approve:{job_id}"),
        InlineKeyboardButton(text="👥 Число спикеров",
                             callback_data=f"spkhint:{job_id}"),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def make_link(job_id: str, user_id: int, cfg) -> str | None:
    """Ссылка на студию. None — если секрет не задан."""
    if not cfg.studio_secret:
        log.warning("STUDIO_SECRET не задан — ссылку на студию выдать нельзя")
        return None
    try:
        return auth.studio_link(cfg.studio_url, job_id, user_id, cfg.studio_secret)
    except auth.AuthError as e:
        log.warning("Ссылка на студию не выдана: %s", e)
        return None


def summary_text(proj: Project) -> str:
    """Короткая сводка: сколько спикеров и что требует внимания."""
    speakers = proj.active_speakers()
    males = sum(1 for s in speakers if s.gender == "male")
    females = sum(1 for s in speakers if s.gender == "female")
    flagged = proj.warnings_count()
    no_clone = [s for s in speakers if not s.reference.clone_allowed]

    lines = [
        "🎬 <b>Готово к проверке</b>",
        "",
        f"Спикеров: <b>{len(speakers)}</b> (♂{males} ♀{females})",
        f"Реплик: {len(proj.segments)}",
    ]
    if flagged:
        lines.append(f"Требуют внимания: <b>{flagged}</b> реплик")
    if no_clone:
        lines.append(f"Голос из банка получат: {len(no_clone)} спикеров "
                     "(мало чистой речи для клонирования)")
    lines += [
        "",
        "Откройте студию, чтобы сверить голоса с видео и поправить "
        "спорные места до озвучки. Или утвердите как есть.",
    ]
    return "\n".join(lines)


async def request_review(job, bot, cfg, proj: Project) -> None:
    """Отправляет сообщение с кнопками и ссылкой.

    Задачу можно запустить и без Telegram (`scripts/run_url.py`) — тогда
    бота нет, и ссылку надо просто напечатать. Иначе прогон из консоли
    падал бы ровно в тот момент, когда всё уже посчитано.
    """
    link = make_link(proj.job_id, job.user_id, cfg)
    text = summary_text(proj)
    if not link:
        text += ("\n\n⚠️ Ссылка на студию недоступна: не задан STUDIO_SECRET. "
                 "См. README, раздел «Студия проверки».")

    if bot is None:
        plain = re.sub(r"</?b>", "", text)
        log.info("\n%s\n\nОткройте студию: %s\n"
                 "Утвердить из консоли: создайте файл %s",
                 plain, link or "(ссылка недоступна)",
                 store.job_dir(proj.job_id) / "approved.flag")
        return

    await bot.send_message(job.chat_id, text, parse_mode="HTML",
                           reply_markup=review_keyboard(proj.job_id, link))


async def wait_for_approval(job_id: str, timeout_hours: int = 72) -> bool:
    """Ждёт утверждения. False — истёк срок ожидания.

    Ждём двумя способами сразу: событие (студия в том же процессе) и
    отметку на диске (студия запущена отдельно). Полагаться только на
    событие нельзя — процессы разные.
    """
    event = _waiting.setdefault(job_id, asyncio.Event())
    marker = store.job_dir(job_id) / "approved.flag"
    deadline = asyncio.get_running_loop().time() + timeout_hours * 3600

    try:
        while asyncio.get_running_loop().time() < deadline:
            if marker.exists():
                return True
            proj = store.load_or_none(job_id)
            if proj and proj.stage in (Stage.APPROVED, Stage.SYNTHESIZING):
                return True
            if proj and proj.stage == Stage.CANCELLED:
                return False
            try:
                await asyncio.wait_for(event.wait(), timeout=POLL_SEC)
                return True
            except asyncio.TimeoutError:
                continue
        log.info("Задача %s не утверждена за %d ч — снимаю с ожидания",
                 job_id, timeout_hours)
        return False
    finally:
        _waiting.pop(job_id, None)


async def enqueue_approved(job_id: str) -> None:
    """Вызывается студией: будит пайплайн, ожидающий утверждения."""
    event = _waiting.get(job_id)
    if event:
        event.set()


def mark_approved(job_id: str) -> None:
    """Утверждение прямо из бота (кнопка «Утвердить как есть»)."""
    def mutate(p: Project) -> None:
        p.stage = Stage.APPROVED

    try:
        store.update(job_id, mutate)
    except store.ProjectNotFound:
        log.warning("Утверждение задачи %s: проект не найден", job_id)
    (store.job_dir(job_id) / "approved.flag").write_text("bot", encoding="utf-8")


# ---------------------------------------------------------------- карта голосов

def voice_map_text(proj: Project, report: dict | None = None,
                   overall: float = 0.0) -> str:
    """«Карта голосов» для итогового сообщения.

    Главное число здесь — стабильность: именно она отвечает на исходную
    жалобу «один человек говорит несколькими голосами».
    """
    report = report or {}
    speakers = sorted(proj.active_speakers(),
                      key=lambda s: -s.stats.total_speech_sec)

    lines = ["🎭 <b>Карта голосов</b>", ""]
    for sp in speakers[:15]:
        rec = report.get(sp.id, {})
        gender = {"male": "♂", "female": "♀"}.get(sp.gender, "?")
        voice = ("клон оригинала" if sp.voice.mode == "clone"
                 else sp.voice.preset_name or sp.voice.preset_id or "пресет")
        passed = rec.get("passed")
        total = rec.get("segments", sp.stats.segments_count)
        qc = f" · QC {passed}/{total}" if passed is not None else ""
        lines.append(f"{gender} <b>{sp.name or sp.id}</b> — {voice} · "
                     f"{sp.stats.segments_count} реплик{qc}")

    if len(speakers) > 15:
        lines.append(f"…и ещё {len(speakers) - 15} спикеров")

    if overall:
        # норма зависит от того, совпадает ли язык озвучки с языком оригинала:
        # при межъязыковом дубляже 0.6 — это хорошо, а не плохо
        target = 0.75 if proj.lang_src == proj.lang_tgt else 0.60
        verdict = ("голоса стабильны" if overall >= target
                   else "есть заметный разброс — проверьте в студии")
        lines += ["", f"<b>Стабильность голосов: {overall:.2f}</b> — {verdict} "
                      f"(норма от {target:.2f})"]
    return "\n".join(lines)
