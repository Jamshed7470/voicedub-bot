"""Точка входа: python -m bot.main (запускать из папки voicedub)."""
from __future__ import annotations

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer

from core.config import LOGS_DIR, load_config


def setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOGS_DIR / "voicedub.log", encoding="utf-8"),
        ],
    )
    logging.getLogger("aiogram.event").setLevel(logging.WARNING)


async def amain() -> None:
    setup_logging()
    log = logging.getLogger("voicedub")
    cfg = load_config()

    if not cfg.bot_token:
        log.error("BOT_TOKEN не задан в .env — бот не может запуститься. "
                  "Скопируйте .env.example в .env и заполните токен.")
        sys.exit(1)

    if cfg.device == "cpu":
        log.warning("=" * 60)
        log.warning("GPU не найден — работаю на CPU. Обработка будет МЕДЛЕННОЙ.")
        log.warning("Рекомендуется MODEL_PROFILE=light в .env (сейчас: %s).",
                    cfg.profile)
        log.warning("=" * 60)
    else:
        log.info("Устройство: CUDA GPU, профиль моделей: %s", cfg.profile)

    if not cfg.hf_token:
        log.warning("HF_TOKEN не задан — определение спикеров работать НЕ будет. "
                    "См. README, раздел «Шаг 4. Токен HuggingFace».")
    if cfg.anthropic_api_key:
        log.info("Перевод: Claude API (%s)",
                 cfg.y("translation", "claude_model", default="claude-sonnet-4-6"))
    else:
        log.info("Перевод: локальный NLLB (%s)", cfg.nllb_model)

    # отправка 50 МБ по медленному каналу легко переваливает за минуту,
    # а таймаут сессии по умолчанию — 60 с: готовый дубляж срывался на выгрузке
    api_timeout = float(cfg.y("limits", "api_timeout_s", default=1800))

    # локальный сервер Bot API снимает лимит 20 МБ на скачивание
    session = AiohttpSession(timeout=api_timeout)
    if cfg.telegram_local_api_url:
        # is_local: сервер отдаёт файлы путями на диске, а не ссылками —
        # aiogram тогда читает их напрямую, без скачивания по HTTP
        kwargs = {"is_local": True}
        if cfg.telegram_local_files_dir:
            # сервер в контейнере видит свой путь, бот в Windows — свой;
            # без сопоставления aiogram не найдёт файл
            from pathlib import Path

            from aiogram.client.telegram import SimpleFilesPathWrapper
            kwargs["wrap_local_file"] = SimpleFilesPathWrapper(
                Path("/var/lib/telegram-bot-api"),
                Path(cfg.telegram_local_files_dir),
            )
        session = AiohttpSession(
            api=TelegramAPIServer.from_base(cfg.telegram_local_api_url, **kwargs),
            timeout=api_timeout)
        log.info("Использую локальный Bot API: %s (лимит отправки %.0f МБ)",
                 cfg.telegram_local_api_url, cfg.upload_limit_mb)

    bot = Bot(cfg.bot_token, session=session,
              default=DefaultBotProperties(parse_mode="HTML"))
    dp = Dispatcher()

    from bot.handlers import router
    from bot.jobqueue import JobQueue
    from core.cache import purge as purge_cache
    from core.pipeline import purge_old_jobs

    purge_old_jobs(int(cfg.y("jobs", "keep_days", default=7)))
    purge_cache(float(cfg.y("cache", "max_gb", default=10)),
                int(cfg.y("cache", "keep_days", default=14)))

    queue = JobQueue(bot, cfg)
    queue.start()
    dp["jobqueue"] = queue
    dp.include_router(router)

    # список команд в меню Telegram (кнопка «/» рядом с полем ввода)
    from aiogram.types import BotCommand
    try:
        await bot.set_my_commands([
            BotCommand(command="menu", description="Показать кнопки меню"),
            BotCommand(command="lang", description="Язык озвучки"),
            BotCommand(command="settings", description="Настройки"),
            BotCommand(command="status", description="Статус задачи"),
            BotCommand(command="cancel", description="Отменить задачу"),
            BotCommand(command="help", description="Справка"),
        ])
    except Exception:  # noqa: BLE001 — без списка команд бот работает как обычно
        log.exception("Не удалось задать список команд")

    log.info("VoiceDub Bot запущен. Ожидаю сообщения…")
    await dp.start_polling(bot)


def main() -> None:
    try:
        asyncio.run(amain())
    except (KeyboardInterrupt, SystemExit):
        print("Остановлено.")


if __name__ == "__main__":
    main()
