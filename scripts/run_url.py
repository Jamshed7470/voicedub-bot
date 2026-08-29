"""Обработка ссылки или файла без Telegram — прямо в папку готовых видео.

Нужен, когда ролик длинный: результат всё равно не влезет в чат, а гонять
задачу через бота ради этого незачем. Прогресс пишется в консоль и в лог.

    python -m scripts.run_url "https://youtu.be/..." --lang ru --voice bank

Запускать из папки voicedub (там же, где bot/ и core/).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import LOGS_DIR, TTS_LANGUAGES, load_config  # noqa: E402
from core.errors import UserError  # noqa: E402
from core.pipeline import PipelineHooks, cleanup_job, run_job  # noqa: E402

log = logging.getLogger("run_url")


@dataclass
class LocalJob:
    """То же, что Job у бота, но без чата и очереди."""
    kind: str
    payload: object
    target_lang: str
    settings: dict
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    user_id: int = 0
    chat_id: int = 0


class ConsoleProgress:
    """Печатает этап при смене подписи и раз в N секунд внутри этапа."""

    def __init__(self, every_s: float = 20.0):
        self.every_s = every_s
        self._last_key = None
        self._last_print = 0.0
        self._t0 = time.monotonic()

    async def report(self, stage: int, label: str, pct: int) -> None:
        now = time.monotonic()
        key = (stage, label)
        if key != self._last_key or now - self._last_print >= self.every_s:
            mm, ss = divmod(int(now - self._t0), 60)
            log.info("[%02d:%02d] этап %d/10: %s… %d%%", mm, ss, stage, label, pct)
            self._last_key = key
            self._last_print = now


def setup_logging(job_name: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    # консоль Windows по умолчанию cp1251: стрелка «→» и кавычки-ёлочки роняют
    # обработчик логов (UnicodeEncodeError) и заваливают вывод трейсбеками
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOGS_DIR / f"{job_name}.log", encoding="utf-8"),
        ],
    )


async def amain(args) -> int:
    cfg = load_config()
    source = args.url
    # по виду ссылки, а не по наличию файла: при продолжении задачи исходник
    # может быть уже удалён, а дорожки лежат в кэше — путь нужен лишь как ключ
    kind = "url" if source.lower().startswith(("http://", "https://")) else "local"
    job = LocalJob(
        kind=kind,
        payload=str(Path(source).resolve()) if kind == "local" else source,
        **({"id": args.job_id} if args.job_id else {}),
        target_lang=args.lang,
        settings={
            "keep_background": not args.no_background,
            "keep_original_track": args.keep_original,
            "translation_style": args.style,
            "voice_mode": args.voice,
            "speakers": args.speakers,
        },
    )
    log.info("Задача %s: %s → %s (голоса: %s, стиль: %s)", job.id, job.payload,
             TTS_LANGUAGES.get(args.lang, args.lang), args.voice, args.style)
    log.info("Устройство: %s, профиль моделей: %s", cfg.device, cfg.profile)

    progress = ConsoleProgress()
    hooks = PipelineHooks(
        report=progress.report,
        # без чата спрашивать некого: совпадение языков — не повод бросать работу
        confirm_same_lang=None,
        cancel_event=None,
    )
    try:
        result = await run_job(job, None, hooks, cfg)
    except UserError as e:
        log.error("Не получилось: %s", e.message_ru)
        return 2
    except Exception:  # noqa: BLE001 — показать причину и не потерять сделанное
        log.exception("Задача упала")
        log.error("Сделанное сохранено. Продолжить: "
                  "python -m scripts.run_url <источник> --job-id %s", job.id)
        return 3

    log.info("%s", result.summary)
    log.info("Готовый файл: %s", result.saved_path)
    log.info("Субтитры: %s", result.saved_srt)
    if result.output_path:
        size_mb = Path(result.output_path).stat().st_size / 1024 / 1024
        log.info("В Telegram влезает (%.0f МБ при лимите %.0f МБ)",
                 size_mb, cfg.upload_limit_mb)
    else:
        log.info("В лимит Telegram не влезает — файл ждёт в папке готовых видео")
    # чистим только после успеха: после сбоя папка задачи — точка продолжения
    cleanup_job(job.id)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Дубляж ссылки или файла без Telegram")
    p.add_argument("url", help="ссылка на видео или путь к файлу на диске")
    p.add_argument("--lang", default="ru", choices=sorted(TTS_LANGUAGES),
                   help="язык озвучки (по умолчанию ru)")
    p.add_argument("--voice", default="clone", choices=["clone", "bank"],
                   help="клон оригинала или голос из банка")
    p.add_argument("--style", default="normal", choices=["normal", "street"])
    p.add_argument("--speakers", default="auto",
                   help="число спикеров или auto")
    p.add_argument("--no-background", action="store_true",
                   help="не сохранять фоновую музыку")
    p.add_argument("--keep-original", action="store_true",
                   help="добавить оригинальную дорожку вторым треком")
    p.add_argument("--job-id", default="",
                   help="продолжить прерванную задачу: её id (папка data/jobs/<id>)")
    args = p.parse_args()

    setup_logging("run_url")
    try:
        sys.exit(asyncio.run(amain(args)))
    except KeyboardInterrupt:
        print("Остановлено.")
        sys.exit(1)


if __name__ == "__main__":
    main()
