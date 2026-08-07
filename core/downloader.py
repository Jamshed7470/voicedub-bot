"""Скачивание входных данных: файлы Telegram и ссылки (yt-dlp)."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from core.errors import UserError

log = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def find_url(text: str) -> str | None:
    m = URL_RE.search(text or "")
    return m.group(0).rstrip(").,>») ") if m else None


async def download_from_telegram(bot, file_id: str, file_size: int | None,
                                 dest: Path, cfg) -> Path:
    """Скачивает файл из Telegram через Bot API.

    Без локального сервера Bot API лимит скачивания — 20 МБ.
    """
    limit_mb = float(cfg.y("limits", "telegram_download_mb", default=20))
    if not cfg.telegram_local_api_url and file_size and file_size > limit_mb * 1024 * 1024:
        raise UserError(
            f"Файл больше {limit_mb:.0f} МБ — Telegram не даёт ботам скачивать такие файлы.\n"
            "Пришлите, пожалуйста, ссылку на видео (YouTube, TikTok, прямую ссылку и т.д.) — "
            "я скачаю его сам."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    await bot.download(file_id, destination=dest)
    if not dest.exists() or dest.stat().st_size == 0:
        raise UserError("Не удалось скачать файл из Telegram. Попробуйте ещё раз.")
    return dest


def download_url(url: str, dest_dir: Path, cfg) -> Path:
    """Скачивает медиа по ссылке через yt-dlp (bestvideo+bestaudio, merge в mp4)."""
    import yt_dlp  # ленивый импорт

    dest_dir.mkdir(parents=True, exist_ok=True)
    max_dur = cfg.max_duration_s

    opts = {
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": str(dest_dir / "input.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            duration = info.get("duration") or 0
            if duration and duration > max_dur:
                raise UserError(
                    f"Видео слишком длинное: {duration / 60:.0f} мин. "
                    f"Лимит — {max_dur / 60:.0f} мин."
                )
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info))
            # после merge расширение может отличаться
            if not path.exists():
                candidates = sorted(dest_dir.glob("input.*"),
                                    key=lambda p: p.stat().st_size, reverse=True)
                if not candidates:
                    raise UserError("Не удалось скачать видео по ссылке.")
                path = candidates[0]
            return path
    except UserError:
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("yt-dlp: ошибка скачивания %s", url)
        raise UserError(
            "Не получилось скачать по этой ссылке. Проверьте, что ссылка открывается, "
            "видео публичное и не длиннее лимита. Если это приватное видео — пришлите файлом."
        ) from e
