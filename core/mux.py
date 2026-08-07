"""Сборка результата: mux видео с новой озвучкой, .srt, пережатие под лимит."""
from __future__ import annotations

import logging
from pathlib import Path

from core.media import run

log = logging.getLogger(__name__)


def _fmt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list[dict], out_path: str | Path) -> Path:
    """Субтитры с переводом."""
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_fmt_ts(seg['start'])} --> {_fmt_ts(seg['end'])}")
        lines.append(f"{seg['speaker']}: {seg['text']}")
        lines.append("")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    return Path(out_path)


def mux_video(video_src: str | Path, dubbed_wav: str | Path, out_path: str | Path,
              cfg, original_audio: str | Path | None = None) -> Path:
    """Собирает видео: -c:v copy, новая дорожка aac 192k.

    original_audio — если задан, добавляется вторым треком (настройка).
    """
    bitrate = cfg.y("output", "audio_bitrate", default="192k")
    cmd = ["ffmpeg", "-y", "-i", str(video_src), "-i", str(dubbed_wav)]
    if original_audio:
        cmd += ["-i", str(original_audio),
                "-map", "0:v", "-map", "1:a", "-map", "2:a",
                "-metadata:s:a:0", "title=Дубляж",
                "-metadata:s:a:1", "title=Оригинал"]
    else:
        cmd += ["-map", "0:v", "-map", "1:a"]
    cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", str(bitrate),
            "-shortest", str(out_path)]
    run(cmd, desc="ffmpeg mux")
    return Path(out_path)


def encode_audio_only(dubbed_wav: str | Path, out_path: str | Path, cfg) -> Path:
    """Если вход был аудио — отдаём только аудио (m4a)."""
    bitrate = cfg.y("output", "audio_bitrate", default="192k")
    run(["ffmpeg", "-y", "-i", str(dubbed_wav), "-c:a", "aac", "-b:a", str(bitrate),
         str(out_path)], desc="ffmpeg encode audio")
    return Path(out_path)


def compress_to_limit(video_path: Path, limit_mb: float, cfg) -> Path:
    """Если результат больше лимита отправки Bot API — пережать h264
    (crf от 26 и выше, до 720p), пока не влезет."""
    if video_path.stat().st_size <= limit_mb * 1024 * 1024:
        return video_path

    crf = int(cfg.y("output", "compress_crf_start", default=26))
    crf_max = int(cfg.y("output", "compress_crf_max", default=34))
    max_h = int(cfg.y("output", "compress_max_height", default=720))

    while crf <= crf_max:
        out = video_path.with_name(video_path.stem + f"_c{crf}.mp4")
        log.info("Пережимаю: crf=%d, до %dp", crf, max_h)
        run(["ffmpeg", "-y", "-i", str(video_path),
             "-vf", f"scale=-2:'min({max_h},ih)'",
             "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
             "-c:a", "copy", str(out)], desc="ffmpeg compress")
        if out.stat().st_size <= limit_mb * 1024 * 1024:
            return out
        out.unlink(missing_ok=True)
        crf += 2

    from core.errors import UserError
    raise UserError(
        f"Даже после сжатия результат больше {limit_mb:.0f} МБ — Telegram не даст "
        "боту его отправить. Попробуйте видео покороче."
    )
