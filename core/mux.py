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


# минимальный битрейт (кбит/с), при котором картинка на такой высоте
# ещё смотрится: ниже — лучше уменьшить разрешение, чем сыпать блоками
QUALITY_LADDER = [(1080, 2000), (720, 900), (540, 550), (360, 320), (270, 180)]


def compress_to_limit(video_path: Path, limit_mb: float, cfg,
                      out_dir: Path | None = None) -> Path:
    """Пережимает видео под лимит отправки, считая битрейт от длительности.

    Перебирать качество вслепую нельзя: каждая попытка — полная перекодировка
    длиной в минуты, а для длинного ролика даже низкое качество не спасает.
    Поэтому считаем, сколько бит вообще можно потратить, и подбираем под это
    разрешение.
    """
    limit_bytes = limit_mb * 1024 * 1024
    if video_path.stat().st_size <= limit_bytes:
        return video_path

    from core.media import probe_duration
    duration = max(1.0, probe_duration(video_path))
    audio_kbps = int(cfg.y("output", "compress_audio_kbps", default=96))
    # 6% запаса на контейнер и неточность кодировщика
    budget_kbps = limit_bytes * 8 * 0.94 / duration / 1000
    video_kbps = int(budget_kbps - audio_kbps)

    if video_kbps < QUALITY_LADDER[-1][1] * 0.6:
        raise TooLongForLimit(duration, limit_mb)

    height = next((h for h, need in QUALITY_LADDER if video_kbps >= need),
                  QUALITY_LADDER[-1][0])
    log.info("Пережимаю: %.0f с при лимите %.0f МБ → %dp, видео %d кбит/с",
             duration, limit_mb, height, video_kbps)

    preset = str(cfg.y("output", "compress_preset", default="slow"))
    out = _encode(video_path, height, video_kbps, audio_kbps, "fit", preset, out_dir)
    if out.stat().st_size <= limit_bytes:
        return out

    # кодировщик промахнулся — повторяем с поправкой на фактический перелёт
    overshoot = out.stat().st_size / limit_bytes
    video_kbps = int(video_kbps / overshoot * 0.92)
    log.info("Перелёт в %.2f раза, повтор с %d кбит/с", overshoot, video_kbps)
    out.unlink(missing_ok=True)
    if video_kbps < 100:
        raise TooLongForLimit(duration, limit_mb)
    out = _encode(video_path, height, video_kbps, audio_kbps, "fit2", preset, out_dir)
    if out.stat().st_size <= limit_bytes:
        return out
    out.unlink(missing_ok=True)
    raise TooLongForLimit(duration, limit_mb)


class TooLongForLimit(Exception):
    """Ролик не влезает в лимит даже при разумном качестве — резать на части."""

    def __init__(self, duration: float, limit_mb: float):
        super().__init__(f"{duration:.0f}s > {limit_mb}MB")
        self.duration = duration
        self.limit_mb = limit_mb


def _encode(src: Path, height: int, video_kbps: int, audio_kbps: int,
            suffix: str, preset: str = "slow", out_dir: Path | None = None) -> Path:
    """Одна перекодировка под заданный битрейт.

    Пресет медленный намеренно: проходов теперь один вместо пяти, а на
    низком битрейте качество на бит заметно важнее скорости.
    """
    folder = Path(out_dir) if out_dir else src.parent
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{src.stem}_{suffix}.mp4"
    run(["ffmpeg", "-y", "-i", str(src),
         "-vf", f"scale=-2:'min({height},ih)'",
         "-c:v", "libx264", "-b:v", f"{video_kbps}k",
         "-maxrate", f"{int(video_kbps * 1.3)}k",
         "-bufsize", f"{int(video_kbps * 2)}k",
         "-preset", preset, "-movflags", "+faststart",
         "-c:a", "aac", "-b:a", f"{audio_kbps}k", str(out)],
        desc="ffmpeg compress")
    return out


def plan_parts(video_path: Path, limit_mb: float) -> int:
    """Сколько частей понадобится, чтобы каждая влезла в лимит."""
    size_mb = video_path.stat().st_size / 1024 / 1024
    # с запасом: части режутся по ключевым кадрам, размер гуляет
    return max(2, int(size_mb / (limit_mb * 0.85)) + 1)


def split_to_parts(video_path: Path, limit_mb: float,
                   out_dir: Path | None = None) -> list[Path]:
    """Режет видео на части под лимит — чтобы работа не пропала зря."""
    from core.media import probe_duration

    duration = max(1.0, probe_duration(video_path))
    parts_count = plan_parts(video_path, limit_mb)
    chunk = duration / parts_count
    folder = Path(out_dir) if out_dir else video_path.parent
    folder.mkdir(parents=True, exist_ok=True)
    pattern = folder / f"{video_path.stem}_part%02d.mp4"
    log.info("Не влезает целиком — режу на %d частей по %.0f с",
             parts_count, chunk)
    run(["ffmpeg", "-y", "-i", str(video_path), "-c", "copy",
         "-f", "segment", "-segment_time", f"{chunk:.2f}",
         "-reset_timestamps", "1", "-movflags", "+faststart", str(pattern)],
        desc="ffmpeg split")
    parts = sorted(folder.glob(f"{video_path.stem}_part*.mp4"))
    return [p for p in parts if p.stat().st_size > 0]
