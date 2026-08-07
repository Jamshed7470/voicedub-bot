"""Работа с ffmpeg/ffprobe: определение формата, извлечение аудио."""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from core.errors import UserError

log = logging.getLogger(__name__)


def run(cmd: list[str], desc: str = "ffmpeg") -> subprocess.CompletedProcess:
    """Запускает команду; при ошибке пишет stderr в лог и бросает исключение."""
    log.debug("run: %s", " ".join(map(str, cmd)))
    proc = subprocess.run(
        [str(c) for c in cmd],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        log.error("%s failed (%s):\n%s", desc, proc.returncode, proc.stderr[-4000:])
        raise RuntimeError(f"{desc}: команда завершилась с ошибкой {proc.returncode}")
    return proc


@dataclass
class MediaInfo:
    has_video: bool
    has_audio: bool
    duration: float
    width: int = 0
    height: int = 0

    @property
    def kind(self) -> str:
        """'video' если есть видеопоток, иначе 'audio'."""
        return "video" if self.has_video else "audio"


def ffprobe_json(path: str | Path) -> dict:
    proc = run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        desc="ffprobe",
    )
    return json.loads(proc.stdout or "{}")


def probe(path: str | Path) -> MediaInfo:
    """Определяет формат НЕ по расширению, а по реальным потокам (ffprobe)."""
    try:
        info = ffprobe_json(path)
    except RuntimeError:
        raise UserError(
            "Не удалось прочитать этот файл. Похоже, он повреждён или это не медиафайл."
        )
    streams = info.get("streams", [])
    # Обложки mp3 приходят как видеопоток с attached_pic — это не видео
    video = [
        s for s in streams
        if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
    ]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    duration = float(info.get("format", {}).get("duration") or 0.0)
    if not duration:
        for s in streams:
            if s.get("duration"):
                duration = max(duration, float(s["duration"]))
    width = height = 0
    if video:
        width = int(video[0].get("width") or 0)
        height = int(video[0].get("height") or 0)
    return MediaInfo(bool(video), bool(audio), duration, width, height)


def probe_duration(path: str | Path) -> float:
    return probe(path).duration


def extract_audio(src: str | Path, job_dir: str | Path,
                  analysis_sr: int = 16000, source_sr: int = 44100) -> tuple[Path, Path]:
    """Извлекает из исходника два wav:
    analysis.wav — 16 кГц mono PCM (для моделей анализа);
    source.wav   — 44.1 кГц stereo (для разделения источников и копирования фрагментов).
    """
    job_dir = Path(job_dir)
    analysis = job_dir / "analysis.wav"
    source = job_dir / "source.wav"
    run(["ffmpeg", "-y", "-i", str(src), "-vn",
         "-ac", "1", "-ar", str(analysis_sr), "-c:a", "pcm_s16le", str(analysis)],
        desc="ffmpeg extract analysis.wav")
    run(["ffmpeg", "-y", "-i", str(src), "-vn",
         "-ac", "2", "-ar", str(source_sr), "-c:a", "pcm_s16le", str(source)],
        desc="ffmpeg extract source.wav")
    return analysis, source


def cut_fragment(src_wav: str | Path, out_wav: str | Path,
                 start: float, end: float,
                 sr: int | None = None, mono: bool = False) -> Path:
    """Вырезает фрагмент [start, end) из wav."""
    cmd = ["ffmpeg", "-y", "-i", str(src_wav),
           "-ss", f"{max(0.0, start):.3f}", "-to", f"{max(start, end):.3f}"]
    if mono:
        cmd += ["-ac", "1"]
    if sr:
        cmd += ["-ar", str(sr)]
    cmd += ["-c:a", "pcm_s16le", str(out_wav)]
    run(cmd, desc="ffmpeg cut")
    return Path(out_wav)


def to_stereo_44k(src_wav: str | Path, out_wav: str | Path, sr: int = 44100) -> Path:
    run(["ffmpeg", "-y", "-i", str(src_wav), "-ac", "2", "-ar", str(sr),
         "-c:a", "pcm_s16le", str(out_wav)], desc="ffmpeg convert")
    return Path(out_wav)
