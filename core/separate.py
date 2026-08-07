"""Разделение голоса и фона (Demucs htdemucs)."""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

from core.media import run

log = logging.getLogger(__name__)


def separate(source_wav: str | Path, job_dir: str | Path, cfg) -> tuple[Path, Path]:
    """Разделяет source.wav на vocals.wav и background.wav (всё остальное).

    Возвращает (vocals, background) — оба 44.1 кГц stereo, длиной с оригинал.
    """
    source_wav, job_dir = Path(source_wav), Path(job_dir)
    out_dir = job_dir / "demucs"
    model = cfg.demucs_model

    cmd = [sys.executable, "-m", "demucs.separate",
           "--two-stems", "vocals",
           "-n", model,
           "-d", cfg.device,
           "-o", str(out_dir),
           str(source_wav)]
    log.info("Demucs: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        log.error("Demucs завершился с ошибкой:\n%s", (proc.stderr or "")[-4000:])
        raise RuntimeError("Demucs: не удалось разделить голос и фон")

    stem_dir = out_dir / model / source_wav.stem
    vocals_src = stem_dir / "vocals.wav"
    bg_src = stem_dir / "no_vocals.wav"
    if not vocals_src.exists() or not bg_src.exists():
        raise RuntimeError(f"Demucs: не найдены результаты в {stem_dir}")

    vocals = job_dir / "vocals.wav"
    background = job_dir / "background.wav"
    # привести к 44.1 кГц stereo pcm_s16le (htdemucs также отдаёт 44.1k)
    run(["ffmpeg", "-y", "-i", str(vocals_src), "-ac", "2", "-ar", "44100",
         "-c:a", "pcm_s16le", str(vocals)], desc="ffmpeg vocals")
    run(["ffmpeg", "-y", "-i", str(bg_src), "-ac", "2", "-ar", "44100",
         "-c:a", "pcm_s16le", str(background)], desc="ffmpeg background")
    shutil.rmtree(out_dir, ignore_errors=True)

    # моно-версия вокала 16 кГц для моделей анализа (события, эмоции, референсы)
    vocals16 = job_dir / "vocals16.wav"
    run(["ffmpeg", "-y", "-i", str(vocals), "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(vocals16)], desc="ffmpeg vocals16")

    return vocals, background
