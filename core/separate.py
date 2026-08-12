"""Разделение голоса и фона (Demucs htdemucs).

Длинные ролики обрабатываются кусками: Demucs держит в памяти вход и оба
выхода целиком, и на двухчасовой дорожке это десятки гигабайт. Куски идут
с перекрытием и склеиваются кроссфейдом — на слух шва нет.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

from core.media import run

log = logging.getLogger(__name__)

SR = 44100


def separate(source_wav: str | Path, job_dir: str | Path, cfg,
             progress=None) -> tuple[Path, Path]:
    """Разделяет source.wav на vocals.wav и background.wav (всё остальное).

    Возвращает (vocals, background) — оба 44.1 кГц stereo, длиной с оригинал.
    """
    source_wav, job_dir = Path(source_wav), Path(job_dir)
    vocals = job_dir / "vocals.wav"
    background = job_dir / "background.wav"

    from core.media import probe_duration
    duration = probe_duration(source_wav)
    chunk_s = float(cfg.y("separate", "chunk_minutes", default=10)) * 60
    overlap = float(cfg.y("separate", "overlap_s", default=4.0))

    if chunk_s <= 0 or duration <= chunk_s + overlap:
        _separate_file(source_wav, job_dir / "demucs", cfg, vocals, background)
        if progress:
            progress(100)
    else:
        _separate_chunked(source_wav, job_dir, cfg, vocals, background,
                          duration, chunk_s, overlap, progress)

    # моно-версия вокала 16 кГц для моделей анализа (события, эмоции, референсы)
    run(["ffmpeg", "-y", "-i", str(vocals), "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(job_dir / "vocals16.wav")], desc="ffmpeg vocals16")

    return vocals, background


# ---------------------------------------------------------------------------


def _run_demucs(src: Path, out_dir: Path, cfg) -> tuple[Path, Path]:
    """Один прогон Demucs. Возвращает (vocals, no_vocals) как их отдал Demucs."""
    model = cfg.demucs_model
    cmd = [sys.executable, "-m", "demucs.separate",
           "--two-stems", "vocals",
           "-n", model,
           "-d", cfg.device,
           "-o", str(out_dir),
           str(src)]
    log.info("Demucs: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        log.error("Demucs завершился с ошибкой:\n%s", (proc.stderr or "")[-4000:])
        raise RuntimeError("Demucs: не удалось разделить голос и фон")

    stem_dir = out_dir / model / src.stem
    vocals_src = stem_dir / "vocals.wav"
    bg_src = stem_dir / "no_vocals.wav"
    if not vocals_src.exists() or not bg_src.exists():
        raise RuntimeError(f"Demucs: не найдены результаты в {stem_dir}")
    return vocals_src, bg_src


def _separate_file(source_wav: Path, work_dir: Path, cfg,
                   vocals: Path, background: Path) -> None:
    """Короткая дорожка: один прогон и приведение к 44.1 кГц stereo."""
    vocals_src, bg_src = _run_demucs(source_wav, work_dir, cfg)
    run(["ffmpeg", "-y", "-i", str(vocals_src), "-ac", "2", "-ar", str(SR),
         "-c:a", "pcm_s16le", str(vocals)], desc="ffmpeg vocals")
    run(["ffmpeg", "-y", "-i", str(bg_src), "-ac", "2", "-ar", str(SR),
         "-c:a", "pcm_s16le", str(background)], desc="ffmpeg background")
    shutil.rmtree(work_dir, ignore_errors=True)


def _separate_chunked(source_wav: Path, job_dir: Path, cfg,
                      vocals: Path, background: Path, duration: float,
                      chunk_s: float, overlap: float, progress=None) -> None:
    """Куски с перекрытием, склейка кроссфейдом в потоковой записи."""
    import numpy as np
    import soundfile as sf

    from core.media import cut_fragment

    work = job_dir / "demucs_chunks"
    work.mkdir(parents=True, exist_ok=True)
    over_n = int(overlap * SR)
    starts = [i * chunk_s for i in range(int(duration // chunk_s) + 1)
              if i * chunk_s < duration - 0.05]
    # последний кусок короче перекрытия склеивать нечем — он весь ушёл бы
    # в кроссфейд, а конец дорожки пропал. Отдаём остаток предыдущему куску.
    while len(starts) > 1 and duration - starts[-1] < max(overlap, 1.0):
        starts.pop()
    log.info("Дорожка %.1f мин — разделяю %d кусками по %.0f мин",
             duration / 60, len(starts), chunk_s / 60)

    writers = {}
    try:
        writers["v"] = sf.SoundFile(str(vocals), "w", samplerate=SR, channels=2,
                                    subtype="PCM_16")
        writers["b"] = sf.SoundFile(str(background), "w", samplerate=SR, channels=2,
                                    subtype="PCM_16")
        tails = {"v": None, "b": None}

        for idx, start in enumerate(starts):
            last = idx == len(starts) - 1
            end = duration if last else min(duration, start + chunk_s + overlap)
            piece = work / f"piece_{idx:03d}.wav"
            cut_fragment(source_wav, piece, start, end)
            try:
                v_src, b_src = _run_demucs(piece, work / f"out_{idx:03d}", cfg)
                for key, src in (("v", v_src), ("b", b_src)):
                    data, sr = sf.read(str(src), dtype="float32", always_2d=True)
                    if sr != SR:
                        import librosa
                        data = librosa.resample(data.T, orig_sr=sr,
                                                target_sr=SR).T
                    if data.shape[1] == 1:
                        data = np.repeat(data, 2, axis=1)
                    tails[key] = _write_with_crossfade(
                        writers[key], tails[key], data.astype(np.float32),
                        0 if last else over_n, over_n)
            finally:
                piece.unlink(missing_ok=True)
                shutil.rmtree(work / f"out_{idx:03d}", ignore_errors=True)
            if progress:
                progress(int(100 * (idx + 1) / len(starts)))
    finally:
        for w in writers.values():
            w.close()
        shutil.rmtree(work, ignore_errors=True)


def _write_with_crossfade(writer, tail, data, keep_tail_n: int, over_n: int):
    """Пишет кусок, сшивая его начало с хвостом предыдущего.

    tail — последние over_n отсчётов предыдущего куска (та же музыка, что и
    в начале текущего). Линейный кроссфейд по этому отрезку убирает щелчок
    на стыке. Возвращает новый хвост (или None для последнего куска).
    """
    import numpy as np

    if tail is not None and len(tail):
        n = min(len(tail), len(data), over_n)
        env = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
        head = tail[:n] * (1.0 - env) + data[:n] * env
        writer.write(head)
        data = data[n:]

    if keep_tail_n and len(data) > keep_tail_n:
        writer.write(data[:-keep_tail_n])
        return data[-keep_tail_n:].copy()

    writer.write(data)
    return None
