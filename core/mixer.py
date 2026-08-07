"""Финальный микс: фон + дублированные голоса + скопированные невербальные события.

- background.wav — вся длина, оригинальная громкость;
- каждый синтезированный сегмент кладётся ТОЧНО на исходный start;
- невербальные события копируются из оригинального вокала с кроссфейдом 50 мс;
- громкость голосов нормализуется к уровню оригинальных голосов.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from core.timing import start_sample

log = logging.getLogger(__name__)

SR = 44100


def _load_stereo(path: str | Path) -> np.ndarray:
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    assert sr == SR, f"Ожидалось {SR} Гц, получено {sr} ({path})"
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return data


def _load_any_to_stereo(path: str | Path, tmp_dir: Path) -> np.ndarray:
    """Загружает wav любой частоты, при необходимости конвертируя в 44.1k stereo."""
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR:
        from core.media import to_stereo_44k
        conv = tmp_dir / (Path(path).stem + "_44k.wav")
        to_stereo_44k(path, conv, SR)
        data, sr = sf.read(str(conv), dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return data


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


def _fade(x: np.ndarray, fade_len: int) -> np.ndarray:
    """Кроссфейд-огибающая по краям фрагмента."""
    x = x.copy()
    n = min(fade_len, len(x) // 2)
    if n > 0:
        env_in = np.linspace(0.0, 1.0, n)[:, None]
        env_out = np.linspace(1.0, 0.0, n)[:, None]
        x[:n] *= env_in
        x[-n:] *= env_out
    return x


def build_mix(job_dir: str | Path, background_wav: str | Path, vocals_wav: str | Path,
              placed_segments: list[dict], events: list[dict], cfg,
              keep_background: bool = True) -> Path:
    """Собирает финальную аудиодорожку dubbed.wav (44.1 кГц stereo).

    placed_segments: [{"start": float, "path": str}] — синтезированные сегменты.
    events: [{"start", "end"}] — отрезки, копируемые из оригинального вокала.
    """
    import soundfile as sf

    job_dir = Path(job_dir)
    tmp_dir = job_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    background = _load_stereo(background_wav)
    vocals = _load_stereo(vocals_wav)
    total = len(background)

    if not keep_background:
        background = np.zeros_like(background)

    # --- шина дублированных голосов ---
    voice_bus = np.zeros((total, 2), dtype=np.float32)
    for item in placed_segments:
        seg_audio = _load_any_to_stereo(item["path"], tmp_dir)
        pos = start_sample(item["start"], SR)  # точно на исходный start
        end = min(total, pos + len(seg_audio))
        if pos >= total:
            continue
        voice_bus[pos:end] += seg_audio[: end - pos]

    # --- невербальные события: копия оригинала с кроссфейдом ---
    fade_len = int(SR * float(cfg.y("events", "crossfade_ms", default=50)) / 1000)
    for ev in events:
        a = start_sample(ev["start"], SR)
        b = min(total, len(vocals), start_sample(ev["end"], SR))
        if b <= a:
            continue
        voice_bus[a:b] += _fade(vocals[a:b], fade_len)

    # --- нормализация громкости голосов к уровню оригинальных (RMS) ---
    orig_rms = _rms(vocals)
    dub_rms = _rms(voice_bus)
    if dub_rms > 1e-9 and orig_rms > 1e-9:
        gain = orig_rms / dub_rms
        gain = float(np.clip(gain, 0.25, 4.0))
        voice_bus *= gain
        log.info("Микс: выравнивание громкости голосов, gain %.2f", gain)

    mix = background + voice_bus

    # защита от клиппинга: пик к -1 dBFS
    peak_target = 10 ** (float(cfg.y("mix", "peak_dbfs", default=-1.0)) / 20)
    peak = float(np.max(np.abs(mix)) + 1e-12)
    if peak > peak_target:
        mix *= peak_target / peak

    out = job_dir / "dubbed.wav"
    sf.write(str(out), mix.astype(np.float32), SR)
    return out
