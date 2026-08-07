"""Профили спикеров: чистые референсы голоса, банк эмоций, пол.

Для каждого спикера строится ref_main.wav (15–30 с самой чистой речи) —
он используется для ВСЕХ сегментов этого спикера, поэтому голос уникален
и стабилен на всём видео.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

REF_SR = 24000  # частота референсов для XTTS


def _load_mono(path: str | Path, sr: int) -> np.ndarray:
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def _snr_proxy(y: np.ndarray, noise_floor: float) -> float:
    """Простой SNR-прокси: RMS сегмента к глобальному «полу» шума."""
    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-9)
    return 20 * np.log10(rms / max(noise_floor, 1e-9))


def _noise_floor(y: np.ndarray, sr: int) -> float:
    """10-й перцентиль RMS по кадрам — оценка пола шума."""
    frame = int(0.05 * sr)
    if len(y) < frame:
        return 1e-6
    n_frames = len(y) // frame
    frames = y[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9
    return float(np.percentile(rms, 10))


def build_profiles(job_dir: str | Path, vocals_wav: str | Path,
                   segments: list[dict], cfg, progress=None) -> dict:
    """Строит профили всех спикеров и сохраняет speakers.json.

    Возвращает dict: {speaker_id: {"ref_main": path, "refs_emotion": {...},
                                   "gender": ..., "confidence": ..., ...}}
    """
    import noisereduce as nr
    import soundfile as sf

    from core.gender import detect_gender

    job_dir = Path(job_dir)
    speakers_dir = job_dir / "speakers"
    speakers_dir.mkdir(parents=True, exist_ok=True)

    min_ref = float(cfg.y("speaker_ref", "min_ref_s", default=3.0))
    target_min = float(cfg.y("speaker_ref", "target_ref_min_s", default=15))
    target_max = float(cfg.y("speaker_ref", "target_ref_max_s", default=30))

    # референсы режем из чистого вокала (24 кГц для XTTS)
    y_ref = _load_mono(vocals_wav, REF_SR)
    # анализ пола — на 16 кГц
    y16 = _load_mono(vocals_wav, 16000)
    floor = _noise_floor(y_ref, REF_SR)

    by_speaker: dict[str, list[dict]] = {}
    for seg in segments:
        by_speaker.setdefault(seg["speaker"], []).append(seg)

    profiles: dict[str, dict] = {}
    speaker_ids = sorted(by_speaker, key=lambda s: int(s[1:]))
    for idx, spk in enumerate(speaker_ids):
        segs = by_speaker[spk]
        spk_dir = speakers_dir / spk
        spk_dir.mkdir(exist_ok=True)

        # оценка качества каждого сегмента
        scored = []
        for seg in segs:
            a, b = int(seg["start"] * REF_SR), int(seg["end"] * REF_SR)
            if b - a < int(1.0 * REF_SR):
                continue
            piece = y_ref[a:b]
            snr = _snr_proxy(piece, floor)
            scored.append((snr, seg, piece))
        scored.sort(key=lambda t: t[0], reverse=True)

        # собрать 15–30 с самой чистой речи
        chunks, total = [], 0.0
        for snr, seg, piece in scored:
            if total >= target_min and total + len(piece) / REF_SR > target_max:
                break
            chunks.append(piece)
            total += len(piece) / REF_SR
            if total >= target_max:
                break
        if not chunks:  # спикер говорит совсем мало — берём всё, что есть
            for seg in segs:
                a, b = int(seg["start"] * REF_SR), int(seg["end"] * REF_SR)
                if b > a:
                    chunks.append(y_ref[a:b])
        ref = np.concatenate(chunks) if chunks else np.zeros(REF_SR, dtype=np.float32)

        # шумоподавление референса
        try:
            ref = nr.reduce_noise(y=ref, sr=REF_SR, stationary=False,
                                  prop_decrease=0.9).astype(np.float32)
        except Exception:  # noqa: BLE001
            log.exception("noisereduce: не удалось почистить референс %s", spk)

        ref_main = spk_dir / "ref_main.wav"
        sf.write(str(ref_main), ref, REF_SR)
        ref_total = len(ref) / REF_SR

        # банк эмоциональных референсов
        refs_emotion: dict[str, str] = {}
        by_emotion: dict[str, list[np.ndarray]] = {}
        for seg in segs:
            emo = seg.get("emotion", "neutral")
            if emo == "neutral":
                continue
            a, b = int(seg["start"] * REF_SR), int(seg["end"] * REF_SR)
            if b - a >= int(1.5 * REF_SR):
                by_emotion.setdefault(emo, []).append(y_ref[a:b])
        for emo, pieces in by_emotion.items():
            arr = np.concatenate(pieces)[: int(target_max * REF_SR)]
            p = spk_dir / f"ref_{emo}.wav"
            sf.write(str(p), arr, REF_SR)
            refs_emotion[emo] = str(p)

        # пол — агрегированно по всем сегментам спикера
        g = detect_gender(y16, 16000, segs, cfg)

        profiles[spk] = {
            "id": spk,
            "ref_main": str(ref_main),
            "ref_total_s": round(ref_total, 2),
            "ref_ok": ref_total >= min_ref,
            "refs_emotion": refs_emotion,
            "gender": g["gender"],
            "gender_confidence": g["confidence"],
            "f0_median": g["f0_median"],
            "segments": len(segs),
            "speech_total_s": round(sum(s["end"] - s["start"] for s in segs), 2),
        }
        log.info("Спикер %s: %s (conf %.2f), референс %.1f с, сегментов %d",
                 spk, g["gender"], g["confidence"], ref_total, len(segs))
        if progress:
            progress(int(100 * (idx + 1) / len(speaker_ids)))

    with open(job_dir / "speakers.json", "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
    return profiles
