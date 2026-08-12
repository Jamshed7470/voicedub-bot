"""Невербальные события (смех, плач, вздохи и т.д.) — классификатор AST.

Найденные отрезки НЕ синтезируются: они копируются из оригинальной вокальной
дорожки в финальный микс на те же тайм-коды с кроссфейдом (см. mixer.py).
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

_clf = None


def _get_classifier(cfg):
    global _clf
    if _clf is None:
        from transformers import pipeline
        model = cfg.y("events", "model",
                      default="MIT/ast-finetuned-audioset-10-10-0.4593")
        _clf = pipeline("audio-classification", model=model,
                        device=0 if cfg.device == "cuda" else -1)
    return _clf


def detect_events(vocals16: np.ndarray, sr: int, cfg, progress=None) -> list[dict]:
    """Скользящим окном ищет невербальные события в вокальной дорожке.

    Возвращает [{start, end, label}] (секунды, объединённые соседние окна).
    """
    window = float(cfg.y("events", "window_s", default=2.0))
    hop = float(cfg.y("events", "hop_s", default=1.0))
    threshold = float(cfg.y("events", "threshold", default=0.5))
    min_event = float(cfg.y("events", "min_event_s", default=0.3))
    wanted = set(cfg.y("events", "classes", default=[]) or [])

    clf = _get_classifier(cfg)

    win = int(window * sr)
    hop_n = int(hop * sr)
    batch_size = max(1, int(cfg.y("events", "batch_size", default=16)))
    hits: list[dict] = []

    offsets = [off for off in range(0, max(1, len(vocals16) - int(0.3 * sr)), hop_n)
               if len(vocals16[off:off + win]) >= int(0.3 * sr)]
    if not offsets:
        return []

    # окна идут пачками: на часовом ролике их тысячи, и накладные расходы
    # одиночного вызова pipeline съедают больше времени, чем сама модель
    for i in range(0, len(offsets), batch_size):
        batch = offsets[i:i + batch_size]
        inputs = [{"array": vocals16[off:off + win].astype(np.float32),
                   "sampling_rate": sr} for off in batch]
        try:
            results = clf(inputs, top_k=5, batch_size=len(inputs))
        except Exception:  # noqa: BLE001
            log.exception("AST: ошибка на окнах %.1f–%.1f c",
                          batch[0] / sr, batch[-1] / sr)
            continue
        if batch and isinstance(results, list) and results and isinstance(results[0], dict):
            results = [results]  # одиночный вход pipeline отдаёт без обёртки
        for off, preds in zip(batch, results):
            for p in preds:
                if p["label"] in wanted and float(p["score"]) >= threshold:
                    hits.append({
                        "start": off / sr,
                        "end": min(len(vocals16), off + win) / sr,
                        "label": p["label"],
                        "score": float(p["score"]),
                    })
                    break
        if progress:
            progress(min(100, int(100 * (i + len(batch)) / len(offsets))))

    # объединить пересекающиеся/соседние окна одного класса
    hits.sort(key=lambda h: h["start"])
    events: list[dict] = []
    for h in hits:
        if events and h["start"] <= events[-1]["end"] + 1e-6:
            events[-1]["end"] = max(events[-1]["end"], h["end"])
            events[-1]["score"] = max(events[-1]["score"], h["score"])
        else:
            events.append(dict(h))

    events = [e for e in events if e["end"] - e["start"] >= min_event]
    log.info("События: найдено %d (%s)", len(events),
             ", ".join(sorted({e["label"] for e in events})) or "—")
    return events


def mark_event_segments(segments: list[dict], events: list[dict], cfg) -> None:
    """Помечает сегменты речи, почти целиком совпадающие с событием, как
    seg["skip_tts"]=True — их не синтезируем (оригинал скопируется в микс)."""
    ratio = float(cfg.y("events", "overlap_skip_ratio", default=0.6))
    for seg in segments:
        dur = max(1e-6, seg["end"] - seg["start"])
        overlap = 0.0
        for ev in events:
            a = max(seg["start"], ev["start"])
            b = min(seg["end"], ev["end"])
            overlap += max(0.0, b - a)
        if overlap / dur >= ratio:
            seg["skip_tts"] = True
            log.debug("Сегмент %s перекрыт событием на %.0f%% — не синтезируем",
                      seg.get("id"), 100 * overlap / dur)
