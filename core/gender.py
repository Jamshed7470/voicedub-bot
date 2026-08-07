"""Определение пола спикера — агрегированно по всем его сегментам.

Метод (из спецификации):
1) классификатор wav2vec2 на каждом сегменте длиной >= 1.5 с;
2) параллельно медиана F0 (librosa.pyin) по всем сегментам:
   медиана < 155 Гц → мужчина, > 175 Гц → женщина;
3) итог — голосование, взвешенное по длительности сегментов;
   при уверенности классификатора < 0.8 решающий голос у F0.
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
        model = cfg.y("gender", "model",
                      default="alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech")
        _clf = pipeline("audio-classification", model=model,
                        device=0 if cfg.device == "cuda" else -1)
    return _clf


def _f0_median(y: np.ndarray, sr: int) -> float | None:
    import librosa
    try:
        f0, _, _ = librosa.pyin(y.astype(np.float32), fmin=60, fmax=400, sr=sr)
        f0 = f0[~np.isnan(f0)]
        if f0.size < 10:
            return None
        return float(np.median(f0))
    except Exception:  # noqa: BLE001
        log.exception("pyin: не удалось оценить F0")
        return None


def detect_gender(vocals16: np.ndarray, sr: int, segments: list[dict], cfg) -> dict:
    """Определяет пол спикера по его сегментам.

    vocals16 — вся вокальная дорожка 16 кГц mono (np.float32),
    segments — сегменты ЭТОГО спикера [{start, end}, ...].

    Возвращает {"gender": "male"|"female", "confidence": float, "f0_median": float|None}.
    """
    min_seg = float(cfg.y("gender", "min_segment_s", default=1.5))
    male_max = float(cfg.y("gender", "f0_male_max_hz", default=155))
    female_min = float(cfg.y("gender", "f0_female_min_hz", default=175))
    conf_thr = float(cfg.y("gender", "clf_confidence_threshold", default=0.8))

    clf = _get_classifier(cfg)

    votes_male = votes_female = 0.0
    conf_sum = weight_sum = 0.0
    f0_values: list[tuple[float, float]] = []  # (f0, duration)

    for seg in segments:
        a, b = int(seg["start"] * sr), int(seg["end"] * sr)
        dur = (b - a) / sr
        if b <= a:
            continue
        y = vocals16[a:b]

        f0 = _f0_median(y, sr)
        if f0:
            f0_values.append((f0, dur))

        if dur < min_seg:
            continue
        try:
            preds = clf({"array": y.astype(np.float32), "sampling_rate": sr})
        except Exception:  # noqa: BLE001
            log.exception("gender clf: ошибка на сегменте")
            continue
        if not preds:
            continue
        top = preds[0]
        label = str(top["label"]).lower()
        score = float(top["score"])
        conf_sum += score * dur
        weight_sum += dur
        if "female" in label or label.startswith("f"):
            votes_female += score * dur
        else:
            votes_male += score * dur

    clf_conf = (conf_sum / weight_sum) if weight_sum else 0.0
    clf_gender = None
    if votes_male or votes_female:
        clf_gender = "male" if votes_male >= votes_female else "female"

    f0_med = None
    f0_gender = None
    if f0_values:
        # взвешенная по длительности медиана
        arr = sorted(f0_values)
        total = sum(d for _, d in arr)
        acc = 0.0
        for f0v, d in arr:
            acc += d
            if acc >= total / 2:
                f0_med = f0v
                break
        if f0_med is not None:
            if f0_med < male_max:
                f0_gender = "male"
            elif f0_med > female_min:
                f0_gender = "female"

    # Итоговое решение
    if clf_gender and clf_conf >= conf_thr:
        gender, confidence = clf_gender, clf_conf
    elif f0_gender:
        gender, confidence = f0_gender, 0.7 if clf_gender != f0_gender else 0.9
    elif clf_gender:
        gender, confidence = clf_gender, clf_conf
    elif f0_med is not None:
        gender, confidence = ("male" if f0_med < 165 else "female"), 0.55
    else:
        gender, confidence = "male", 0.5  # совсем нет данных — нейтральный резерв

    return {"gender": gender, "confidence": round(confidence, 3),
            "f0_median": round(f0_med, 1) if f0_med else None}
