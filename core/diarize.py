"""Диаризация спикеров (pyannote 3.1) и объединение с WhisperX."""
from __future__ import annotations

import logging

from core.errors import UserError

log = logging.getLogger(__name__)


def diarize_and_assign(analysis_wav: str, asr_result: dict, cfg) -> list[dict]:
    """Определяет спикеров и присваивает их сегментам/словам.

    Возвращает список сегментов:
    [{id, start, end, text, speaker}] — speaker: стабильные S1, S2, ...
    по порядку первого появления. Наложение речи поддерживается: у каждого
    сегмента свои тайм-коды, при синтезе они микшируются каждый на своём месте.
    """
    if not cfg.hf_token:
        raise UserError(
            "Не задан HF_TOKEN — без него не работает определение спикеров.\n"
            "Как получить бесплатный токен HuggingFace — см. README, "
            "раздел «Шаг 4. Токен HuggingFace»."
        )

    import whisperx

    try:
        try:  # whisperx >= 3.2
            from whisperx.diarize import DiarizationPipeline
        except ImportError:  # старые версии
            DiarizationPipeline = whisperx.DiarizationPipeline
        pipeline = DiarizationPipeline(use_auth_token=cfg.hf_token, device=cfg.device)
        audio = whisperx.load_audio(str(analysis_wav))
        diarization = pipeline(audio)
    except UserError:
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("pyannote: ошибка диаризации")
        raise UserError(
            "Не удалось определить спикеров. Чаще всего это значит, что для моделей "
            "pyannote/speaker-diarization-3.1 и pyannote/segmentation-3.0 не принято "
            "соглашение на HuggingFace (кнопка «Agree and access» на странице модели) "
            "или неверный HF_TOKEN. Подробности — в README."
        ) from e

    result = whisperx.assign_word_speakers(diarization, asr_result)
    segments = result.get("segments") or []

    # Стабильные ID: S1, S2, ... по порядку первого появления
    mapping: dict[str, str] = {}
    out: list[dict] = []
    last_speaker = None
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        raw = seg.get("speaker")
        if raw is None:
            # у сегмента нет спикера (тихая речь) — приписать предыдущему
            raw = last_speaker or "SPEAKER_00"
        last_speaker = raw
        if raw not in mapping:
            mapping[raw] = f"S{len(mapping) + 1}"
        out.append({
            "start": float(seg.get("start") or 0.0),
            "end": float(seg.get("end") or 0.0),
            "text": text,
            "speaker": mapping[raw],
        })

    out = merge_short_segments(
        out,
        min_dur=float(cfg.y("segments", "min_duration_s", default=1.0)),
        max_gap=float(cfg.y("segments", "merge_gap_s", default=0.3)),
    )
    for i, seg in enumerate(out, 1):
        seg["id"] = i
    log.info("Диаризация: %d сегментов, %d спикеров", len(out), len(mapping))
    return out


def merge_short_segments(segments: list[dict], min_dur: float = 1.0,
                         max_gap: float = 0.3) -> list[dict]:
    """Сегменты короче min_dur склеиваются с соседним сегментом того же спикера,
    если пауза между ними меньше max_gap (правило из спецификации)."""
    if not segments:
        return segments
    merged: list[dict] = []
    for seg in segments:
        if merged:
            prev = merged[-1]
            gap = seg["start"] - prev["end"]
            short = (seg["end"] - seg["start"] < min_dur
                     or prev["end"] - prev["start"] < min_dur)
            if prev["speaker"] == seg["speaker"] and gap < max_gap and short:
                prev["end"] = max(prev["end"], seg["end"])
                prev["text"] = (prev["text"] + " " + seg["text"]).strip()
                continue
        merged.append(dict(seg))
    return merged
