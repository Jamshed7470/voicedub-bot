"""Диаризация спикеров (pyannote 3.1) и объединение с WhisperX."""
from __future__ import annotations

import logging

from core.errors import UserError

log = logging.getLogger(__name__)


def _speaker_bounds(hint: str, cfg) -> tuple[int | None, int | None, int | None]:
    """Подсказка о числе голосов → (num, min, max) для pyannote.

    Без подсказки pyannote кластеризует консервативно и сливает похожие
    голоса — на многоголосых роликах это даёт 2 спикера вместо шести.
    """
    hint = (hint or "auto").strip()
    cap = int(cfg.y("diarization", "max_speakers", default=60))
    if not hint.isdigit():
        # «авто»: верхнюю границу задаём щедро — на фильме бывает 20-30+ голосов,
        # низкий cap схлопывал их в 10 кластеров и один голос доставался многим
        return None, None, cap
    n = int(hint)
    if n >= 10:  # «10+» — верхнюю границу не угадать, задаём только нижнюю
        return None, max(9, n), max(cap, n)
    return n, None, None


def diarize_and_assign(analysis_wav: str, asr_result: dict, cfg,
                       speakers_hint: str = "auto",
                       with_turns: bool = False):
    """Определяет спикеров и присваивает их сегментам/словам.

    Возвращает список сегментов:
    [{id, start, end, text, speaker}] — speaker: СЫРЫЕ метки кластеров
    pyannote. Сведение кластеров в реальных людей и выдача стабильных ID
    S1, S2… — задача Speaker Identity Engine (пакет identity/).

    with_turns=True — вернуть ещё и сырые интервалы диаризации
    (start, end, speaker): по ним определяются наложения речи.
    """
    if not cfg.hf_token:
        raise UserError(
            "Не задан HF_TOKEN — без него не работает определение спикеров.\n"
            "Как получить бесплатный токен HuggingFace — см. README, "
            "раздел «Шаг 4. Токен HuggingFace»."
        )

    from core.sb_compat import patch_speechbrain_lazy_imports
    patch_speechbrain_lazy_imports()

    import whisperx

    try:
        try:  # whisperx >= 3.2
            from whisperx.diarize import DiarizationPipeline
        except ImportError:  # старые версии
            DiarizationPipeline = whisperx.DiarizationPipeline
        pipeline = DiarizationPipeline(use_auth_token=cfg.hf_token, device=cfg.device)
        threshold = cfg.y("diarization", "clustering_threshold", default=None)
        if threshold:
            # ниже порога — охотнее разделяет похожие голоса
            try:
                pipeline.model.clustering.threshold = float(threshold)
            except AttributeError:
                log.warning("pyannote: не удалось задать порог кластеризации")
        num, mn, mx = _speaker_bounds(speakers_hint, cfg)
        log.info("Диаризация: подсказка «%s» → num=%s min=%s max=%s",
                 speakers_hint, num, mn, mx)
        audio = whisperx.load_audio(str(analysis_wav))
        diarization = pipeline(audio, num_speakers=num,
                               min_speakers=mn, max_speakers=mx)
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
    turns = _raw_turns(diarization)

    # Метки остаются сырыми (SPEAKER_00…): переименование в S1, S2…
    # делает identity/registry.py — только он знает, какие кластеры
    # на самом деле один человек
    seen: set[str] = set()
    out: list[dict] = []
    last_speaker = None
    orphans = 0
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        raw = seg.get("speaker")
        if raw is None:
            # у сегмента нет спикера (тихая речь) — приписать предыдущему
            raw = last_speaker or "SPEAKER_00"
            orphans += 1
        last_speaker = raw
        seen.add(raw)
        out.append({
            "start": float(seg.get("start") or 0.0),
            "end": float(seg.get("end") or 0.0),
            "text": text,
            "speaker": raw,
        })

    out = merge_short_segments(
        out,
        min_dur=float(cfg.y("segments", "min_duration_s", default=1.0)),
        max_gap=float(cfg.y("segments", "merge_gap_s", default=0.3)),
    )
    for i, seg in enumerate(out, 1):
        seg["id"] = i
    log.info("Диаризация: %d сегментов, %d сырых кластеров%s", len(out), len(seen),
             f", без метки спикера: {orphans}" if orphans else "")
    return (out, turns) if with_turns else out


def _raw_turns(diarization) -> list[dict]:
    """Интервалы диаризации как есть — нужны для поиска наложений речи."""
    turns: list[dict] = []
    try:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append({"start": float(turn.start), "end": float(turn.end),
                          "speaker": str(speaker)})
    except AttributeError:
        # whisperx может отдать DataFrame вместо pyannote.Annotation
        try:
            for _, row in diarization.iterrows():
                turns.append({"start": float(row["start"]), "end": float(row["end"]),
                              "speaker": str(row["speaker"])})
        except Exception:  # noqa: BLE001
            log.warning("Не удалось прочитать интервалы диаризации — "
                        "наложения речи не будут помечены")
    return turns


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
