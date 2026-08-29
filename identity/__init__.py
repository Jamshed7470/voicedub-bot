"""Speaker Identity Engine — устойчивое определение «кто говорит».

Между сырой диаризацией pyannote и синтезом стоит один слой, который
отвечает на вопрос «сколько здесь на самом деле людей». Он сводит
переразбитые кластеры обратно в одного человека, переприсваивает
сомнительные сегменты и честно помечает то, в чём не уверен, — чтобы это
увидел человек в студии, а не услышал зритель в готовом дубляже.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# Версия алгоритма разбора. Входит в имя папки кэша, поэтому её нужно
# поднимать при ЛЮБОМ изменении, влияющем на разбиение или сведение
# спикеров, — иначе старый разбор подхватится из кэша и исправление молча
# не применится к уже обработанным роликам.
#
#   1 — слияние кластеров, переприсвоение, поглощение обрывков
#   2 — реплики режутся по смене говорящего внутри блока распознавания
ALGO_VERSION = 2


def analyze(vocals16_wav, segments: list[dict], cfg,
            raw_turns: list[dict] | None = None,
            job_dir=None, speakers_hint: str = "auto") -> dict:
    """Полный проход: эмбеддинги → слияние → переприсвоение → стабильные ID.

    Меняет segments на месте: speaker, speaker_confidence, speaker_margin,
    overlap, flags. Возвращает сводку по спикерам.
    """
    import librosa

    from identity import clustering, registry
    from identity.embeddings import get_embedder

    if not segments:
        return {"speakers": {}, "clusters": []}

    y = librosa.load(str(vocals16_wav), sr=16000, mono=True)[0].astype(np.float32)
    embedder = get_embedder(cfg)

    min_sec = float(cfg.y("speaker_identity", "min_segment_sec_for_embedding",
                          default=0.8))
    spans: list[tuple[float, float]] = []
    for seg in segments:
        start, end = float(seg["start"]), float(seg["end"])
        if end - start < min_sec:
            # короткая реплика: берём окно чуть шире, иначе эмбеддинг шумный.
            # флаг остаётся — такой сегмент не годится в референс голоса
            start, end = max(0.0, start - 0.3), end + 0.3
            seg.setdefault("flags", [])
            if "too_short_for_embedding" not in seg["flags"]:
                seg["flags"].append("too_short_for_embedding")
        spans.append((start, end))

    log.info("SIE: считаю голосовые отпечатки для %d сегментов…", len(spans))
    embeddings = embedder.embed_windows(y, spans, sr=16000)

    num, mn, mx = _hint_bounds(speakers_hint, cfg)
    clusters = clustering.run(segments, embeddings, cfg, raw_turns=raw_turns,
                              num_speakers=num, min_speakers=mn, max_speakers=mx)

    previous = registry.load(job_dir) if job_dir else {}
    mapping = registry.assign_ids(
        clusters, segments, previous,
        match_threshold=float(cfg.y("speaker_identity",
                                    "registry_match_threshold", default=0.75)),
    )
    if job_dir:
        registry.save(job_dir, clusters, mapping)

    summary = {}
    for cl in clusters:
        sid = mapping[cl.label]
        summary[sid] = {
            "id": sid,
            "segments": len(cl.seg_idx),
            "speech_sec": round(float(cl.speech_sec), 2),
            "first_sec": round(float(cl.first_sec), 2),
            "spread": round(float(cl.spread), 4),
            "merged_from": cl.merged_from,
            "centroid": cl.centroid,
        }
    return {"speakers": summary, "clusters": clusters, "embeddings": embeddings}


def _hint_bounds(hint: str, cfg) -> tuple[int | None, int | None, int | None]:
    """Подсказка пользователя о числе голосов → (num, min, max)."""
    hint = (hint or "auto").strip()
    if not hint.isdigit():
        return None, None, None
    n = int(hint)
    if n >= 10:            # «10+» — точное число не задано, только нижняя граница
        return None, max(9, n), None
    return n, None, None
