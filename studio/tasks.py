"""Тяжёлые операции студии: пересборка профиля, повторная диаризация,
уведомление бота об утверждении.

Вынесены из api.py, потому что выполняются в рабочем потоке и обращаются
к моделям, а обработчики запросов должны оставаться быстрыми.
"""
from __future__ import annotations

import logging

from project import store
from project.schema import Project, Stage

log = logging.getLogger(__name__)


def rebuild_speaker_profile(job_id: str, speaker_id: str, cfg) -> dict:
    """Заново собирает референс и латенты одного спикера.

    Нужна после объединения спикеров (набор реплик изменился) и по кнопке
    «Пересобрать профиль», когда человек поправил принадлежность реплик.
    """
    import librosa

    from identity.embeddings import get_embedder
    from synth.xtts_engine import get_engine
    from voices import profiles as prof_mod

    proj = store.load(job_id)
    speaker = proj.speaker(speaker_id)
    if speaker is None:
        raise ValueError(f"спикер {speaker_id} не найден")

    vocals = proj.source.vocals_path
    if not vocals:
        raise ValueError("дорожка голоса недоступна — медиа удалены")

    segs = [s.model_dump(mode="json") for s in proj.segments
            if s.speaker_id == speaker_id]
    for s in segs:                     # profiles ждёт ключи пайплайна
        s["text"] = s.get("text_tgt", "")
    if not segs:
        raise ValueError("у спикера не осталось реплик")

    y_ref = librosa.load(str(vocals), sr=prof_mod.REF_SR, mono=True)[0]
    y16 = librosa.load(str(vocals), sr=prof_mod.EMB_SR, mono=True)[0]

    embedder = get_embedder(cfg)
    centroid = _load_centroid(job_id, speaker_id)
    ref = prof_mod.select_reference(segs, y_ref, y16, centroid, embedder, cfg)

    spk_dir = store.job_dir(job_id) / "speakers" / speaker_id
    spk_dir.mkdir(parents=True, exist_ok=True)
    ref_path = spk_dir / "ref_main.wav"
    if ref["audio"] is None or not len(ref["audio"]):
        raise ValueError("не набралось чистой речи для референса")

    import soundfile as sf
    sf.write(str(ref_path), ref["audio"], prof_mod.REF_SR)

    profile = prof_mod.build_profile(speaker_id, ref_path, get_engine(cfg),
                                     embedder, mode="clone")
    paths = prof_mod.save_profile(profile, spk_dir)

    def mutate(p: Project) -> None:
        sp = p.speaker(speaker_id)
        sp.reference.path = str(ref_path)
        sp.reference.clean_sec = ref["clean_sec"]
        sp.reference.snr_db = ref["snr_db"]
        sp.reference.score = ref["score"]
        sp.reference.clone_allowed = bool(ref["clone_allowed"])
        sp.reference.best_samples = ref["best_samples"]
        sp.voice.profile_path = paths["profile_path"]
        sp.voice.identity_path = paths["identity_path"]
        sp.voice.locked = True
        for seg in p.segments:
            if seg.speaker_id == speaker_id:
                seg.synth.status = "pending"

    store.update(job_id, mutate)
    log.info("Профиль %s пересобран: %.1f с чистой речи", speaker_id,
             ref["clean_sec"])
    return {"clean_sec": ref["clean_sec"], "clone_allowed": ref["clone_allowed"]}


def _load_centroid(job_id: str, speaker_id: str):
    import numpy as np

    path = store.job_dir(job_id) / "speakers" / speaker_id / "centroid.npy"
    if path.exists():
        try:
            return np.load(str(path)).astype(np.float32)
        except (OSError, ValueError):
            pass
    return None


def rediarize_project(job_id: str, cfg, num_speakers: int | None = None,
                      min_speakers: int | None = None,
                      max_speakers: int | None = None) -> dict:
    """Пересчитывает разбиение на спикеров по уже готовому разбору.

    Заново гонять pyannote не нужно: сырые кластеры сохранены в реестре,
    а меняется только их сведение. Правки пользователя сохраняются через
    реестр ID — в этом и смысл INV-3.
    """
    import identity as sie

    proj = store.load(job_id)
    vocals = proj.source.vocals_path
    if not vocals:
        raise ValueError("дорожка голоса недоступна — медиа удалены")

    hint = "auto"
    if num_speakers:
        hint = str(num_speakers)
    elif min_speakers:
        hint = str(min_speakers)

    segments = []
    for seg in proj.segments:
        segments.append({
            "id": seg.id, "start": seg.start, "end": seg.end,
            "text": seg.text_tgt, "speaker": seg.speaker_id, "flags": [],
            "emotion": seg.emotion, "events": list(seg.events),
        })

    result = sie.analyze(vocals, segments, cfg, raw_turns=None,
                         job_dir=store.job_dir(job_id), speakers_hint=hint)
    summary = result["speakers"]

    def mutate(p: Project) -> None:
        from project.schema import Speaker, free_color

        by_id = {s.id: s for s in p.speakers}
        manual = {s.id: s.speaker_id for s in p.segments
                  if "speaker_id" in s.edited_by_user.fields}

        for seg, fresh in zip(p.segments, segments):
            # ручное назначение сильнее пересчёта: человек уже посмотрел
            # это место с видео, а алгоритм — нет
            seg.speaker_id = manual.get(seg.id, fresh["speaker"])
            seg.speaker_confidence = float(fresh.get("speaker_confidence", 0.0))
            seg.speaker_margin = float(fresh.get("speaker_margin", 0.0))
            seg.flags = list(fresh.get("flags", []))

        new_ids = set(summary) | set(manual.values())
        for idx, sid in enumerate(sorted(new_ids)):
            if sid not in by_id:
                p.speakers.append(Speaker(id=sid, label=f"Спикер {sid[1:]}",
                                          color=free_color([x.color for x in p.speakers])))
        used = {s.speaker_id for s in p.segments}
        for sp in p.speakers:
            if sp.id not in used:
                sp.merged_into = sp.merged_into or "—"
        p.stage = Stage.REVIEW

    store.update(job_id, mutate)
    return {"speakers": len(summary)}


async def notify_approved(proj: Project) -> None:
    """Сообщает боту, что проект утверждён и его можно рендерить.

    Студия может работать отдельным процессом, поэтому связь идёт через
    файл-флаг и очередь бота: прямой вызов в чужой процесс невозможен.
    """
    marker = store.job_dir(proj.job_id) / "approved.flag"
    try:
        marker.write_text(proj.updated_at, encoding="utf-8")
    except OSError:
        log.exception("Не удалось поставить отметку об утверждении")

    try:
        from bot.review import enqueue_approved

        await enqueue_approved(proj.job_id)
    except Exception:  # noqa: BLE001 — бот может быть не запущен
        log.info("Бот не отвечает: рендер начнётся, когда он подхватит отметку")
