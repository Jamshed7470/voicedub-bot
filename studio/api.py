"""REST API студии.

Правила, общие для всех обработчиков:

* доступ только по подписанной ссылке владельца проекта (401 иначе);
* любая правка требует заголовок If-Match с версией проекта — при
  расхождении 409, а не молчаливая перезапись чужих изменений;
* каждая правка помечается edited_by_user и попадает в журнал операций,
  чтобы пережить повторный рендер и повторную диаризацию (INV-3);
* рендер невозможен, пока проект на стадии REVIEW и не нажато
  «Утвердить» (INV-5).
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from core.config import load_config
from core.errors import UserError
from project import history, store
from project.schema import Project, Segment, Stage, VoiceOverride, color_for
from studio import auth, preview
from studio.ws import hub

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


# ---------------------------------------------------------------- доступ

def _token_from(request: Request, authorization: str | None,
                t: str | None) -> str:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    if t:
        return t
    raise HTTPException(401, "Нужна ссылка на студию из бота")


async def require_project(
    job_id: str,
    request: Request,
    authorization: str | None = Header(None),
    t: str | None = Query(None),
) -> Project:
    """Проверяет токен и отдаёт проект. Общая зависимость всех обработчиков."""
    cfg = load_config()
    token = _token_from(request, authorization, t)
    try:
        auth.check(token, job_id, cfg.studio_secret, cfg.studio_link_ttl_h)
    except auth.AuthError as e:
        raise HTTPException(401, str(e)) from e
    try:
        return store.load(job_id)
    except store.ProjectNotFound as e:
        raise HTTPException(
            404, "Проект не найден: возможно, срок хранения истёк. "
                 "Пришлите видео заново.") from e


def _check_version(proj: Project, if_match: str | None) -> int:
    """If-Match обязателен для правок: без него клиент затрёт чужие изменения."""
    if if_match is None:
        raise HTTPException(428, "Нужен заголовок If-Match с версией проекта")
    try:
        version = int(str(if_match).strip('"'))
    except ValueError as e:
        raise HTTPException(400, "If-Match должен быть числом версии") from e
    if version != proj.version:
        raise HTTPException(409, {
            "error": "Проект изменился в другом окне",
            "your_version": version, "current_version": proj.version,
        })
    return version


def _apply(job_id: str, version: int, mutate, op: str) -> Project:
    try:
        return store.update(job_id, mutate, expected_version=version)
    except store.VersionConflict as e:
        raise HTTPException(409, {"error": "Проект изменился в другом окне",
                                  "current_version": e.actual}) from e
    except UserError as e:
        raise HTTPException(400, str(e)) from e


# ---------------------------------------------------------------- проект

@router.get("/projects/{job_id}")
async def get_project(proj: Project = Depends(require_project)):
    data = proj.model_dump(mode="json")
    data["warnings_count"] = proj.warnings_count()
    data["media_available"] = bool(
        proj.source.video_path and Path(proj.source.video_path).exists())
    return data


@router.get("/projects/{job_id}/export/project.json")
async def export_project(proj: Project = Depends(require_project)):
    return FileResponse(store.project_path(proj.job_id),
                        filename=f"{proj.job_id}.json")


@router.get("/projects/{job_id}/export/srt", response_class=PlainTextResponse)
async def export_srt(proj: Project = Depends(require_project)):
    def ts(value: float) -> str:
        h, rest = divmod(max(0.0, value), 3600)
        m, s = divmod(rest, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(s % 1 * 1000):03d}"

    lines = []
    for i, seg in enumerate(sorted(proj.segments, key=lambda s: s.start), 1):
        lines += [str(i), f"{ts(seg.start)} --> {ts(seg.end)}",
                  seg.text_tgt or seg.text_src, ""]
    return "\n".join(lines)


@router.get("/projects/{job_id}/report", response_class=PlainTextResponse)
async def get_report(proj: Project = Depends(require_project)):
    path = store.job_dir(proj.job_id) / "report.md"
    if not path.exists():
        raise HTTPException(404, "Отчёт появится после рендера")
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------- медиа

def _media(path: str | None, what: str) -> FileResponse:
    if not path or not Path(path).exists():
        raise HTTPException(
            404, f"{what} больше нет на диске: медиа задачи удаляются через "
                 "7 дней. Отчёт и правки доступны, превью — нет.")
    # FileResponse сам обрабатывает Range-запросы, без них видео не
    # перематывается, а грузится целиком с начала
    return FileResponse(path)


@router.get("/projects/{job_id}/media/video")
async def media_video(proj: Project = Depends(require_project)):
    return _media(proj.source.video_path, "Видео")


@router.get("/projects/{job_id}/media/vocals")
async def media_vocals(proj: Project = Depends(require_project)):
    return _media(proj.source.vocals_path, "Дорожка голоса")


@router.get("/projects/{job_id}/media/background")
async def media_background(proj: Project = Depends(require_project)):
    return _media(proj.source.background_path, "Фоновая дорожка")


@router.get("/projects/{job_id}/segments/{sid}/original.wav")
async def segment_original(sid: int, proj: Project = Depends(require_project)):
    """Вырезка оригинального звука реплики — для сверки «кто это сказал»."""
    from core.media import cut_fragment

    seg = proj.segment(sid)
    if seg is None:
        raise HTTPException(404, "Реплика не найдена")
    src = proj.source.vocals_path
    if not src or not Path(src).exists():
        raise HTTPException(404, "Дорожка голоса удалена вместе с медиа")

    out = preview.preview_dir(proj.job_id) / f"orig_{sid}.wav"
    if not out.exists():
        cut_fragment(src, out, seg.start, seg.end, sr=24000, mono=True)
    return FileResponse(out)


@router.get("/projects/{job_id}/segments/{sid}/synth.wav")
async def segment_synth(sid: int, proj: Project = Depends(require_project)):
    seg = proj.segment(sid)
    if seg is None:
        raise HTTPException(404, "Реплика не найдена")
    path = seg.synth.path
    if not path or not Path(path).exists():
        raise HTTPException(404, "Эта реплика ещё не озвучена")
    return FileResponse(path)


@router.get("/projects/{job_id}/speakers/{spk}/samples/{n}.wav")
async def speaker_sample(spk: str, n: int, proj: Project = Depends(require_project)):
    """Один из трёх лучших оригинальных фрагментов речи спикера."""
    from core.media import cut_fragment

    speaker = proj.speaker(spk)
    if speaker is None:
        raise HTTPException(404, "Спикер не найден")
    ids = speaker.reference.best_samples
    if n < 0 or n >= len(ids):
        raise HTTPException(404, "У этого спикера нет такого образца")
    seg = proj.segment(ids[n])
    src = proj.source.vocals_path
    if seg is None or not src or not Path(src).exists():
        raise HTTPException(404, "Образец недоступен")

    out = preview.preview_dir(proj.job_id) / f"sample_{spk}_{n}.wav"
    if not out.exists():
        cut_fragment(src, out, seg.start, seg.end, sr=24000, mono=True)
    return FileResponse(out)


@router.get("/projects/{job_id}/speakers/{spk}/reference.wav")
async def speaker_reference(spk: str, proj: Project = Depends(require_project)):
    speaker = proj.speaker(spk)
    if speaker is None or not speaker.reference.path:
        raise HTTPException(404, "Референс не собран")
    return _media(speaker.reference.path, "Референс голоса")


# ---------------------------------------------------------------- банк

@router.get("/voices")
async def list_voices():
    from voices.bank import get_bank

    return {"voices": [v.to_dict() for v in get_bank().all()]}


@router.get("/voices/{voice_id}/sample.wav")
async def voice_sample(voice_id: str):
    from voices.bank import get_bank

    voice = get_bank().get(voice_id)
    if voice is None or not voice.sample_path or not Path(voice.sample_path).exists():
        raise HTTPException(404, "У голоса нет образца")
    return FileResponse(voice.sample_path)


# ---------------------------------------------------------------- правки

class SegmentPatch(BaseModel):
    speaker_id: str | None = None
    text_tgt: str | None = None
    voice_override: VoiceOverride | None = None
    start: float | None = None
    end: float | None = None


@router.patch("/projects/{job_id}/segments/{sid}")
async def patch_segment(sid: int, patch: SegmentPatch,
                        if_match: str | None = Header(None, alias="If-Match"),
                        proj: Project = Depends(require_project)):
    version = _check_version(proj, if_match)

    def mutate(p: Project) -> None:
        seg = p.segment(sid)
        if seg is None:
            raise UserError("Реплика не найдена")
        before = {"segments": history.snapshot_segments(p, [sid])}

        if patch.speaker_id is not None:
            if not p.speaker(patch.speaker_id):
                raise UserError(f"Спикера {patch.speaker_id} нет в проекте")
            seg.speaker_id = patch.speaker_id
            seg.edited_by_user.touch("speaker_id")
            # человек решил вопрос, из-за которого стоял флаг
            seg.flags = [f for f in seg.flags
                         if f not in ("low_speaker_conf", "suspicious_isolated")]
        if patch.text_tgt is not None:
            seg.text_tgt = patch.text_tgt
            seg.text_tts = patch.text_tgt
            seg.edited_by_user.touch("text_tgt")
            # текст изменился — прежняя озвучка больше не соответствует
            seg.synth.status = "pending"
        if patch.voice_override is not None:
            seg.voice_override = patch.voice_override
            seg.edited_by_user.touch("voice_override")
            seg.synth.status = "pending"
        if patch.start is not None:
            seg.start = float(patch.start)
        if patch.end is not None:
            seg.end = float(patch.end)

        history.record(p, "patch_segment", before,
                       {"segments": [seg.model_dump(mode="json")]})

    updated = _apply(proj.job_id, version, mutate, "patch_segment")
    await hub.qc_updated(proj.job_id, updated.warnings_count())
    return updated.segment(sid).model_dump(mode="json")


class BulkPatch(BaseModel):
    segment_ids: list[int]
    speaker_id: str | None = None
    voice_override: VoiceOverride | None = None
    clear_flags: bool = False


@router.post("/projects/{job_id}/segments/bulk")
async def bulk_patch(body: BulkPatch,
                     if_match: str | None = Header(None, alias="If-Match"),
                     proj: Project = Depends(require_project)):
    version = _check_version(proj, if_match)

    def mutate(p: Project) -> None:
        before = {"segments": history.snapshot_segments(p, body.segment_ids)}
        targets = set(body.segment_ids)
        if body.speaker_id and not p.speaker(body.speaker_id):
            raise UserError(f"Спикера {body.speaker_id} нет в проекте")
        for seg in p.segments:
            if seg.id not in targets:
                continue
            if body.speaker_id:
                seg.speaker_id = body.speaker_id
                seg.edited_by_user.touch("speaker_id")
            if body.voice_override:
                seg.voice_override = body.voice_override
                seg.edited_by_user.touch("voice_override")
                seg.synth.status = "pending"
            if body.clear_flags or body.speaker_id:
                seg.flags = [f for f in seg.flags
                             if f not in ("low_speaker_conf", "suspicious_isolated")]
        history.record(p, "bulk_patch", before, {"count": len(targets)})

    updated = _apply(proj.job_id, version, mutate, "bulk_patch")
    await hub.qc_updated(proj.job_id, updated.warnings_count())
    return {"updated": len(body.segment_ids), "version": updated.version}


class SplitBody(BaseModel):
    at_sec: float


@router.post("/projects/{job_id}/segments/{sid}/split")
async def split_segment(sid: int, body: SplitBody,
                        if_match: str | None = Header(None, alias="If-Match"),
                        proj: Project = Depends(require_project)):
    """Делит реплику пополам по границе ближайшего слова."""
    version = _check_version(proj, if_match)
    created: list[int] = []

    def mutate(p: Project) -> None:
        seg = p.segment(sid)
        if seg is None:
            raise UserError("Реплика не найдена")
        if not (seg.start + 0.15 < body.at_sec < seg.end - 0.15):
            raise UserError("Точка деления должна быть внутри реплики")

        before = {"segments": history.snapshot_segments(p, [sid])}
        left_text, right_text = _split_text_at(seg, body.at_sec)
        new_id = max((s.id for s in p.segments), default=0) + 1
        created.append(new_id)

        right = Segment(
            id=new_id, start=float(body.at_sec), end=seg.end,
            speaker_id=seg.speaker_id,
            speaker_confidence=seg.speaker_confidence,
            speaker_margin=seg.speaker_margin,
            text_src="", text_tgt=right_text, text_tts=right_text,
            emotion=seg.emotion, voice_override=seg.voice_override,
            flags=list(seg.flags),
        )
        right.edited_by_user.touch("split")
        seg.end = float(body.at_sec)
        seg.text_tgt = seg.text_tts = left_text
        seg.synth.status = right.synth.status = "pending"
        seg.edited_by_user.touch("split")

        p.segments.append(right)
        p.segments.sort(key=lambda s: (s.start, s.id))
        history.record(p, "split_segment",
                       {**before, "segments_added": [new_id]},
                       {"segments": [seg.model_dump(mode="json"),
                                     right.model_dump(mode="json")]})

    updated = _apply(proj.job_id, version, mutate, "split_segment")
    return [updated.segment(sid).model_dump(mode="json"),
            updated.segment(created[0]).model_dump(mode="json")]


def _split_text_at(seg: Segment, at_sec: float) -> tuple[str, str]:
    """Делит текст пропорционально положению точки внутри реплики.

    Точных пословных таймингов у переведённого текста нет — перевод не
    выравнивается по словам оригинала. Поэтому делим по ближайшей границе
    слова к пропорциональной позиции: это предсказуемо и всегда даёт
    осмысленные куски, а точную правку человек сделает руками.
    """
    text = (seg.text_tgt or "").strip()
    if not text:
        return "", ""
    ratio = (at_sec - seg.start) / max(1e-6, seg.end - seg.start)
    words = text.split()
    cut = max(1, min(len(words) - 1, round(len(words) * ratio))) if len(words) > 1 else 1
    return " ".join(words[:cut]), " ".join(words[cut:])


class MergeBody(BaseModel):
    segment_ids: list[int]


@router.post("/projects/{job_id}/segments/merge")
async def merge_segments(body: MergeBody,
                         if_match: str | None = Header(None, alias="If-Match"),
                         proj: Project = Depends(require_project)):
    version = _check_version(proj, if_match)
    if len(body.segment_ids) < 2:
        raise HTTPException(400, "Для склейки нужно минимум две реплики")
    kept = min(body.segment_ids)

    def mutate(p: Project) -> None:
        chosen = sorted((s for s in p.segments if s.id in set(body.segment_ids)),
                        key=lambda s: s.start)
        if len(chosen) != len(body.segment_ids):
            raise UserError("Не все реплики найдены")
        if len({s.speaker_id for s in chosen}) > 1:
            raise UserError("Склеивать можно только реплики одного спикера")

        before = {"segments": history.snapshot_segments(p, body.segment_ids)}
        first = chosen[0]
        first.end = max(s.end for s in chosen)
        first.text_tgt = " ".join(s.text_tgt for s in chosen if s.text_tgt).strip()
        first.text_src = " ".join(s.text_src for s in chosen if s.text_src).strip()
        first.text_tts = first.text_tgt
        first.synth.status = "pending"
        first.edited_by_user.touch("merge")

        drop = {s.id for s in chosen if s.id != first.id}
        p.segments = [s for s in p.segments if s.id not in drop]
        history.record(p, "merge_segments",
                       {**before, "segments_removed":
                        [s for s in before["segments"] if s["id"] in drop]},
                       {"segments": [first.model_dump(mode="json")]})

    updated = _apply(proj.job_id, version, mutate, "merge_segments")
    merged = next((s for s in updated.segments if s.id in set(body.segment_ids)), None)
    return merged.model_dump(mode="json") if merged else {"id": kept}


# ---------------------------------------------------------------- спикеры

class SpeakerPatch(BaseModel):
    name: str | None = None
    gender: str | None = None
    voice: dict | None = None
    notes: str | None = None


@router.patch("/projects/{job_id}/speakers/{spk}")
async def patch_speaker(spk: str, patch: SpeakerPatch,
                        if_match: str | None = Header(None, alias="If-Match"),
                        proj: Project = Depends(require_project)):
    version = _check_version(proj, if_match)

    def mutate(p: Project) -> None:
        speaker = p.speaker(spk)
        if speaker is None:
            raise UserError("Спикер не найден")
        before = {"speakers": history.snapshot_speakers(p, [spk])}

        if patch.name is not None:
            speaker.name = patch.name.strip() or None
        if patch.gender is not None:
            if patch.gender not in ("male", "female", "unknown"):
                raise UserError("Пол может быть male, female или unknown")
            speaker.gender = patch.gender
            speaker.gender_edited_by_user = True
        if patch.notes is not None:
            speaker.notes = patch.notes
        if patch.voice is not None:
            mode = patch.voice.get("mode", speaker.voice.mode)
            preset_id = patch.voice.get("preset_id")
            if mode == "clone" and not speaker.reference.clone_allowed:
                # спецификация разрешает принудительный клон, но с честным
                # предупреждением: короткий референс звучит нестабильно
                log.warning("Спикер %s: клон включён принудительно при %.1f с "
                            "чистой речи", spk, speaker.reference.clean_sec)
            if mode == "preset":
                from voices.bank import get_bank

                voice = get_bank().get(preset_id or "")
                if voice is None:
                    raise UserError(f"Голоса {preset_id} нет в банке")
                speaker.voice.preset_name = voice.display_name
            speaker.voice.mode = mode
            speaker.voice.preset_id = preset_id if mode == "preset" else None
            speaker.voice.edited_by_user = True
            speaker.voice.locked = True
            for seg in p.segments:
                if seg.speaker_id == spk:
                    seg.synth.status = "pending"

        history.record(p, "patch_speaker", before,
                       {"speakers": [speaker.model_dump(mode="json")]})

    updated = _apply(proj.job_id, version, mutate, "patch_speaker")
    return updated.speaker(spk).model_dump(mode="json")


class SpeakerMerge(BaseModel):
    from_id: str
    into_id: str


@router.post("/projects/{job_id}/speakers/merge")
async def merge_speakers(body: SpeakerMerge,
                         if_match: str | None = Header(None, alias="If-Match"),
                         proj: Project = Depends(require_project)):
    """Объединяет спикеров: все реплики уходят в into_id."""
    version = _check_version(proj, if_match)
    if body.from_id == body.into_id:
        raise HTTPException(400, "Нельзя объединить спикера с самим собой")

    moved = 0

    def mutate(p: Project) -> None:
        nonlocal moved
        src, dst = p.speaker(body.from_id), p.speaker(body.into_id)
        if src is None or dst is None:
            raise UserError("Спикер не найден")

        before = {"speakers": history.snapshot_speakers(p, [body.from_id, body.into_id]),
                  "segments": [s.model_dump(mode="json") for s in p.segments
                               if s.speaker_id == body.from_id]}
        for seg in p.segments:
            if seg.speaker_id == body.from_id:
                seg.speaker_id = body.into_id
                seg.edited_by_user.touch("speaker_id")
                seg.synth.status = "pending"
                moved += 1

        if src.gender != dst.gender and src.gender != "unknown":
            log.warning("Объединение %s → %s: разный пол (%s и %s), "
                        "берётся пол принимающего", body.from_id, body.into_id,
                        src.gender, dst.gender)
        src.merged_into = body.into_id
        dst.merged_from = sorted(set(dst.merged_from + [body.from_id] + src.merged_from))
        # профиль принимающего собран по другому набору реплик — он больше
        # не соответствует своему спикеру и должен быть пересобран
        dst.voice.locked = False
        history.record(p, "merge_speakers", before, {"moved": moved})

    updated = _apply(proj.job_id, version, mutate, "merge_speakers")
    return {"moved_segments": moved, "version": updated.version,
            "speaker": updated.speaker(body.into_id).model_dump(mode="json"),
            "rebuild_required": True}


@router.post("/projects/{job_id}/speakers/{spk}/rebuild-profile")
async def rebuild_profile(spk: str, request: Request,
                          if_match: str | None = Header(None, alias="If-Match"),
                          proj: Project = Depends(require_project)):
    """Пересобирает референс и профиль голоса спикера."""
    version = _check_version(proj, if_match)
    cfg = load_config()

    async def job():
        import asyncio

        def work():
            from studio.tasks import rebuild_speaker_profile

            return rebuild_speaker_profile(proj.job_id, spk, cfg)

        try:
            await asyncio.to_thread(work)
            fresh = store.load(proj.job_id)
            await hub.publish(proj.job_id, {
                "type": "profile_rebuilt", "speaker_id": spk,
                "version": fresh.version})
        except Exception as e:  # noqa: BLE001
            log.exception("Пересборка профиля %s не удалась", spk)
            await hub.publish(proj.job_id, {"type": "error", "message": str(e)})

    await preview.queue.submit(proj.job_id, job)
    return {"status": "accepted", "speaker_id": spk, "version": version}


# ---------------------------------------------------------------- превью

class RangeBody(BaseModel):
    start_sec: float
    end_sec: float


@router.post("/projects/{job_id}/preview/segment/{sid}", status_code=202)
async def preview_segment(sid: int, proj: Project = Depends(require_project)):
    cfg = load_config()

    async def job():
        import asyncio

        try:
            path = await asyncio.to_thread(preview.synth_segment_sync, proj, sid, cfg)
            store.save(proj)
            await hub.preview_ready(
                proj.job_id, "segment", str(sid),
                f"/api/projects/{proj.job_id}/segments/{sid}/synth.wav")
            return path
        except Exception as e:  # noqa: BLE001
            log.exception("Превью реплики %s не удалось", sid)
            await hub.preview_failed(proj.job_id, str(sid), str(e))

    await preview.queue.submit(proj.job_id, job)
    return {"status": "accepted", "segment_id": sid}


@router.post("/projects/{job_id}/preview/range", status_code=202)
async def preview_range(body: RangeBody, proj: Project = Depends(require_project)):
    cfg = load_config()

    async def job():
        import asyncio

        try:
            path = await asyncio.to_thread(preview.render_range_sync, proj,
                                           body.start_sec, body.end_sec, cfg)
            await hub.preview_ready(
                proj.job_id, "range", f"{body.start_sec:.1f}",
                f"/api/projects/{proj.job_id}/preview/file/{Path(path).name}")
            return path
        except Exception as e:  # noqa: BLE001
            log.exception("Превью отрывка не удалось")
            await hub.preview_failed(proj.job_id, "range", str(e))

    await preview.queue.submit(proj.job_id, job)
    return {"status": "accepted",
            "range": [body.start_sec, min(body.end_sec,
                                          body.start_sec + preview.MAX_RANGE_SEC)]}


@router.get("/projects/{job_id}/preview/file/{name}")
async def preview_file(name: str, proj: Project = Depends(require_project)):
    path = preview.preview_dir(proj.job_id) / Path(name).name
    if not path.exists():
        raise HTTPException(404, "Превью не найдено")
    return FileResponse(path)


# ---------------------------------------------------------------- стадии

class RediarizeBody(BaseModel):
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None


@router.post("/projects/{job_id}/rediarize", status_code=202)
async def rediarize(body: RediarizeBody,
                    if_match: str | None = Header(None, alias="If-Match"),
                    proj: Project = Depends(require_project)):
    """Пересчитывает спикеров. Правки пользователя переносятся по реестру."""
    _check_version(proj, if_match)
    cfg = load_config()

    async def job():
        import asyncio

        def work():
            from studio.tasks import rediarize_project

            return rediarize_project(proj.job_id, cfg, body.num_speakers,
                                     body.min_speakers, body.max_speakers)

        try:
            await asyncio.to_thread(work)
            fresh = store.load(proj.job_id)
            await hub.publish(proj.job_id, {
                "type": "rediarized", "speakers": len(fresh.active_speakers()),
                "version": fresh.version})
        except Exception as e:  # noqa: BLE001
            log.exception("Повторная диаризация не удалась")
            await hub.publish(proj.job_id, {"type": "error", "message": str(e)})

    await preview.queue.submit(proj.job_id, job)
    return {"status": "accepted"}


@router.post("/projects/{job_id}/approve")
async def approve(if_match: str | None = Header(None, alias="If-Match"),
                  proj: Project = Depends(require_project)):
    """Утверждение: только отсюда проект уходит в рендер (INV-5)."""
    version = _check_version(proj, if_match)
    if proj.stage not in (Stage.REVIEW, Stage.TRANSLATED, Stage.PROFILED):
        raise HTTPException(409, f"Проект на стадии «{proj.stage.value}» — "
                                 "утверждать нечего")

    def mutate(p: Project) -> None:
        p.stage = Stage.APPROVED
        history.record(p, "approve", None, {"stage": "approved"})

    updated = _apply(proj.job_id, version, mutate, "approve")
    await hub.stage_changed(proj.job_id, Stage.APPROVED.value)

    from studio.tasks import notify_approved

    await notify_approved(updated)
    return {"stage": updated.stage.value, "version": updated.version}


@router.post("/projects/{job_id}/undo")
async def undo_last(if_match: str | None = Header(None, alias="If-Match"),
                    proj: Project = Depends(require_project)):
    version = _check_version(proj, if_match)
    undone: list[str | None] = []

    def mutate(p: Project) -> None:
        undone.append(history.undo(p))

    updated = _apply(proj.job_id, version, mutate, "undo")
    if not undone or undone[0] is None:
        raise HTTPException(400, "Отменять нечего")
    return {"undone": undone[0], "version": updated.version}
