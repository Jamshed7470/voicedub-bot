"""Чтение и запись project.json.

Запись атомарная: сначала во временный файл, потом переименование. Обрыв
питания посреди сохранения не должен оставлять пользователя с обрезанным
JSON — восстанавливать правки будет неоткуда.

Версия увеличивается при каждой записи. Клиент, приславший правку со
старой версией, получает конфликт, а не тихую потерю чужих изменений.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path

from core.config import JOBS_DIR
from project.schema import Project, Stage, color_for, now_iso

log = logging.getLogger(__name__)

PROJECT_FILE = "project.json"
_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()


class VersionConflict(RuntimeError):
    """Проект изменился между чтением и записью."""

    def __init__(self, expected: int, actual: int):
        super().__init__(f"версия проекта {actual}, а правка сделана для {expected}")
        self.expected = expected
        self.actual = actual


class ProjectNotFound(FileNotFoundError):
    pass


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def project_path(job_id: str) -> Path:
    return job_dir(job_id) / PROJECT_FILE


def _lock(job_id: str) -> threading.RLock:
    with _locks_guard:
        if job_id not in _locks:
            _locks[job_id] = threading.RLock()
        return _locks[job_id]


# ---------------------------------------------------------------- чтение

def exists(job_id: str) -> bool:
    return project_path(job_id).exists()


def load(job_id: str) -> Project:
    path = project_path(job_id)
    if not path.exists():
        raise ProjectNotFound(f"проект {job_id} не найден")
    with _lock(job_id):
        raw = json.loads(path.read_text(encoding="utf-8"))
    return Project.model_validate(raw)


def load_or_none(job_id: str) -> Project | None:
    try:
        return load(job_id)
    except (ProjectNotFound, ValueError):
        return None


# ---------------------------------------------------------------- запись

def save(proj: Project, expected_version: int | None = None) -> Project:
    """Атомарно сохраняет проект, увеличивая версию.

    expected_version — оптимистичная блокировка: если на диске уже другая
    версия, правка отклоняется.
    """
    path = project_path(proj.job_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    with _lock(proj.job_id):
        if expected_version is not None and path.exists():
            try:
                current = json.loads(path.read_text(encoding="utf-8")).get("version", 0)
            except (OSError, ValueError):
                current = expected_version
            if current != expected_version:
                raise VersionConflict(expected_version, current)

        proj.version += 1
        proj.updated_at = now_iso()
        data = proj.model_dump(mode="json")

        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    return proj


def update(job_id: str, mutate, expected_version: int | None = None) -> Project:
    """Читает, применяет функцию, сохраняет — под одним замком.

    Так правка не может «потеряться» между чтением и записью внутри
    процесса; между процессами страхует expected_version.
    """
    with _lock(job_id):
        proj = load(job_id)
        if expected_version is not None and proj.version != expected_version:
            raise VersionConflict(expected_version, proj.version)
        mutate(proj)
        proj.recompute_stats()
        return save(proj)


# ---------------------------------------------------------------- создание

def create(job_id: str, owner_telegram_id: int = 0, chat_id: int = 0,
           **fields) -> Project:
    proj = Project(job_id=job_id, owner_telegram_id=owner_telegram_id,
                   chat_id=chat_id, **fields)
    return save(proj)


def from_pipeline(job_id: str, segments: list[dict], speakers: dict,
                  lang_src: str, lang_tgt: str, settings: dict | None = None,
                  source: dict | None = None, owner_telegram_id: int = 0,
                  chat_id: int = 0, stage: Stage = Stage.TRANSLATED) -> Project:
    """Собирает project.json из внутренних структур пайплайна.

    Пайплайн работает со словарями (так исторически устроены transcript.json
    и speakers.json); студии нужна типизированная модель. Конвертация живёт
    здесь, в одном месте, а не расползается по вызывающим.
    """
    from project.schema import (Reference, Segment, Speaker, SpeakerStats,
                                SynthInfo, Voice)

    existing = load_or_none(job_id)
    proj = existing or Project(job_id=job_id, owner_telegram_id=owner_telegram_id,
                               chat_id=chat_id)
    proj.lang_src, proj.lang_tgt = lang_src, lang_tgt
    proj.stage = stage
    if settings:
        proj.settings = proj.settings.model_copy(update={
            k: v for k, v in settings.items()
            if k in proj.settings.model_fields})
    if source:
        proj.source = proj.source.model_copy(update=source)

    # правки пользователя имеют приоритет над тем, что насчитал пайплайн
    edits_speaker = {s.id: s for s in proj.speakers}
    edits_segment = {s.id: s for s in proj.segments}

    ordered = sorted(speakers.items(),
                     key=lambda kv: kv[1].get("first_sec", 0.0))
    new_speakers: list[Speaker] = []
    for idx, (sid, rec) in enumerate(ordered):
        old = edits_speaker.get(sid)
        ref = rec.get("reference") or {}
        voice_rec = rec.get("voice") or {}
        speaker = Speaker(
            id=sid,
            label=rec.get("label") or f"Спикер {sid[1:]}",
            name=old.name if old else None,
            role=rec.get("role"),
            gender=(old.gender if old and old.gender_edited_by_user
                    else rec.get("gender", "unknown")),
            gender_confidence=float(rec.get("gender_confidence", 0.0)),
            gender_edited_by_user=bool(old.gender_edited_by_user) if old else False,
            age=rec.get("age", "adult"),
            color=old.color if old else color_for(idx),
            centroid_path=f"speakers/{sid}/centroid.npy",
            reference=Reference(
                path=ref.get("path"), clean_sec=float(ref.get("clean_sec", 0.0)),
                snr_db=float(ref.get("snr_db", 0.0)),
                score=float(ref.get("score", 0.0)),
                clone_allowed=bool(ref.get("clone_allowed", False)),
                best_samples=list(ref.get("best_samples", [])),
            ),
            voice=(old.voice if old and old.voice.edited_by_user else Voice(
                mode=voice_rec.get("mode", "clone"),
                preset_id=voice_rec.get("preset_id"),
                preset_name=rec.get("bank_voice"),
                profile_path=voice_rec.get("profile_path"),
                identity_path=voice_rec.get("identity_path"),
                locked=bool(voice_rec.get("locked", False)),
                casting_candidates=voice_rec.get("casting_candidates", []),
            )),
            merged_from=list(rec.get("merged_from", [])),
            notes=old.notes if old else "",
        )
        new_speakers.append(speaker)
    proj.speakers = new_speakers

    new_segments: list[Segment] = []
    for seg in segments:
        old = edits_segment.get(int(seg["id"]))
        edited = old.edited_by_user if old else None
        speaker_id = seg.get("speaker") or seg.get("speaker_id") or ""
        if edited and "speaker_id" in edited.fields and old:
            speaker_id = old.speaker_id      # ручное назначение сильнее пересчёта
        text_tgt = seg.get("text", "")
        if edited and "text_tgt" in edited.fields and old:
            text_tgt = old.text_tgt

        new_segments.append(Segment(
            id=int(seg["id"]), start=float(seg["start"]), end=float(seg["end"]),
            speaker_id=speaker_id,
            speaker_confidence=float(seg.get("speaker_confidence", 0.0)),
            speaker_margin=float(seg.get("speaker_margin", 0.0)),
            overlap=bool(seg.get("overlap", False)),
            overlap_with=list(seg.get("overlap_with", [])),
            text_src=seg.get("text_src", "") or seg.get("original", ""),
            text_tgt=text_tgt,
            text_tts=seg.get("text_tts", "") or text_tgt,
            emotion=seg.get("emotion", "neutral"),
            events=list(seg.get("events", [])),
            budget_chars=int(seg.get("budget_chars", 0)),
            over_budget=bool(seg.get("over_budget", False)),
            voice_override=old.voice_override if old else None,
            synth=SynthInfo(**seg["synth"]) if isinstance(seg.get("synth"), dict)
            else (old.synth if old else SynthInfo()),
            flags=list(seg.get("flags", [])),
            edited_by_user=edited or Segment.model_fields["edited_by_user"].default_factory(),
        ))
    proj.segments = new_segments
    proj.recompute_stats()
    return save(proj)


def save_render_results(job_id: str, segments: list[dict], stats,
                        stage: Stage = Stage.DONE,
                        result_path: str | None = None) -> Project | None:
    """Переносит результаты озвучки в проект: статус QC и карту голосов.

    Без этого шага студия после рендера показывает реплики как «ещё не
    озвучены»: она читает только project.json, а рендер работает со
    словарями пайплайна. Тогда фильтр «Тембр не совпал» ничего не находит,
    и пересинтезировать проблемные реплики человеку нечем (INV-4).
    """
    from project.schema import IdentityReport, SynthInfo

    if not exists(job_id):
        return None
    by_id = {int(seg["id"]): seg for seg in segments}

    def mutate(proj: Project) -> None:
        for item in proj.segments:
            src = by_id.get(item.id)
            if not src:
                continue
            synth = src.get("synth")
            if isinstance(synth, dict):
                item.synth = SynthInfo(**{k: v for k, v in synth.items()
                                          if k in SynthInfo.model_fields})
            flags = set(item.flags) | set(src.get("flags") or [])
            item.flags = sorted(flags)
            if src.get("text_tts"):
                item.text_tts = src["text_tts"]

        report = getattr(stats, "per_speaker_report", None) or {}
        proj.qc.identity_report = {
            sid: IdentityReport(**rec) for sid, rec in report.items()}
        proj.qc.overall_identity = float(getattr(stats, "overall_identity", 0.0))
        proj.stage = stage
        if result_path:
            proj.result_path = result_path

    try:
        return update(job_id, mutate)
    except (ProjectNotFound, VersionConflict):
        log.warning("Итоги рендера не записаны в проект %s", job_id)
        return None


# ---------------------------------------------------------------- обслуживание

def purge(keep_media_days: int = 7, keep_report_days: int = 30) -> int:
    """Удаляет старые задачи: сначала тяжёлые файлы, отчёт живёт дольше.

    Медиа занимают гигабайты, а project.json и report.md — килобайты, и
    именно они нужны, чтобы студия могла открыть завершённый проект.
    """
    import shutil
    import time

    if not JOBS_DIR.exists():
        return 0
    now = time.time()
    freed = 0
    keep_light = {PROJECT_FILE, "report.md", "speakers.json",
                  "speaker_registry.json", "translated.json", "transcript.json"}

    for entry in JOBS_DIR.iterdir():
        if not entry.is_dir():
            continue
        age_days = (now - entry.stat().st_mtime) / 86400
        if age_days > keep_report_days:
            shutil.rmtree(entry, ignore_errors=True)
            freed += 1
        elif age_days > keep_media_days:
            for item in entry.iterdir():
                if item.name in keep_light or item.name == "speakers":
                    continue
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
    return freed
