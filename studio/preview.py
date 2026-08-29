"""Превью по запросу: отдельная реплика и отрывок с фоном.

Превью считается на той же видеокарте, что и основные задачи, поэтому
запросы выстраиваются в очередь: два одновременных синтеза не ускорят
работу, а вытеснят друг друга из видеопамяти. На проект — не более одного
превью одновременно.

Результат кэшируется по содержимому запроса (текст + профиль + скорость):
пользователь в студии переслушивает одну и ту же реплику много раз, и
повторный синтез каждый раз был бы издевательством над ожиданием.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path

from core.errors import UserError
from project import store
from project.schema import Project

log = logging.getLogger(__name__)

PREVIEW_DIR = "preview"
MAX_RANGE_SEC = 90.0


class PreviewQueue:
    """Одно превью на проект за раз, общая очередь на видеокарту."""

    def __init__(self, concurrency: int = 1):
        self._gpu = asyncio.Semaphore(concurrency)
        self._per_project: dict[str, asyncio.Task] = {}

    def busy(self, job_id: str) -> bool:
        task = self._per_project.get(job_id)
        return bool(task and not task.done())

    async def submit(self, job_id: str, coro_factory) -> asyncio.Task:
        previous = self._per_project.get(job_id)
        if previous and not previous.done():
            previous.cancel()      # пользователь передумал — старое не нужно

        async def runner():
            async with self._gpu:
                return await coro_factory()

        task = asyncio.create_task(runner())
        self._per_project[job_id] = task
        return task


queue = PreviewQueue()


def cache_key(*parts) -> str:
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def preview_dir(job_id: str) -> Path:
    path = store.job_dir(job_id) / PREVIEW_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------- реплика

def synth_segment_sync(proj: Project, seg_id: int, cfg) -> Path:
    """Синтезирует одну реплику текущим голосом её спикера."""
    from identity.embeddings import get_embedder
    from synth.render import ProfileCache
    from synth.xtts_engine import get_engine
    from voices.bank import get_bank

    seg = proj.segment(seg_id)
    if seg is None:
        raise UserError(f"реплика {seg_id} не найдена")
    text = (seg.text_tts or seg.text_tgt or "").strip()
    if not text:
        raise UserError("в этой реплике нет текста для озвучки")

    speaker = proj.speaker(seg.speaker_id)
    override = seg.voice_override.model_dump() if seg.voice_override else None
    voice_id = (override or {}).get("preset_id") or (
        speaker.voice.preset_id if speaker else None)

    key = cache_key(text, seg.speaker_id, voice_id,
                    speaker.voice.mode if speaker else "clone")
    out = preview_dir(proj.job_id) / f"seg_{seg_id}_{key}.wav"
    if out.exists() and out.stat().st_size > 1000:
        return out

    speakers = {s.id: s.model_dump(mode="json") for s in proj.speakers}
    cache = ProfileCache(speakers, get_bank())
    profile = cache.get(seg.speaker_id, override)

    engine = get_engine(cfg)
    engine.synthesize(text, proj.lang_tgt, profile, out, speed=1.0,
                      seed=engine.make_seed(proj.job_id, seg_id, 0))

    # заодно считаем QC, чтобы человек сразу видел, годится ли результат
    try:
        from synth import qc as qc_mod

        result = qc_mod.check(out, text, proj.lang_tgt, profile,
                              seg.duration, get_embedder(cfg), cfg,
                              with_backcheck=False)
        seg.synth.identity_sim = result.identity_sim
        seg.synth.status = result.status
        seg.synth.path = str(out)
    except Exception:  # noqa: BLE001 — превью важнее его оценки
        log.exception("Превью: не удалось посчитать QC для реплики %s", seg_id)
    return out


# ---------------------------------------------------------------- отрывок

def render_range_sync(proj: Project, start: float, end: float, cfg) -> Path:
    """Мини-рендер отрывка: синтез реплик отрезка поверх фона.

    Нужен, чтобы услышать результат до полного рендера — на полуторачасовом
    фильме тот идёт часами, и утверждать голоса вслепую бессмысленно.
    """
    import numpy as np
    import soundfile as sf

    end = min(end, start + MAX_RANGE_SEC)
    segs = [s for s in proj.segments if s.end > start and s.start < end]
    if not segs:
        raise UserError("на этом отрезке нет речи")

    key = cache_key(start, end, proj.version)
    out = preview_dir(proj.job_id) / f"range_{key}.wav"
    if out.exists() and out.stat().st_size > 1000:
        return out

    placed = []
    for seg in segs:
        try:
            wav = synth_segment_sync(proj, seg.id, cfg)
            placed.append((seg.start - start, wav))
        except Exception:  # noqa: BLE001
            log.exception("Превью отрывка: реплика %s не синтезировалась", seg.id)
    if not placed:
        raise UserError("не удалось озвучить ни одной реплики отрезка")

    # речевая дорожка: реплики кладутся каждая на своё место
    sr = 24000
    speech = np.zeros(int((end - start) * sr) + sr, dtype=np.float32)
    for at_sec, wav in placed:
        data, file_sr = sf.read(str(wav), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if file_sr != sr:
            import librosa
            data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
        at = max(0, int(at_sec * sr))
        width = min(len(data), len(speech) - at)
        if width > 0:
            speech[at:at + width] += data[:width]

    peak = float(np.max(np.abs(speech))) or 1.0
    speech_path = preview_dir(proj.job_id) / f"speech_{key}.wav"
    sf.write(str(speech_path), speech * min(1.0, 0.89 / peak), sr)

    # фон подмешивается тише речи; полный микс с дакингом делает основной
    # рендер — здесь важно услышать голоса, а не идеальный баланс
    bg_path = Path(proj.source.background_path or "")
    if proj.settings.keep_background and bg_path.exists():
        bg_cut = _cut(bg_path, start, end, preview_dir(proj.job_id) / f"bg_{key}.wav")
        _amix(speech_path, bg_cut, out)
        speech_path.unlink(missing_ok=True)
        bg_cut.unlink(missing_ok=True)
    else:
        speech_path.replace(out)
    return out


def _amix(speech: Path, background: Path, out: Path) -> None:
    from core.media import run

    run(["ffmpeg", "-y", "-i", str(speech), "-i", str(background),
         "-filter_complex",
         "[1:a]volume=0.35[bg];[0:a][bg]amix=inputs=2:duration=first:"
         "dropout_transition=0,alimiter=limit=0.95[a]",
         "-map", "[a]", "-c:a", "pcm_s16le", str(out)], desc="превью: микс")


def _cut(src: Path, start: float, end: float, dst: Path) -> Path:
    from core.media import run

    run(["ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
         "-i", str(src), "-c:a", "pcm_s16le", str(dst)], desc="превью: фон")
    return dst
