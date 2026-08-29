"""Оркестратор пайплайна дубляжа.

Выполняет этапы последовательно для каждой задачи, пишет результаты
в data/jobs/<job_id>/ и обновляет прогресс в Telegram через reporter.
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from core import media
from core.config import JOBS_DIR, TTS_LANGUAGES
from core.errors import JobCancelled, UserError

log = logging.getLogger(__name__)

# Пользовательские этапы (номер, подпись)
STAGES = [
    "скачиваю файл",                 # 1
    "извлекаю аудио",                # 2
    "отделяю голос от фона",         # 3
    "распознаю речь",                # 4
    "распознаю спикеров",            # 5
    "анализирую эмоции и события",   # 6
    "перевожу",                      # 7
    "готовлю текст к озвучке",       # 8
    "озвучиваю",                     # 9
    "собираю результат",             # 10
]

GENDER_RU = {"male": "мужчина", "female": "женщина"}

# метка «медиа этой задачи в кэш не класть» (длинный ролик — гигабайты wav)
NO_MEDIA_CACHE = "no_media_cache"


@dataclass
class JobResult:
    kind: str                       # "video" | "audio"
    output_path: Path | None        # что отправлять в чат (None — не влезает в лимит)
    srt_path: Path
    src_lang: str
    tgt_lang: str
    speakers: dict
    elapsed_s: float
    summary: str = ""
    parts: list[Path] = field(default_factory=list)  # если не влез одним файлом
    saved_path: Path | None = None  # копия в папке готовых видео (всегда)
    saved_srt: Path | None = None


@dataclass
class PipelineHooks:
    """Связь пайплайна с ботом (все вызовы — из event loop)."""
    report: object                  # async def report(stage_idx1, label, pct)
    confirm_same_lang: object = None  # async def (src_lang) -> bool
    cancel_event: object = None     # asyncio.Event


def _check_cancel(hooks: PipelineHooks) -> None:
    if hooks.cancel_event is not None and hooks.cancel_event.is_set():
        raise JobCancelled()


async def run_job(job, bot, hooks: PipelineHooks, cfg) -> JobResult:
    """Полный цикл обработки одной задачи."""
    t0 = time.monotonic()
    job_dir = JOBS_DIR / job.id
    job_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = job_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    loop = asyncio.get_running_loop()

    async def report(stage: int, pct: int = 0) -> None:
        _check_cancel(hooks)
        await hooks.report(stage, STAGES[stage - 1], pct)

    def report_ts(stage: int):
        """Потокобезопасный колбэк прогресса для тяжёлых синхронных этапов."""
        def cb(pct: int) -> None:
            asyncio.run_coroutine_threadsafe(
                hooks.report(stage, STAGES[stage - 1], int(pct)), loop)
        return cb

    tgt_lang = job.target_lang

    # ---------- Кэш: медиа и разбор голосов могли уже считаться ----------
    from core import cache as source_cache
    spk_hint = str(job.settings.get("speakers", "auto"))
    src_key = source_cache.source_key(job.kind, job.payload)
    (job_dir / "cache_key.txt").write_text(f"{src_key}\n{spk_hint}",
                                           encoding="utf-8")
    cached_media = source_cache.media_dir(src_key)
    cached_analysis = source_cache.analysis_dir(src_key, spk_hint)
    cached_analysis = _analysis_still_valid(cached_analysis, cfg)

    if cached_media:
        # ---------- 1–3 из кэша ----------
        await hooks.report(3, "беру видео и дорожки из кэша", 100)
        source_cache.touch(cached_media)
        input_path = source_cache.input_file(cached_media)
        info = media.probe(input_path)
        analysis_wav = cached_media / "analysis.wav"
        source_wav = cached_media / "source.wav"
        vocals_wav = cached_media / "vocals.wav"
        vocals16_wav = cached_media / "vocals16.wav"
        background_wav = cached_media / "background.wav"
    elif (resumed := _resume_media(job_dir)) is not None:
        # ---------- 1–3 из папки прерванной задачи ----------
        await hooks.report(3, "беру видео и дорожки прерванной задачи", 100)
        (input_path, analysis_wav, source_wav, vocals_wav,
         vocals16_wav, background_wav) = resumed
        info = media.probe(input_path)
        log.info("Продолжаю задачу: медиа взяты из %s", job_dir)
    else:
        # ---------- 1. Скачивание ----------
        await report(1)
        if job.kind == "url":
            from core.downloader import download_url
            input_path = await asyncio.to_thread(download_url, job.payload,
                                                 job_dir, cfg)
        elif job.kind == "local":
            # файл уже на диске: копировать гигабайты незачем, читаем на месте
            input_path = Path(job.payload)
            if not input_path.exists():
                raise UserError(f"Файл не найден: {input_path}")
        else:
            from core.downloader import download_from_telegram
            input_path = job_dir / "input.bin"
            await download_from_telegram(bot, job.payload["file_id"],
                                         job.payload.get("file_size"),
                                         input_path, cfg)

        # ---------- 2. Извлечение аудио ----------
        await report(2)
        info = media.probe(input_path)
        if not info.has_audio:
            raise UserError("В этом видео нет аудиодорожки — дублировать нечего.")
        if info.duration > cfg.max_duration_s:
            raise UserError(
                f"Файл слишком длинный: {info.duration / 60:.0f} мин. "
                f"Лимит — {cfg.max_duration_s / 60:.0f} мин."
            )
        _check_disk(job_dir, info.duration, input_path, cfg)
        cache_media = _should_cache_media(info.duration, input_path, cfg, job_dir)
        if not cache_media:
            # метка для cleanup_job: медиа этого ролика в кэш не переносим
            (job_dir / NO_MEDIA_CACHE).write_text("1", encoding="utf-8")
        if info.duration > 3600:
            log.info("Длинный ролик: %.0f мин — обработка займёт заметное время",
                     info.duration / 60)
        analysis_wav, source_wav = await asyncio.to_thread(
            media.extract_audio, input_path, job_dir,
            int(cfg.y("audio", "analysis_sr", default=16000)),
            int(cfg.y("audio", "source_sr", default=44100)),
        )

        # ---------- 3. Разделение голоса и фона ----------
        await report(3)
        from core.separate import separate
        vocals_wav, background_wav = await asyncio.to_thread(
            separate, source_wav, job_dir, cfg, report_ts(3))
        vocals16_wav = job_dir / "vocals16.wav"

        # в кэш сразу: дальше задача может упасть или бота перезапустят,
        # а скачивание и Demucs — самые дорогие этапы.
        # длинные ролики не кэшируем: их дорожки — это гигабайты wav
        stored = (await asyncio.to_thread(source_cache.store_media, job_dir, src_key)
                  if cache_media else None)
        if stored:
            input_path = source_cache.input_file(stored) or input_path
            analysis_wav = stored / "analysis.wav"
            source_wav = stored / "source.wav"
            vocals_wav = stored / "vocals.wav"
            vocals16_wav = stored / "vocals16.wav"
            background_wav = stored / "background.wav"

    if cached_analysis:
        # ---------- 4–6 из кэша (то же число спикеров) ----------
        await hooks.report(6, "беру разбор голосов из кэша", 100)
        source_cache.touch(cached_analysis)
        tr = source_cache.load_transcript(cached_analysis)
        src_lang = tr["language"]
        segments = tr["segments"]
        events = tr["events"]
        profiles = source_cache.load_profiles(cached_analysis)
        _save_json(job_dir / "transcript.json", tr)

        same_lang = _norm_lang(src_lang) == _norm_lang(tgt_lang)
        if same_lang and hooks.confirm_same_lang is not None:
            proceed = await hooks.confirm_same_lang(src_lang)
            if not proceed:
                raise JobCancelled()
    else:
        # ---------- 4. Распознавание речи ----------
        await report(4)
        from core.asr import transcribe
        # по очищенному голосу: под музыку и шум Whisper глохнет, а вокал
        # после Demucs у нас уже посчитан этапом раньше
        asr_input = vocals16_wav if Path(vocals16_wav).exists() else analysis_wav
        asr_result = await asyncio.to_thread(transcribe, str(asr_input), cfg,
                                             report_ts(4))
        src_lang = asr_result.get("language") or "en"

        same_lang = _norm_lang(src_lang) == _norm_lang(tgt_lang)
        if same_lang and hooks.confirm_same_lang is not None:
            # исходный язык совпадает с целевым → предупредить и спросить
            proceed = await hooks.confirm_same_lang(src_lang)
            if not proceed:
                raise JobCancelled()

        # ---------- 5. Диаризация спикеров ----------
        await report(5)
        from core.diarize import diarize_and_assign
        # по очищенному голосу: музыка и шум смазывают голосовые отпечатки,
        # и похожие спикеры слипаются в один кластер
        segments, raw_turns = await asyncio.to_thread(
            diarize_and_assign, str(vocals16_wav), asr_result, cfg, spk_hint,
            True)
        if not segments:
            raise UserError("В этом файле не нашлось речи — нечего дублировать.")

        # Speaker Identity Engine: сводит переразбитые кластеры в реальных
        # людей. Без него один человек получал по несколько «спикеров», и
        # каждому доставался свой голос — главная жалоба на версию 1.
        await hooks.report(5, "свожу голоса спикеров", 60)
        import identity as sie

        sie_result = await asyncio.to_thread(
            sie.analyze, vocals16_wav, segments, cfg, raw_turns, job_dir,
            spk_hint)
        speaker_summary = sie_result["speakers"]
        log.info("Спикеров после сведения: %d", len(speaker_summary))

        # ---------- 6. Эмоции, события, профили спикеров ----------
        await report(6)
        import numpy as np  # noqa: F401

        from core.emotions import classify_segments
        from core.events import detect_events, mark_event_segments

        def _stage6() -> tuple[list[dict], dict]:
            import librosa

            from identity.embeddings import get_embedder
            from synth.xtts_engine import get_engine
            from voices import profiles as prof_mod

            y16, _ = librosa.load(str(vocals16_wav), sr=16000, mono=True)
            events = detect_events(y16, 16000, cfg, report_ts(6))
            mark_event_segments(segments, events, cfg)
            # эмоция нужна до сборки референса: выразительная речь в
            # референс не идёт, иначе «истеричный» тембр закрепится за
            # спикером на весь фильм
            classify_segments(vocals16_wav, segments, cfg, tmp_dir)
            profiles = prof_mod.build_all(
                job_dir, vocals_wav, vocals16_wav, segments, speaker_summary,
                cfg, get_engine(cfg), get_embedder(cfg),
                voice_mode=str(job.settings.get("voice_mode") or
                               cfg.y("casting", "default_mode", default="auto")),
                progress=report_ts(6))
            return events, profiles

        events, profiles = await asyncio.to_thread(_stage6)

        _save_json(job_dir / "transcript.json",
                   {"language": src_lang, "segments": segments, "events": events})

        # разбор голосов — тоже в кэш сразу, до перевода и озвучки
        stored = await asyncio.to_thread(source_cache.store_analysis, job_dir,
                                         src_key, spk_hint)
        if stored:
            cached_profiles = source_cache.load_profiles(stored)
            # профили переехали в кэш вместе с папкой speakers/: пути в них
            # переписаны, но проверить это надо СЕЙЧАС, а не на синтезе через
            # полчаса, когда причина уже не видна
            if source_cache.verify_profiles(cached_profiles):
                profiles = cached_profiles
            else:
                log.warning("Кэш разбора неполон — работаю с профилями задачи")

    # ---------- 7. Перевод ----------
    await report(7)
    # оригинал сохраняется ДО перевода: перевод пишется в то же поле, и без
    # этой строки в студии нечего сверять с видео — а ради сверки она и есть
    for seg in segments:
        seg.setdefault("text_src", seg.get("text", ""))
    style = job.settings.get("translation_style", "normal")
    from core.translate import get_translator
    translator = get_translator(cfg, style)

    done_translation = _load_translated(job_dir, tgt_lang, len(segments))
    if done_translation is not None:
        # задачу уже переводили в этой же папке (продолжение после остановки) —
        # десятки запросов к модели повторять незачем
        log.info("Перевод взят из папки задачи: %d реплик", len(done_translation))
        segments = done_translation
        await hooks.report(7, "беру готовый перевод", 100)
    elif not same_lang or style == "street":
        # уличный стиль применяется даже без смены языка (переозвучка со стёбом)
        await asyncio.to_thread(translator.translate_segments, segments,
                                src_lang, tgt_lang, profiles, report_ts(7))
    else:
        log.info("Языки совпадают — переозвучка без перевода")
    _save_json(job_dir / "translated.json",
               {"language": tgt_lang, "segments": segments})

    # ---------- 8. Нормализация текста ----------
    await report(8)
    from core.normalize import normalize_for_tts
    for seg in segments:
        seg["text"] = normalize_for_tts(seg["text"], tgt_lang)
        seg["text_tts"] = seg["text"]

    # ---------- 8.5. Точка проверки человеком ----------
    # Рендер полутора тысяч реплик идёт часами: ошибку в назначении голосов
    # дешевле поймать здесь, чем услышать в готовом фильме.
    proj = await _open_review(job, bot, cfg, hooks, job_dir, segments, profiles,
                              src_lang, tgt_lang,
                              media={"video_path": str(input_path),
                                     "vocals_path": str(vocals_wav),
                                     "background_path": str(background_wav),
                                     "duration_sec": float(info.duration),
                                     "file_name": input_path.name,
                                     "title": _read_title(job_dir, cached_media)})
    if proj is not None:
        segments, profiles = _apply_review(proj, segments, profiles)

    # ---------- 9. Синтез + подгонка таймингов ----------
    await report(9)
    from identity.embeddings import get_embedder
    from synth import render as render_mod
    from synth.xtts_engine import get_engine
    from voices import casting

    # кастинг: спикерам без клона (мало чистой речи) назначается голос банка,
    # причём разным спикерам — разные голоса
    bank = await asyncio.to_thread(casting.assign_voices, profiles, cfg)

    # длинные реплики сокращаются пачкой ДО синтеза: поштучные запросы к
    # переводчику из цикла на полуторачасовом фильме стоят часов
    limits = render_mod.slot_limits(segments, cfg)
    done_ids = render_mod.done_ids(job_dir / "synth")
    pending = [s for s in segments
               if not s.get("skip_tts") and s.get("text", "").strip()
               and not render_mod.reusable(job_dir / "synth" / f"seg_{s['id']}.wav",
                                           done_ids, s["id"])]
    await asyncio.to_thread(_precompress, pending, limits, translator, tgt_lang,
                            cfg, None)

    def _stage9():
        return render_mod.render_all(
            job_dir, segments, profiles, cfg, get_engine(cfg),
            get_embedder(cfg), tgt_lang, job.id, translator=translator,
            bank=bank, progress=report_ts(9),
            cancel_event=hooks.cancel_event, lang_src=src_lang)

    placed, render_stats = await asyncio.to_thread(_stage9)
    _save_json(job_dir / "speakers.json", profiles)
    render_mod.write_report(job_dir, profiles, render_stats, src_lang, tgt_lang)
    log.info("Стабильность голосов: %.2f", render_stats.overall_identity)

    # итоги озвучки — в проект: иначе студия покажет реплики как
    # неозвученные, и пересинтезировать проблемные будет нечем (INV-4)
    from project import store as project_store
    from project.schema import Stage as ProjectStage

    await asyncio.to_thread(project_store.save_render_results, job.id, segments,
                            render_stats, ProjectStage.MIXING)

    # ---------- 10. Микс и сборка ----------
    await report(10)
    from core.mixer import build_mix
    from core.mux import (TooLongForLimit, compress_to_limit, encode_audio_only,
                          mux_video, write_srt)

    keep_bg = job.settings.get("keep_background", True)
    keep_orig = job.settings.get("keep_original_track", False)
    title = _read_title(job_dir, cached_media)
    if not title and job.kind == "local":
        title = Path(job.payload).stem

    def _stage10() -> tuple[Path | None, Path, Path, Path, list[Path]]:
        """Возвращает (что отправить | None, файл в папке, srt, srt в папке, части)."""
        dubbed = build_mix(job_dir, background_wav, vocals_wav, placed,
                           events, cfg, keep_background=keep_bg)
        srt = write_srt(segments, job_dir / "subtitles.srt")
        limit_mb = cfg.upload_limit_mb
        base = _output_basename(title, job.id, tgt_lang)

        if not info.has_video:
            audio = encode_audio_only(dubbed, job_dir / "result.m4a", cfg)
            saved = _save_output(audio, cfg, base, ".m4a")
            saved_srt = _save_output(srt, cfg, base, ".srt", copy=True)
            fits = saved.stat().st_size <= limit_mb * 1024 * 1024
            return (saved if fits else None), saved, srt, saved_srt, []

        out = mux_video(input_path, dubbed, job_dir / "result.mp4", cfg,
                        original_audio=source_wav if keep_orig else None)
        # тяжёлые дорожки больше не нужны — освобождаем диск до пережатия
        _free_tracks(job_dir, [dubbed])

        # готовый дубляж в полном качестве всегда ложится в папку готовых видео:
        # даже если в Telegram он не влезет, работа не пропадает
        saved = _save_output(out, cfg, base, ".mp4")
        saved_srt = _save_output(srt, cfg, base, ".srt", copy=True)

        try:
            fitted = compress_to_limit(saved, limit_mb, cfg, out_dir=job_dir)
            return fitted, saved, srt, saved_srt, []
        except TooLongForLimit:
            parts = _maybe_split(saved, limit_mb, cfg, job_dir)
            return None, saved, srt, saved_srt, parts

    output_path, saved_path, srt_path, saved_srt, parts = await asyncio.to_thread(_stage10)
    await asyncio.to_thread(project_store.save_render_results, job.id, segments,
                            render_stats, ProjectStage.DONE, str(saved_path))

    elapsed = time.monotonic() - t0
    summary = _build_summary(src_lang, tgt_lang, profiles, elapsed)
    return JobResult(
        kind="video" if info.has_video else "audio",
        output_path=output_path,
        srt_path=srt_path,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        speakers=profiles,
        elapsed_s=elapsed,
        summary=summary,
        parts=parts,
        saved_path=saved_path,
        saved_srt=saved_srt,
    )


# ---------------------------------------------------------------------------


async def _open_review(job, bot, cfg, hooks, job_dir: Path, segments: list[dict],
                       profiles: dict, src_lang: str, tgt_lang: str,
                       media: dict | None = None):
    """Сохраняет project.json и, если нужно, ждёт утверждения человеком.

    Возвращает проект с правками пользователя или None, если студия
    выключена (INV-6: старый сценарий продолжает работать).
    """
    from project import store
    from project.schema import Stage

    auto_approve = bool(job.settings.get("auto_approve", False))
    try:
        proj = await asyncio.to_thread(
            store.from_pipeline, job.id, segments, profiles, src_lang, tgt_lang,
            {"voice_mode": job.settings.get("voice_mode", "auto"),
             "auto_approve": auto_approve,
             "keep_background": job.settings.get("keep_background", True),
             "translation_style": job.settings.get("translation_style", "normal")},
            media or {},
            job.user_id, job.chat_id,
            Stage.REVIEW if (cfg.studio_on and not auto_approve) else Stage.APPROVED,
        )
    except Exception:  # noqa: BLE001 — проект не должен ронять дубляж
        log.exception("Не удалось сохранить project.json — иду без студии")
        return None

    if not cfg.studio_on:
        return proj
    if auto_approve:
        log.info("auto_approve включён — проверка пропускается")
        return proj

    from bot.review import request_review, wait_for_approval

    await hooks.report(8, "жду проверки в студии", 100)
    await request_review(job, bot, cfg, proj)
    log.info("Задача %s ждёт утверждения в студии", job.id)

    approved = await wait_for_approval(job.id,
                                       timeout_hours=cfg.studio_link_ttl_h)
    if not approved:
        raise UserError(
            "Проверка так и не была подтверждена. Пришлите видео заново или "
            "включите автоматический режим в настройках.")

    return await asyncio.to_thread(store.load, job.id)


def _apply_review(proj, segments: list[dict], profiles: dict) -> tuple[list[dict], dict]:
    """Переносит правки из проекта обратно в структуры пайплайна.

    Проект — источник правды после проверки: человек мог переназначить
    спикеров, поправить перевод и сменить голоса, и рендер обязан
    использовать именно это (INV-3).
    """
    by_id = {seg["id"]: seg for seg in segments}
    for item in proj.segments:
        seg = by_id.get(item.id)
        if seg is None:
            # реплика появилась при делении в студии
            seg = {"id": item.id}
            segments.append(seg)
        seg["start"], seg["end"] = item.start, item.end
        seg["speaker"] = item.speaker_id
        seg["text"] = item.text_tts or item.text_tgt
        seg["text_tts"] = seg["text"]
        seg["flags"] = list(item.flags)
        if item.voice_override:
            seg["voice_override"] = item.voice_override.model_dump()
    # удалённые склейкой реплики уходят и из пайплайна
    alive = {item.id for item in proj.segments}
    segments[:] = sorted((s for s in segments if s["id"] in alive),
                         key=lambda s: s["start"])

    for sp in proj.speakers:
        entry = profiles.setdefault(sp.id, {"id": sp.id})
        entry["gender"] = sp.gender
        entry["label"] = sp.label
        entry["name"] = sp.name
        entry["voice"] = sp.voice.model_dump()
        entry["reference"] = sp.reference.model_dump()
        entry["ref_main"] = sp.reference.path
        entry["ref_ok"] = sp.reference.clone_allowed
        if sp.merged_into:
            entry["merged_into"] = sp.merged_into
    return segments, profiles


def _check_disk(job_dir: Path, duration: float, input_path: Path, cfg) -> None:
    """Хватит ли места на дорожки и результат.

    Пайплайн держит на диске несжатый звук: source + vocals + background +
    dubbed — около 0.7 МБ на секунду ролика. На двухчасовом видео это ~5 ГБ
    плюс сам файл и результат. Узнать об этом лучше сейчас, чем через час
    обработки на записи микса.
    """
    if duration <= 0:
        return
    mb_per_s = float(cfg.y("limits", "disk_mb_per_second", default=0.9))
    try:
        input_gb = input_path.stat().st_size / 1e9
    except OSError:
        input_gb = 0.0
    need_gb = duration * mb_per_s / 1000 + input_gb * 1.5 + 1.0
    free_gb = media.free_disk_gb(job_dir)
    log.info("Диск: нужно ~%.1f ГБ, свободно %.1f ГБ", need_gb, free_gb)
    if free_gb < need_gb:
        raise UserError(
            f"Не хватает места на диске: ролику на {duration / 60:.0f} мин нужно "
            f"около {need_gb:.0f} ГБ, а свободно {free_gb:.0f} ГБ. "
            "Освободите место и пришлите ссылку снова."
        )


def _should_cache_media(duration: float, input_path: Path, cfg,
                        job_dir: Path) -> bool:
    """Класть ли дорожки источника в кэш.

    Час несжатого звука — около 3 ГБ, поэтому длинный ролик кэшируем только
    когда на диске остаётся ощутимый запас. Кэш того стоит: повторный прогон
    с другим числом спикеров иначе заново качает и заново гоняет Demucs.
    """
    if duration <= cfg.cache_max_source_s:
        return True
    try:
        input_gb = input_path.stat().st_size / 1e9
    except OSError:
        input_gb = 0.0
    need_gb = duration * 0.8 / 1000 + input_gb
    keep_free = float(cfg.y("cache", "keep_free_gb", default=15))
    free_gb = media.free_disk_gb(job_dir)
    ok = free_gb - need_gb >= keep_free
    log.info("Кэш медиа: %s (нужно %.1f ГБ, свободно %.1f ГБ)",
             "сохраняю" if ok else "пропускаю", need_gb, free_gb)
    return ok


def _resume_media(job_dir: Path) -> tuple[Path, ...] | None:
    """Медиа прерванной задачи, если они целы в её собственной папке.

    Дорожки длинного ролика в общий кэш не переносятся (это гигабайты wav),
    поэтому при продолжении брать их надо здесь. Иначе пайплайн заново качает
    видео — а к тому времени ссылка может уже не открыться (регион, приватность),
    и часы уже сделанной работы пропадают.
    """
    from core import cache as source_cache

    inp = source_cache.input_file(job_dir)
    if inp is None or not inp.is_file() or inp.stat().st_size < 1_000_000:
        return None
    # background.wav нужен миксу так же, как вокал: проверяем весь комплект
    names = ["analysis.wav", "source.wav", "vocals.wav", "vocals16.wav",
             "background.wav"]
    tracks = [job_dir / n for n in names]
    if not all(p.is_file() and p.stat().st_size > 1000 for p in tracks):
        return None
    analysis, source, vocals, vocals16, background = tracks
    return inp, analysis, source, vocals, vocals16, background


def _free_tracks(job_dir: Path, extra: list[Path]) -> None:
    """Удаляет несжатые дорожки задачи: после микса они больше не нужны.

    Трогаем только файлы самой задачи — то, что уехало в кэш, лежит в другой
    папке и должно там остаться.
    """
    names = ["source.wav", "vocals.wav", "background.wav",
             "analysis.wav", "vocals16.wav"]
    freed = 0
    for path in [job_dir / n for n in names] + [Path(p) for p in extra]:
        try:
            if path.parent == job_dir and path.exists():
                freed += path.stat().st_size
                path.unlink()
        except OSError:
            log.warning("Не удалось удалить временную дорожку %s", path)
    if freed:
        log.info("Освободил %.1f ГБ временных дорожек", freed / 1e9)


def _read_title(job_dir: Path, cached_media: Path | None) -> str:
    for folder in (job_dir, cached_media):
        if not folder:
            continue
        path = folder / "title.txt"
        if path.exists():
            try:
                return path.read_text(encoding="utf-8").strip()
            except OSError:
                pass
    return ""


_BAD_CHARS = '<>:"/\\|?*\n\r\t'


def _output_basename(title: str, job_id: str, tgt_lang: str) -> str:
    """Имя файла в папке готовых видео: название ролика + язык озвучки."""
    clean = "".join(" " if c in _BAD_CHARS else c for c in title).strip()
    clean = " ".join(clean.split())[:80].strip(" .")
    stamp = time.strftime("%Y-%m-%d")
    return f"{stamp} {clean or job_id} [{tgt_lang}]"


def _save_output(src: Path, cfg, base: str, ext: str, copy: bool = False) -> Path:
    """Кладёт файл в папку готовых видео, не затирая уже лежащий там."""
    out_dir = cfg.output_dir
    dst = out_dir / f"{base}{ext}"
    n = 2
    while dst.exists():
        dst = out_dir / f"{base} ({n}){ext}"
        n += 1
    if copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))
    log.info("Готовый файл: %s", dst)
    return dst


def _maybe_split(video: Path, limit_mb: float, cfg, job_dir: Path) -> list[Path]:
    """Части для чата — только если их немного. Иначе результат ждёт в папке."""
    from core.mux import plan_parts, split_to_parts

    max_parts = int(cfg.y("output", "max_parts", default=6))
    count = plan_parts(video, limit_mb)
    if max_parts and count > max_parts:
        log.info("Понадобилось бы %d частей — не режу, файл остаётся в папке",
                 count)
        return []
    if media.free_disk_gb(job_dir) < video.stat().st_size / 1e9 * 1.2:
        log.info("Мало места для нарезки на части — файл остаётся в папке")
        return []
    return split_to_parts(video, limit_mb, out_dir=job_dir)


def _load_translated(job_dir: Path, tgt_lang: str, count: int) -> list[dict] | None:
    """Готовый перевод из папки задачи — если это продолжение прерванной работы.

    Подходит только когда язык тот же и реплик столько же: иначе разбор
    изменился и старый перевод к нему не относится.
    """
    path = job_dir / "translated.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        log.warning("Не удалось прочитать сохранённый перевод — перевожу заново")
        return None
    segments = data.get("segments") or []
    if data.get("language") != tgt_lang or len(segments) != count:
        log.info("Сохранённый перевод не подходит (язык %s, реплик %d) — заново",
                 data.get("language"), len(segments))
        return None
    return segments


SYNTH_DONE = "done.txt"


def _synth_done_ids(synth_dir: Path) -> set[int]:
    """Реплики, озвучка которых точно доведена до конца.

    Судить по самому файлу нельзя: оборванный wav выглядит целым — длину
    libsndfile берёт из заголовка. Поэтому id дописывается в журнал ПОСЛЕ
    того, как файл закрыт; оборвали задачу — последняя строка просто не
    успела появиться, и реплика переозвучится.
    """
    path = synth_dir / SYNTH_DONE
    if not path.exists():
        return set()
    ids = set()
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
        # последняя строка без перевода строки могла не дописаться (обрыв
        # посреди записи превратил бы «1234» в «12» — чужой номер)
        if lines and lines[-1] != "":
            lines = lines[:-1]
        for line in lines:
            line = line.strip()
            if line.isdigit():
                ids.add(int(line))
    except OSError:
        log.warning("Журнал озвученных реплик не читается — озвучиваю заново")
    return ids


def _mark_synth_done(synth_dir: Path, seg_id: int) -> None:
    try:
        with open(synth_dir / SYNTH_DONE, "a", encoding="utf-8") as f:
            f.write(f"{seg_id}\n")
            f.flush()
    except OSError:
        log.warning("Не удалось отметить реплику %s в журнале", seg_id)


def _reusable_synth(path: Path, done: set[int], seg_id: int) -> bool:
    """Готова ли реплика: есть в журнале и файл на месте."""
    return seg_id in done and path.exists() and path.stat().st_size > 1000


def _analysis_still_valid(cached_analysis, cfg):
    """Отбрасывает разбор, полученный при других настройках распознавания.

    Кэш хранит готовый транскрипт и НЕ помнит, чем он сделан. Из-за этого
    исправление распознавания молча не применяется: этап 4 просто не
    вызывается, и задача берёт старый результат. Ошибка при этом выглядит
    не как «кэш устарел», а как «исправление не работает» — ищется долго.
    """
    if cached_analysis is None:
        return None
    forced = cfg.y("asr", "language", default=None)
    if not forced:
        return cached_analysis
    from core import cache as source_cache
    try:
        tr = source_cache.load_transcript(cached_analysis)
    except Exception:  # noqa: BLE001 — нечитаемый кэш просто игнорируем
        log.exception("Кэш разбора не читается — распознаю заново")
        return None
    got = (tr.get("language") or "").lower()
    if got != str(forced).lower():
        log.warning("Кэш разбора сделан на языке %r, а задан %r — "
                    "распознаю заново", got, forced)
        return None
    return cached_analysis


def _precompress(segments: list[dict], limits: dict[int, float], translator,
                 tgt_lang: str, cfg, progress=None) -> int:
    """Заранее укорачивает реплики, которые не поместятся даже с ускорением.

    Иначе то же самое делается в цикле синтеза по одной реплике: на часовом
    ролике это сотни отдельных обращений к модели, и они стоят дороже самого
    синтеза. Считаем бюджет по табличному темпу речи и просим сократить
    пачкой; не получилось — цикл синтеза разберётся сам, как и раньше.
    """
    if not bool(cfg.y("translation", "precompress", default=True)):
        return 0
    compress_batch = getattr(translator, "compress_batch", None)
    if compress_batch is None:
        return 0

    from core.normalize import normalize_for_tts
    from core.translate import compute_max_chars

    atempo = cfg.atempo_max * float(cfg.y("timing", "speed_soft_max", default=1.15))
    ratio = float(cfg.y("translation", "precompress_ratio", default=1.15))
    batch_size = int(cfg.y("translation", "precompress_batch", default=30))

    items, budgets = [], {}
    for seg in segments:
        if seg.get("skip_tts") or not seg["text"].strip():
            continue
        slot = max(0.4, limits.get(seg["id"], seg["end"]) - seg["start"])
        # сколько символов реально произносится в слоте с допустимым ускорением
        budget = compute_max_chars(seg, tgt_lang, cfg, slot * atempo)
        if len(seg["text"]) > budget * ratio:
            items.append({"id": seg["id"], "max_chars": budget,
                          "text": seg["text"]})
            budgets[seg["id"]] = budget
    if not items:
        return 0

    log.info("Предварительное сжатие: %d реплик из %d не влезают в слоты",
             len(items), len(segments))
    by_id = {s["id"]: s for s in segments}
    changed = 0
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        try:
            result = compress_batch(batch, tgt_lang)
        except Exception:  # noqa: BLE001 — сжатие необязательно
            log.exception("Предварительное сжатие: пачка не удалась")
            result = {}
        for seg_id, text in result.items():
            seg = by_id.get(seg_id)
            if not seg or not text.strip():
                continue
            # длиннее исходного — это не сокращение, а пересказ: не берём
            if len(text) >= len(seg["text"]):
                continue
            seg["text"] = normalize_for_tts(text, tgt_lang)
            changed += 1
        if progress:
            progress(int(100 * min(len(items), i + batch_size) / len(items)))
    log.info("Предварительное сжатие: укорочено %d реплик", changed)
    return changed


def _free_vram() -> None:
    """Отдать кэш видеопамяти обратно: следующая реплика может уместиться."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — без CUDA чистить нечего
        pass


def _slot_limits(segments: list[dict], cfg) -> dict[int, float]:
    """До какого момента каждая реплика может звучать.

    Слот = её собственная длительность плюс пауза до следующей реплики
    (любого спикера) минус зазор. Это снимает нужду в сильном ускорении:
    в оригинале между фразами почти всегда есть тишина.
    """
    import bisect

    guard = float(cfg.y("timing", "gap_guard_ms", default=80)) / 1000
    max_extend = float(cfg.y("timing", "max_extend_s", default=1.5))
    starts = sorted(s["start"] for s in segments)

    limits: dict[int, float] = {}
    for s in segments:
        limit = s["end"] + max_extend
        idx = bisect.bisect_right(starts, s["start"])
        if idx < len(starts):
            limit = min(limit, starts[idx] - guard)
        # при наложении речи следующая реплика может начаться раньше конца этой
        limits[s["id"]] = max(s["end"], limit)
    return limits


def _norm_lang(code: str) -> str:
    code = (code or "").lower()
    return "zh-cn" if code.startswith("zh") else code


def _save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _build_summary(src_lang: str, tgt_lang: str, profiles: dict,
                   elapsed: float) -> str:
    src_name = TTS_LANGUAGES.get(_norm_lang(src_lang), src_lang)
    tgt_name = TTS_LANGUAGES.get(tgt_lang, tgt_lang)
    spk_lines = "\n".join(
        f"  • {p['id']} — {GENDER_RU.get(p['gender'], p['gender'])}"
        + (" (ребёнок)" if p.get("age") == "child" else "")
        + f" (уверенность {p['gender_confidence']:.0%})"
        + (f", голос: {p['bank_voice']}" if p.get("bank_voice") else "")
        for p in profiles.values()
    )
    mm, ss = divmod(int(elapsed), 60)
    return (
        f"✅ Готово!\n\n"
        f"Язык оригинала: {src_name}\n"
        f"Язык озвучки: {tgt_name}\n"
        f"Спикеров: {len(profiles)}\n{spk_lines}\n"
        f"Время обработки: {mm:02d}:{ss:02d}"
    )


# ---------------------------------------------------------------------------
# Очистка временных файлов
# ---------------------------------------------------------------------------

KEEP_FILES = {"speakers.json", "transcript.json", "translated.json", "job.log"}


def cleanup_job(job_id: str) -> None:
    """После отправки результата удаляем медиа, оставляем логи и speakers.json.

    Перед удалением артефакты анализа переезжают в кэш источника —
    повторная обработка того же ролика начнётся сразу с перевода.
    """
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        return
    key_file = job_dir / "cache_key.txt"
    if key_file.exists():
        try:
            from core import cache as source_cache
            key, _, hint = key_file.read_text(encoding="utf-8").strip().partition("\n")
            if not (job_dir / NO_MEDIA_CACHE).exists():
                source_cache.store_media(job_dir, key)
            source_cache.store_analysis(job_dir, key, hint or "auto")
        except Exception:  # noqa: BLE001 — кэш не должен ломать очистку
            log.exception("Кэш: не удалось сохранить артефакты задачи %s", job_id)
    for item in job_dir.iterdir():
        if item.name in KEEP_FILES:
            continue
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
        except OSError:
            log.warning("Не удалось удалить %s", item)


def purge_old_jobs(keep_days: int = 7) -> None:
    """Удаляет папки задач старше keep_days (вызывается при старте)."""
    cutoff = time.time() - keep_days * 86400
    if not JOBS_DIR.exists():
        return
    for d in JOBS_DIR.iterdir():
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                log.info("Удалена старая папка задачи: %s", d.name)
        except OSError:
            pass
