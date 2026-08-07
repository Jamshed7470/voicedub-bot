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
from dataclasses import dataclass
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


@dataclass
class JobResult:
    kind: str                       # "video" | "audio"
    output_path: Path
    srt_path: Path
    src_lang: str
    tgt_lang: str
    speakers: dict
    elapsed_s: float
    summary: str = ""


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

    # ---------- 1. Скачивание ----------
    await report(1)
    if job.kind == "url":
        from core.downloader import download_url
        input_path = await asyncio.to_thread(download_url, job.payload, job_dir, cfg)
    else:
        from core.downloader import download_from_telegram
        input_path = job_dir / "input.bin"
        await download_from_telegram(bot, job.payload["file_id"],
                                     job.payload.get("file_size"), input_path, cfg)

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
    analysis_wav, source_wav = await asyncio.to_thread(
        media.extract_audio, input_path, job_dir,
        int(cfg.y("audio", "analysis_sr", default=16000)),
        int(cfg.y("audio", "source_sr", default=44100)),
    )

    # ---------- 3. Разделение голоса и фона ----------
    await report(3)
    from core.separate import separate
    vocals_wav, background_wav = await asyncio.to_thread(
        separate, source_wav, job_dir, cfg)
    vocals16_wav = job_dir / "vocals16.wav"

    # ---------- 4. Распознавание речи ----------
    await report(4)
    from core.asr import transcribe
    asr_result = await asyncio.to_thread(transcribe, str(analysis_wav), cfg,
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
    segments = await asyncio.to_thread(diarize_and_assign, str(analysis_wav),
                                       asr_result, cfg)
    if not segments:
        raise UserError("В этом файле не нашлось речи — нечего дублировать.")

    # ---------- 6. Эмоции, события, профили спикеров ----------
    await report(6)
    import numpy as np  # noqa: F401

    from core.emotions import classify_segments
    from core.events import detect_events, mark_event_segments
    from core.speakers import build_profiles

    def _stage6() -> tuple[list[dict], dict]:
        import librosa
        y16, _ = librosa.load(str(vocals16_wav), sr=16000, mono=True)
        events = detect_events(y16, 16000, cfg, report_ts(6))
        mark_event_segments(segments, events, cfg)
        classify_segments(vocals16_wav, segments, cfg, tmp_dir)
        profiles = build_profiles(job_dir, vocals_wav, segments, cfg)
        return events, profiles

    events, profiles = await asyncio.to_thread(_stage6)

    _save_json(job_dir / "transcript.json",
               {"language": src_lang, "segments": segments, "events": events})

    # ---------- 7. Перевод ----------
    await report(7)
    if not same_lang:
        from core.translate import get_translator
        translator = get_translator(cfg)
        await asyncio.to_thread(translator.translate_segments, segments,
                                src_lang, tgt_lang, profiles, report_ts(7))
    else:
        from core.translate import get_translator
        translator = get_translator(cfg)  # понадобится для сжатия сегментов
        log.info("Языки совпадают — переозвучка без перевода")
    _save_json(job_dir / "translated.json",
               {"language": tgt_lang, "segments": segments})

    # ---------- 8. Нормализация текста ----------
    await report(8)
    from core.normalize import normalize_for_tts
    for seg in segments:
        seg["text"] = normalize_for_tts(seg["text"], tgt_lang)

    # ---------- 9. Синтез + подгонка таймингов ----------
    await report(9)
    placed = await asyncio.to_thread(
        _synthesize_all, job_dir, tmp_dir, segments, profiles, vocals_wav,
        translator, tgt_lang, cfg, report_ts(9), hooks)

    # ---------- 10. Микс и сборка ----------
    await report(10)
    from core.mixer import build_mix
    from core.mux import compress_to_limit, encode_audio_only, mux_video, write_srt

    keep_bg = job.settings.get("keep_background", True)
    keep_orig = job.settings.get("keep_original_track", False)

    def _stage10() -> tuple[Path, Path]:
        dubbed = build_mix(job_dir, background_wav, vocals_wav, placed,
                           events, cfg, keep_background=keep_bg)
        srt = write_srt(segments, job_dir / "subtitles.srt")
        limit_mb = float(cfg.y("limits", "telegram_upload_mb", default=50))
        if info.has_video:
            out = mux_video(input_path, dubbed, job_dir / "result.mp4", cfg,
                            original_audio=source_wav if keep_orig else None)
            out = compress_to_limit(out, limit_mb, cfg)
        else:
            out = encode_audio_only(dubbed, job_dir / "result.m4a", cfg)
        return out, srt

    output_path, srt_path = await asyncio.to_thread(_stage10)

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
    )


# ---------------------------------------------------------------------------


def _synthesize_all(job_dir: Path, tmp_dir: Path, segments: list[dict],
                    profiles: dict, vocals_wav: Path, translator, tgt_lang: str,
                    cfg, progress, hooks: PipelineHooks) -> list[dict]:
    """Синтез каждого сегмента + подгонка под слот (синхронно, в потоке)."""
    from core.normalize import normalize_for_tts
    from core.timing import STATUS_TOO_LONG, fit_to_slot
    from core.translate import compute_max_chars
    from core.tts import choose_style_ref, synthesize

    synth_dir = job_dir / "synth"
    synth_dir.mkdir(exist_ok=True)
    atempo_max = cfg.atempo_max
    speed_soft_max = float(cfg.y("timing", "speed_soft_max", default=1.3))

    to_do = [s for s in segments if not s.get("skip_tts") and s["text"].strip()]
    placed: list[dict] = []

    for i, seg in enumerate(to_do):
        if hooks.cancel_event is not None and hooks.cancel_event.is_set():
            raise JobCancelled()

        slot = max(0.4, seg["end"] - seg["start"])
        profile = profiles.get(seg["speaker"], {})
        speaker_wav, preset = choose_style_ref(seg, profile, vocals_wav, tmp_dir, cfg)

        raw = synth_dir / f"seg_{seg['id']}_raw.wav"
        fitted = synth_dir / f"seg_{seg['id']}.wav"

        def _synth(text: str, speed: float = 1.0) -> None:
            synthesize(text, tgt_lang, raw, cfg,
                       speaker_wav=speaker_wav, preset=preset, speed=speed)

        try:
            _synth(seg["text"])
            fit = fit_to_slot(raw, fitted, slot, atempo_max)

            # мягкая подгонка параметром speed до применения atempo
            if fit.status == STATUS_TOO_LONG and fit.tempo <= atempo_max * speed_soft_max:
                _synth(seg["text"], speed=min(speed_soft_max, fit.tempo))
                fit = fit_to_slot(raw, fitted, slot, atempo_max)

            if fit.status == STATUS_TOO_LONG:
                # запросить у переводчика сжатый вариант и синтезировать заново
                max_chars = compute_max_chars(seg, tgt_lang, cfg)
                compressed = translator.compress_segment(seg["text"], max_chars,
                                                         tgt_lang)
                compressed = normalize_for_tts(compressed, tgt_lang)
                if compressed.strip() and compressed != seg["text"]:
                    seg["text"] = compressed
                    _synth(compressed, speed=speed_soft_max)
                    fit = fit_to_slot(raw, fitted, slot, atempo_max)

            if fit.status == STATUS_TOO_LONG:
                # последний резерв: максимальное ускорение, лёгкий заход на паузу
                log.warning("Сегмент %s не влез даже после сжатия (×%.2f)",
                            seg["id"], fit.tempo)
                from core.media import run as ffrun
                ffrun(["ffmpeg", "-y", "-i", str(raw),
                       "-filter:a", f"atempo={atempo_max}",
                       "-c:a", "pcm_s16le", str(fitted)], desc="ffmpeg atempo max")

            placed.append({"start": seg["start"], "path": str(fitted),
                           "id": seg["id"]})
        except AssertionError:
            raise
        except Exception:  # noqa: BLE001 — один сломанный сегмент не валит задачу
            log.exception("Синтез сегмента %s не удался, пропускаю", seg["id"])
        finally:
            raw.unlink(missing_ok=True)

        if progress:
            progress(int(100 * (i + 1) / max(1, len(to_do))))

    if not placed:
        raise UserError("Не удалось синтезировать ни одного сегмента речи.")
    return placed


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
        f"  • {p['id']} — {GENDER_RU.get(p['gender'], p['gender'])} "
        f"(уверенность {p['gender_confidence']:.0%})"
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
    """После отправки результата удаляем медиа, оставляем логи и speakers.json."""
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        return
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
