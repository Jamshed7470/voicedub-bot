"""Кэш исходников: медиа отдельно, разбор голосов отдельно.

Два уровня, потому что они устаревают по-разному:

    data/cache/<источник>/
        input.mp4, source.wav, analysis.wav, vocals.wav, vocals16.wav,
        background.wav, media.done          ← зависит только от ссылки
        analysis_auto/                      ← зависит ещё и от числа спикеров
        analysis_6/    {transcript.json, speakers.json, speakers/, analysis.done}

Повтор той же ссылки пропускает скачивание, разделение дорожек, распознавание
и разбор голосов — начинается сразу с перевода. Смена числа спикеров
пересчитывает только разбор, медиа берётся из кэша.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from pathlib import Path

from core.config import DATA_DIR

log = logging.getLogger(__name__)

CACHE_DIR = DATA_DIR / "cache"
MEDIA_MARKER = "media.done"
ANALYSIS_MARKER = "analysis.done"

MEDIA_FILES = ["source.wav", "analysis.wav", "vocals.wav",
               "vocals16.wav", "background.wav"]
ANALYSIS_FILES = ["transcript.json", "speakers.json"]


def source_key(kind: str, payload) -> str:
    """Отпечаток источника: ссылка, путь к файлу или file_id телеграм-файла."""
    raw = (payload.strip() if isinstance(payload, str)
           else str(payload.get("file_id", "")))
    return hashlib.sha256(f"{kind}:{raw}".encode()).hexdigest()[:16]


def analysis_name(speakers_hint: str) -> str:
    return f"analysis_{speakers_hint or 'auto'}"


def media_dir(key: str) -> Path | None:
    d = CACHE_DIR / key
    return d if (d / MEDIA_MARKER).exists() else None


def analysis_dir(key: str, speakers_hint: str) -> Path | None:
    d = CACHE_DIR / key / analysis_name(speakers_hint)
    return d if (d / ANALYSIS_MARKER).exists() else None


def touch(path: Path) -> None:
    """Отмечает запись как использованную (для LRU-очистки)."""
    try:
        for marker in (path / MEDIA_MARKER, path / ANALYSIS_MARKER):
            if marker.exists():
                marker.touch()
    except OSError:
        pass


def input_file(cache_dir: Path) -> Path | None:
    return next((p for p in cache_dir.iterdir()
                 if p.name.startswith("input.")), None)


def load_transcript(analysis: Path) -> dict:
    with open(analysis / "transcript.json", encoding="utf-8") as f:
        return json.load(f)


def load_profiles(analysis: Path) -> dict:
    """Профили спикеров с путями, переписанными на папку кэша."""
    with open(analysis / "speakers.json", encoding="utf-8") as f:
        profiles = json.load(f)
    spk_root = analysis / "speakers"
    for spk, prof in profiles.items():
        if prof.get("ref_main"):
            prof["ref_main"] = str(spk_root / spk / Path(prof["ref_main"]).name)
        prof["refs_emotion"] = {
            emo: str(spk_root / spk / Path(p).name)
            for emo, p in prof.get("refs_emotion", {}).items()
        }
    return profiles


def store_media(job_dir: Path, key: str) -> Path | None:
    """Переносит видео и дорожки в кэш (move — дёшево в пределах диска).

    Вызывается сразу, как только дорожки готовы: если задача потом упадёт
    или бота перезапустят, скачивание и Demucs не придётся повторять.
    Возвращает папку кэша — вызывающий код должен перевести пути на неё.
    """
    dst = CACHE_DIR / key
    if (dst / MEDIA_MARKER).exists():
        return dst
    inp = next((p for p in job_dir.iterdir()
                if p.name.startswith("input.")), None)
    needed = [job_dir / n for n in MEDIA_FILES]
    if inp is None or not all(p.exists() for p in needed):
        return None  # дорожки ещё не готовы — кэшировать нечего
    dst.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(inp), str(dst / inp.name))
        for n in MEDIA_FILES:
            shutil.move(str(job_dir / n), str(dst / n))
        title = job_dir / "title.txt"
        if title.exists():  # имя ролика нужно и при обработке из кэша
            shutil.copy2(title, dst / "title.txt")
        (dst / MEDIA_MARKER).write_text(str(int(time.time())), encoding="utf-8")
        log.info("Кэш: сохранил медиа источника %s", key)
        return dst
    except OSError:
        log.exception("Кэш: не удалось сохранить медиа %s", key)
        shutil.rmtree(dst, ignore_errors=True)
        return None


def store_analysis(job_dir: Path, key: str, speakers_hint: str) -> Path | None:
    """Переносит транскрипт и профили спикеров в кэш."""
    dst = CACHE_DIR / key / analysis_name(speakers_hint)
    if (dst / ANALYSIS_MARKER).exists():
        return dst
    needed = [job_dir / n for n in ANALYSIS_FILES] + [job_dir / "speakers"]
    if not all(p.exists() for p in needed):
        return None
    dst.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(job_dir / "speakers"), str(dst / "speakers"))
        for n in ANALYSIS_FILES:
            # json остаются в папке задачи для отладки
            shutil.copy2(job_dir / n, dst / n)
        (dst / ANALYSIS_MARKER).write_text(str(int(time.time())), encoding="utf-8")
        log.info("Кэш: сохранил разбор голосов %s/%s", key,
                 analysis_name(speakers_hint))
        return dst
    except OSError:
        log.exception("Кэш: не удалось сохранить разбор %s", key)
        shutil.rmtree(dst, ignore_errors=True)
        return None


def purge(max_gb: float = 10.0, keep_days: int = 14) -> None:
    """Удаляет записи старше keep_days и ужимает кэш до max_gb (старые первыми)."""
    if not CACHE_DIR.exists():
        return
    entries = []
    for d in CACHE_DIR.iterdir():
        if not d.is_dir():
            continue
        if (d / ANALYSIS_MARKER).exists() and not (d / MEDIA_MARKER).exists():
            shutil.rmtree(d, ignore_errors=True)  # запись старого формата
            log.info("Кэш: удалил запись старого формата %s", d.name)
            continue
        marker = d / MEDIA_MARKER
        mtime = marker.stat().st_mtime if marker.exists() else 0
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        entries.append((mtime, size, d))
    total = sum(s for _, s, _ in entries)
    cutoff = time.time() - keep_days * 86400
    entries.sort()  # самые старые первыми
    for mtime, size, d in entries:
        if mtime < cutoff or total > max_gb * 1e9:
            shutil.rmtree(d, ignore_errors=True)
            total -= size
            log.info("Кэш: удалил устаревшую запись %s", d.name)
