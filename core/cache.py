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
    """Имя папки с разбором. Содержит версию алгоритма спикеров.

    Без версии старый разбор подхватывается после смены алгоритма, и
    исправление молча не применяется к уже обработанным видео: этап
    диаризации просто не вызывается. Симптом при этом выглядит как «фикс
    не работает», а не как «кэш устарел», и ищется долго.
    """
    from identity import ALGO_VERSION

    return f"analysis_{speakers_hint or 'auto'}_v{ALGO_VERSION}"


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


# Все поля speakers.json, в которых лежит путь внутрь speakers/<id>/.
# Папка целиком переезжает в кэш, поэтому каждый такой путь надо переписать.
# Забытое поле не даёт ошибки при чтении — оно молча указывает в никуда, и
# задача падает много позже, уже на синтезе. Поэтому список явный.
PROFILE_PATH_FIELDS = (
    ("ref_main",),
    ("reference", "path"),
    ("voice", "profile_path"),
    ("voice", "identity_path"),
    ("centroid_path",),
)


def load_profiles(analysis: Path) -> dict:
    """Профили спикеров с путями, переписанными на указанную папку.

    Работает и для кэша, и для папки задачи: важно лишь, что рядом с
    speakers.json лежит папка speakers/.
    """
    with open(analysis / "speakers.json", encoding="utf-8") as f:
        profiles = json.load(f)
    spk_root = analysis / "speakers"

    for spk, prof in profiles.items():
        for field in PROFILE_PATH_FIELDS:
            node = prof
            for key in field[:-1]:
                node = node.get(key) if isinstance(node, dict) else None
                if node is None:
                    break
            if not isinstance(node, dict):
                continue
            value = node.get(field[-1])
            if value:
                node[field[-1]] = str(spk_root / spk / Path(value).name)

        # эмоциональные референсы остались в старых кэшах версии 1
        if prof.get("refs_emotion"):
            prof["refs_emotion"] = {
                emo: str(spk_root / spk / Path(p).name)
                for emo, p in prof["refs_emotion"].items()
            }
    return profiles


def verify_profiles(profiles: dict) -> bool:
    """Все ли файлы профилей на месте после переезда в кэш.

    Проверка нужна здесь, а не при синтезе: там до неё дело дойдёт через
    десятки минут работы, и причина будет уже не видна.
    """
    missing = []
    for spk, prof in profiles.items():
        path = ((prof.get("voice") or {}).get("profile_path"))
        if path and not Path(path).exists():
            missing.append(spk)
    if missing:
        log.warning("Кэш: у спикеров %s нет файлов профиля — пересоберу разбор",
                    ", ".join(missing))
    return not missing


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
    """Кладёт КОПИЮ разбора и профилей в кэш.

    Именно копию, а не перенос. Раньше папка speakers/ переезжала в кэш, и
    задача начинала зависеть от него: очистка кэша по возрасту или размеру
    уносила профили голосов активной задачи, а та узнавала об этом только
    на синтезе — «у спикера нет профиля», хотя всё было посчитано.

    Профили весят мегабайты, кэш экономил единицы гигабайт на диске — цена
    несопоставима с потерей часа работы. Кэш остаётся ускорителем для
    следующих задач, но источником правды для текущей быть перестал.
    """
    dst = CACHE_DIR / key / analysis_name(speakers_hint)
    if (dst / ANALYSIS_MARKER).exists():
        return dst
    needed = [job_dir / n for n in ANALYSIS_FILES] + [job_dir / "speakers"]
    if not all(p.exists() for p in needed):
        return None
    dst.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(job_dir / "speakers", dst / "speakers",
                        dirs_exist_ok=True)
        for n in ANALYSIS_FILES:
            shutil.copy2(job_dir / n, dst / n)
        (dst / ANALYSIS_MARKER).write_text(str(int(time.time())), encoding="utf-8")
        log.info("Кэш: сохранил разбор голосов %s/%s", key,
                 analysis_name(speakers_hint))
        return dst
    except OSError:
        log.exception("Кэш: не удалось сохранить разбор %s", key)
        shutil.rmtree(dst, ignore_errors=True)
        return None


def adopt_analysis(analysis: Path, job_dir: Path) -> dict:
    """Забирает разбор из кэша В ПАПКУ ЗАДАЧИ и отдаёт профили с её путями.

    Задача, взявшая разбор из кэша, тоже не должна от него зависеть:
    очистка кэша не имеет права её сломать.
    """
    src = analysis / "speakers"
    if src.exists():
        try:
            shutil.copytree(src, job_dir / "speakers", dirs_exist_ok=True)
        except OSError:
            log.exception("Кэш: не удалось скопировать профили в задачу — "
                          "работаю по путям кэша")
            return load_profiles(analysis)
    for n in ANALYSIS_FILES:
        if (analysis / n).exists() and not (job_dir / n).exists():
            try:
                shutil.copy2(analysis / n, job_dir / n)
            except OSError:
                pass
    return load_profiles(job_dir)


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
