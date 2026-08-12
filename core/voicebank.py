"""Банк голосов: подбор голоса из voice_db/ по основному тону (Гц).

Пользователь кладёт образцы голосов (wav/mp3/m4a/ogg, 10–30 с чистой речи)
в voice_db/ (или data/voices/). При включённом режиме «голоса из банка»
каждому спикеру ролика подбирается ближайший по частоте основного тона
голос того же пола; XTTS озвучивает этим голосом вместо клона оригинала.

Метаданные (тон, пол) считаются один раз и кэшируются.
Пол задаётся префиксом имени файла: "male_"/"m_" / "female_"/"f_"
(иначе определяется по тону).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from core.config import DATA_DIR, ROOT

log = logging.getLogger(__name__)

VOICE_DIRS = [ROOT / "voice_db", DATA_DIR / "voices"]
META_FILE = DATA_DIR / "voicebank_meta.json"
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".opus"}
GENDER_PREFIXES = {"male": ("male_", "m_", "м_"),
                   "female": ("female_", "f_", "ж_")}
# граница мужской/женский по медианному F0, если пол не задан префиксом
GENDER_F0_SPLIT = 165.0
# выше этого медианного F0 спикер считается ребёнком (для подбора детского голоса)
CHILD_F0_MIN = 260.0


def measure_f0(path: str | Path, max_seconds: float = 25.0) -> float | None:
    """Медианная частота основного тона голоса в Гц (None — тона не нашлось)."""
    import librosa
    import numpy as np

    y, sr = librosa.load(str(path), sr=16000, mono=True,
                         duration=max_seconds)
    if not len(y):
        return None
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=420, sr=sr)
    voiced_f0 = f0[voiced & ~np.isnan(f0)] if f0 is not None else []
    if len(voiced_f0) < 10:
        return None
    return float(np.median(voiced_f0))


def _gender_from_name(name: str) -> str | None:
    low = name.lower()
    for gender, prefixes in GENDER_PREFIXES.items():
        if low.startswith(prefixes):
            return gender
    return None


def _display_name(stem: str) -> str:
    low = stem.lower()
    for prefixes in GENDER_PREFIXES.values():
        for pre in prefixes:
            if low.startswith(pre):
                return stem[len(pre):] or stem
    return stem


def scan() -> list[dict]:
    """Список голосов банка: [{name, path, f0_hz, gender}]. Метаданные кэшируются."""
    try:
        meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}

    voices, dirty = [], False
    files = [p for d in VOICE_DIRS if d.exists()
             for p in sorted(d.iterdir())
             if p.suffix.lower() in AUDIO_EXT and p.is_file()]
    for p in files:
        key = str(p)
        m = meta.get(key)
        if not m or m.get("mtime") != int(p.stat().st_mtime):
            f0 = None
            try:
                f0 = measure_f0(p)
            except Exception:  # noqa: BLE001
                log.exception("Банк голосов: не удалось измерить тон %s", p.name)
            if f0 is None:
                log.warning("Банк голосов: в %s не нашлось речи — пропускаю", p.name)
                meta[key] = {"mtime": int(p.stat().st_mtime), "f0_hz": None}
                dirty = True
                continue
            gender = _gender_from_name(p.name) or (
                "male" if f0 < GENDER_F0_SPLIT else "female")
            meta[key] = {"mtime": int(p.stat().st_mtime),
                         "f0_hz": round(f0, 1), "gender": gender}
            dirty = True
            log.info("Банк голосов: %s — %.0f Гц, %s", p.name, f0, gender)
        m = meta[key]
        if m.get("f0_hz"):
            voices.append({"name": _display_name(p.stem), "path": str(p),
                           "f0_hz": m["f0_hz"], "gender": m["gender"],
                           # детский голос помечается словом child в имени файла
                           # (male_child_… / female_child_…): пол берётся из префикса
                           "is_child": "child" in p.stem.lower()})
    if dirty:
        try:
            META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        except OSError:
            pass
    return voices


def count() -> int:
    return len(scan())


def _speaker_is_child(prof: dict, f0: float | None) -> bool:
    """Ребёнок ли спикер: сперва по метке из анализа пола, иначе по F0."""
    age = prof.get("age")
    if age in ("child", "adult"):
        return age == "child"
    f0v = prof.get("f0_median") or f0
    return bool(f0v and f0v >= CHILD_F0_MIN)


def assign(profiles: dict) -> dict[str, dict]:
    """Подбирает каждому спикеру голос банка: тот же пол и возраст, ближайший тон.

    Возвращает {speaker_id: voice_dict}; пустой dict — банк пуст.
    Правила:
      • пол — жёсткий: женскому спикеру только женский голос (чужой пол —
        лишь если своего в банке нет вовсе);
      • возраст — детскому спикеру детский голос, взрослому — взрослый
        (в рамках своего пола), пока такие голоса есть;
      • уникальность — пока голосов хватает, один голос не достаётся двум
        спикерам; повтор включается, только когда свободные кончились.
    """
    voices = scan()
    if not voices:
        return {}

    def speaker_f0(prof: dict) -> float | None:
        ref = prof.get("ref_main")
        if prof.get("ref_ok") and ref and Path(ref).exists():
            try:
                return measure_f0(ref)
            except Exception:  # noqa: BLE001
                log.exception("Банк голосов: не измерить тон спикера %s",
                              prof.get("id"))
        return prof.get("f0_median")

    def distance(voice: dict, f0: float | None) -> float:
        if not f0:
            return 0.0
        return abs(math.log2(voice["f0_hz"] / f0))       # близость тона в октавах

    def nearest(pool: list[dict], f0: float | None) -> dict | None:
        return min(pool, key=lambda v: distance(v, f0)) if pool else None

    n_spk, n_voice = len(profiles), len(voices)
    if n_spk > n_voice:
        log.warning("Банк голосов: спикеров %d, а голосов %d — некоторым "
                    "придётся повторно выдать уже занятый голос", n_spk, n_voice)

    result: dict[str, dict] = {}
    used: set[str] = set()
    for spk, prof in profiles.items():
        f0 = speaker_f0(prof)
        gender = prof.get("gender", "male")
        is_child = _speaker_is_child(prof, f0)

        same = [v for v in voices if v["gender"] == gender]
        age_match = [v for v in same if v["is_child"] == is_child]
        free = lambda pool: [v for v in pool if v["path"] not in used]  # noqa: E731
        # приоритет: свой пол+возраст (свободный) → свой пол любой возраст
        # (свободный) → свой пол+возраст (повтор) → свой пол (повтор)
        # → другой пол (свободный) → что угодно
        best = None
        for pool in (free(age_match), free(same), age_match, same,
                     free(voices), voices):
            best = nearest(pool, f0)
            if best:
                break

        used.add(best["path"])
        result[spk] = best
        log.info("Банк голосов: %s (%s%s, %s Гц) → «%s» (%s%s, %.0f Гц)",
                 spk, gender, ", ребёнок" if is_child else "",
                 f"{f0:.0f}" if f0 else "?", best["name"], best["gender"],
                 ", ребёнок" if best["is_child"] else "", best["f0_hz"])
    return result
