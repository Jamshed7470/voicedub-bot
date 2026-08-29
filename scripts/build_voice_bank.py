"""Сборка банка пресет-голосов.

Два источника:

    python -m scripts.build_voice_bank --from-xtts
        встроенные голоса XTTS-v2: для каждого синтезируется нейтральная
        фраза, определяется пол, считаются латенты и отпечаток.

    python -m scripts.build_voice_bank --from-dir voice_db
        свои записи (30–60 с чистой речи), имя файла становится названием
        голоса. Пол берётся из префикса male_/female_ или определяется по
        основному тону.

Голос банка хранится ровно как профиль спикера: с готовыми латентами и
эталонным отпечатком. Поэтому назначение голоса в студии мгновенно, а
синтез пресетом так же заблокирован, как и клон (INV-1, INV-2).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config          # noqa: E402
from voices.bank import BANK_DIR            # noqa: E402

log = logging.getLogger("bank")

# нейтральные фразы: референс должен быть спокойным, иначе эмоция
# закрепится за голосом навсегда
PHRASES = {
    "ru": ("Сегодня в городе ясная погода, и до самого вечера обещают тепло. "
           "На улицах много людей, работают кафе, ходят автобусы."),
    "en": ("The weather in the city is clear today, and it should stay warm "
           "until evening. The streets are busy and the cafes are open."),
}

# понятные русские названия вместо оригинальных имён пресетов XTTS
GENDER_RU = {"male": "Мужской", "female": "Женский", "unknown": "Голос"}
TIMBRE_BY_F0 = [
    (0, 105, "низкий"), (105, 135, "средний"), (135, 165, "светлый"),
    (165, 200, "низкий"), (200, 235, "средний"), (235, 400, "высокий"),
]


def timbre_word(f0: float | None) -> str:
    if not f0:
        return "ровный"
    for lo, hi, word in TIMBRE_BY_F0:
        if lo <= f0 < hi:
            return word
    return "ровный"


def build_from_xtts(cfg, limit: int = 0, langs: tuple[str, ...] = ("ru", "en")) -> int:
    """Проходит по встроенным спикерам XTTS-v2 и делает из них банк."""
    import numpy as np
    import soundfile as sf

    from core.gender import detect_gender
    from core.voicebank import measure_f0
    from identity.embeddings import get_embedder
    from synth.xtts_engine import get_engine
    from voices.profiles import build_profile, save_profile

    engine = get_engine(cfg)
    embedder = get_embedder(cfg)
    manager = getattr(engine.model, "speaker_manager", None)
    names = list(getattr(manager, "speakers", {}) or {})
    if not names:
        log.error("У модели XTTS не нашлось встроенных голосов")
        return 0

    if limit:
        names = names[:limit]
    log.info("Встроенных голосов XTTS: %d", len(names))

    BANK_DIR.mkdir(parents=True, exist_ok=True)
    built = 0
    for i, name in enumerate(names, 1):
        voice_id = "xtts_" + _slug(name)
        out_dir = BANK_DIR / voice_id
        if (out_dir / "profile_xtts.pt").exists():
            log.info("[%d/%d] %s — уже собран, пропускаю", i, len(names), name)
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        sample = out_dir / "sample.wav"
        try:
            pieces = []
            for lang in langs:
                part = out_dir / f"_tmp_{lang}.wav"
                engine.tts.tts_to_file(text=PHRASES[lang], speaker=name,
                                       language=lang, file_path=str(part))
                data, sr = sf.read(str(part), dtype="float32")
                pieces.append(data)
                part.unlink(missing_ok=True)
            sf.write(str(sample), np.concatenate(pieces), 24000)

            f0 = measure_f0(sample)
            y16 = _load16(sample)
            g = detect_gender(y16, 16000,
                              [{"start": 0.0, "end": len(y16) / 16000}], cfg)

            profile = build_profile(voice_id, sample, engine, embedder,
                                    mode="preset")
            profile.preset_id = voice_id
            save_profile(profile, out_dir)
            shutil.copyfile(out_dir / "voice_profile.pt", out_dir / "profile_xtts.pt")
            shutil.copyfile(out_dir / "identity_embedding.npy", out_dir / "identity.npy")

            display = f"{GENDER_RU.get(g['gender'], 'Голос')} {timbre_word(f0)} {i}"
            _write_meta(out_dir, {
                "id": voice_id, "display_name": display,
                "gender": g["gender"], "age_group": g.get("age", "adult"),
                "timbre_tags": [timbre_word(f0)], "languages": list(langs),
                "source": "xtts_builtin", "source_id": name,
                "license_note": "встроенный голос модели XTTS-v2",
                "f0_hz": round(f0, 1) if f0 else None,
            })
            built += 1
            log.info("[%d/%d] %s → «%s» (%s, %s Гц)", i, len(names), name,
                     display, g["gender"], f"{f0:.0f}" if f0 else "?")
        except Exception:  # noqa: BLE001 — один сломанный голос не валит сборку
            log.exception("Голос %s собрать не удалось", name)
            shutil.rmtree(out_dir, ignore_errors=True)
    return built


def build_from_dir(cfg, source: Path, gender_override: str | None = None) -> int:
    """Собирает банк из пользовательских записей."""
    from core.gender import detect_gender
    from core.voicebank import _gender_from_name, measure_f0
    from identity.embeddings import get_embedder
    from synth.xtts_engine import get_engine
    from voices.profiles import build_profile, save_profile

    exts = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".opus"}
    files = sorted(p for p in source.iterdir()
                   if p.is_file() and p.suffix.lower() in exts)
    if not files:
        log.error("В папке %s нет звуковых файлов", source)
        return 0

    engine = get_engine(cfg)
    embedder = get_embedder(cfg)
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    built = 0

    for i, path in enumerate(files, 1):
        voice_id = _slug(path.stem)
        out_dir = BANK_DIR / voice_id
        if (out_dir / "profile_xtts.pt").exists():
            log.info("[%d/%d] %s — уже собран", i, len(files), path.name)
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            sample = out_dir / "sample.wav"
            _to_wav(path, sample)

            f0 = measure_f0(sample)
            gender = (gender_override or _gender_from_name(path.name)
                      or detect_gender(_load16(sample), 16000,
                                       [{"start": 0.0, "end": 20.0}], cfg)["gender"])

            profile = build_profile(voice_id, sample, engine, embedder,
                                    mode="preset")
            profile.preset_id = voice_id
            save_profile(profile, out_dir)
            shutil.copyfile(out_dir / "voice_profile.pt", out_dir / "profile_xtts.pt")
            shutil.copyfile(out_dir / "identity_embedding.npy", out_dir / "identity.npy")

            _write_meta(out_dir, {
                "id": voice_id,
                "display_name": _display(path.stem),
                "gender": gender,
                "age_group": "child" if "child" in path.stem.lower() else "adult",
                "timbre_tags": [timbre_word(f0)],
                "languages": ["ru"], "source": "user",
                "source_id": path.name, "license_note": "запись пользователя",
                "f0_hz": round(f0, 1) if f0 else None,
            })
            built += 1
            log.info("[%d/%d] %s → «%s» (%s, %s Гц)", i, len(files), path.name,
                     _display(path.stem), gender, f"{f0:.0f}" if f0 else "?")
        except Exception:  # noqa: BLE001
            log.exception("Голос %s собрать не удалось", path.name)
            shutil.rmtree(out_dir, ignore_errors=True)
    return built


# ---------------------------------------------------------------- утилиты

def _slug(name: str) -> str:
    out = "".join(c if c.isalnum() else "_" for c in name.lower())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "voice"


def _display(stem: str) -> str:
    for prefix in ("male_", "female_", "m_", "f_", "м_", "ж_"):
        if stem.lower().startswith(prefix):
            stem = stem[len(prefix):]
            break
    return stem.replace("_", " ").strip().capitalize() or stem


def _write_meta(out_dir: Path, meta: dict) -> None:
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _to_wav(src: Path, dst: Path, seconds: float = 25.0) -> None:
    from core.media import run

    run(["ffmpeg", "-y", "-i", str(src), "-t", f"{seconds}", "-ac", "1",
         "-ar", "24000", "-c:a", "pcm_s16le", str(dst)], desc="банк: образец")


def _load16(path: Path):
    import librosa

    return librosa.load(str(path), sr=16000, mono=True)[0]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Сборка банка голосов VoiceDub")
    ap.add_argument("--from-xtts", action="store_true",
                    help="взять встроенные голоса XTTS-v2")
    ap.add_argument("--from-dir", type=Path,
                    help="папка со своими записями (30-60 с чистой речи)")
    ap.add_argument("--gender", choices=["male", "female"],
                    help="задать пол всем голосам папки принудительно")
    ap.add_argument("--limit", type=int, default=0,
                    help="ограничить число голосов (для проверки)")
    args = ap.parse_args()

    if not args.from_xtts and not args.from_dir:
        ap.error("укажите --from-xtts или --from-dir")

    cfg = load_config()
    built = 0
    if args.from_xtts:
        built += build_from_xtts(cfg, limit=args.limit)
    if args.from_dir:
        if not args.from_dir.exists():
            log.error("Папка %s не найдена", args.from_dir)
            return 1
        built += build_from_dir(cfg, args.from_dir, args.gender)

    from voices.bank import VoiceBank

    total = len(VoiceBank().load(refresh=True))
    print(f"\nСобрано голосов за этот запуск: {built}")
    print(f"Всего в банке: {total} → {BANK_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
