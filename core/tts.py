"""Локальный синтез речи: XTTS-v2 (coqui-tts) с клонированием голоса.

Каждый спикер говорит СВОИМ голосом: все референсы взяты из его собственной
речи. Style-референс сегмента (для сохранения эмоции):
1) если оригинальный звук ЭТОГО ЖЕ сегмента >= 3 с — используем его;
2) иначе — референс спикера с той же меткой эмоции;
3) иначе — ref_main.wav спикера.
Если референс спикера слишком короткий/грязный — пресет-голос XTTS по полу.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"

# встроенные голоса XTTS-v2 (резерв, если референс спикера непригоден)
PRESET_VOICES = {
    "male": ["Damien Black", "Viktor Eka", "Luis Moray", "Badr Odhiambo",
             "Andrew Chipper", "Eugenio Mataracı"],
    "female": ["Claribel Dervla", "Daisy Studious", "Gracie Wise",
               "Alison Dietlinde", "Ana Florence", "Sofia Hellen"],
}

_tts = None


def _get_tts(cfg):
    global _tts
    if _tts is None:
        import torch  # noqa: F401
        from TTS.api import TTS
        log.info("XTTS-v2: загружаю модель (устройство %s)…", cfg.device)
        _tts = TTS(MODEL_ID).to(cfg.device)
    return _tts


def choose_style_ref(seg: dict, profile: dict, vocals_wav: str | Path,
                     tmp_dir: str | Path, cfg) -> tuple[str | None, str | None]:
    """Возвращает (speaker_wav, preset_name) — ровно одно из двух не None."""
    from core.media import cut_fragment

    min_ref = float(cfg.y("speaker_ref", "min_ref_s", default=3.0))
    gender = profile.get("gender", "male")

    # референс спикера непригоден → пресет по полу (стабильный для спикера)
    if not profile.get("ref_ok"):
        presets = PRESET_VOICES.get(gender, PRESET_VOICES["male"])
        idx = (int(profile["id"][1:]) - 1) % len(presets)
        return None, presets[idx]

    dur = seg["end"] - seg["start"]
    # 1) оригинальный звук этого же сегмента — эмоция переносится напрямую
    if dur >= min_ref:
        piece = Path(tmp_dir) / f"style_{seg['id']}.wav"
        try:
            cut_fragment(vocals_wav, piece, seg["start"], seg["end"],
                         sr=24000, mono=True)
            return str(piece), None
        except Exception:  # noqa: BLE001
            log.exception("Не удалось вырезать style-референс сегмента %s", seg["id"])

    # 2) референс спикера с той же эмоцией
    emo = seg.get("emotion", "neutral")
    emo_ref = (profile.get("refs_emotion") or {}).get(emo)
    if emo_ref and Path(emo_ref).exists():
        return emo_ref, None

    # 3) основной референс спикера
    return profile["ref_main"], None


def synthesize(text: str, lang: str, out_path: str | Path, cfg,
               speaker_wav: str | None = None, preset: str | None = None,
               speed: float = 1.0) -> Path:
    """Синтезирует text в out_path. Возвращает путь к wav (24 кГц mono)."""
    from core.normalize import assert_no_digits

    assert_no_digits(text)  # финальная гарантия: цифры в TTS не попадают
    tts = _get_tts(cfg)
    kwargs = {
        "text": text,
        "language": lang,
        "file_path": str(out_path),
        "speed": max(0.9, min(1.3, speed)),
    }
    if speaker_wav:
        kwargs["speaker_wav"] = speaker_wav
    else:
        kwargs["speaker"] = preset or PRESET_VOICES["male"][0]
    tts.tts_to_file(**kwargs)
    return Path(out_path)
