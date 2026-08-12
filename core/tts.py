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


# лимиты XTTS-v2 на длину строки (символы). Длиннее — модель обрезает звук,
# а на одном сверхдлинном предложении падает с «maximum of 400 tokens»
CHAR_LIMITS = {
    "en": 250, "de": 253, "fr": 273, "es": 239, "it": 213, "pt": 203,
    "pl": 224, "zh": 82, "zh-cn": 82, "ar": 166, "cs": 186, "ru": 182,
    "nl": 251, "tr": 226, "ja": 71, "hu": 224, "ko": 95, "hi": 150,
}

SENTENCE_END = ("...", ".", "!", "?", "…", ";")


def split_for_tts(text: str, lang: str) -> list[str]:
    """Режет реплику на куски по силе разрыва: предложения, запятые, слова.

    XTTS сам делит текст по предложениям, но одно длинное предложение
    (например, зациклившийся перевод) он не осилит и уронит синтез.
    """
    limit = CHAR_LIMITS.get(lang, 200)
    text = text.strip()
    if len(text) <= limit:
        return [text]

    def cut(parts: list[str], seps: tuple[str, ...]) -> list[str]:
        out: list[str] = []
        for part in parts:
            if len(part) <= limit:
                out.append(part)
                continue
            buf = ""
            for piece in _split_keep(part, seps):
                if buf and len(buf) + len(piece) > limit:
                    out.append(buf.strip())
                    buf = piece
                else:
                    buf += piece
            if buf.strip():
                out.append(buf.strip())
        return out

    chunks = cut([text], SENTENCE_END)
    chunks = cut(chunks, (",", ":", "—", "–"))
    chunks = cut(chunks, (" ",))
    # слово длиннее лимита — режем жёстко, иначе синтез снова упадёт
    final: list[str] = []
    for c in chunks:
        while len(c) > limit:
            final.append(c[:limit])
            c = c[limit:]
        if c.strip():
            final.append(c.strip())
    return final or [text[:limit]]


def _split_keep(text: str, seps: tuple[str, ...]) -> list[str]:
    """Делит текст, оставляя разделитель в конце куска."""
    parts, buf = [], ""
    for ch in text:
        buf += ch
        if ch in seps or any(buf.endswith(s) for s in seps if len(s) > 1):
            parts.append(buf)
            buf = ""
    if buf:
        parts.append(buf)
    return parts


# отпечатки голоса по пути к референсу: их расчёт занимает около секунды,
# а в банке голосов и у коротких реплик референс один и тот же на весь ролик
_LATENTS: dict[str, tuple] = {}
# спикеров в ролике бывает и десяток: при меньшем кэше реплики разных голосов
# вытесняют друг друга, и отпечаток считается заново на каждой реплике
_LATENTS_MAX = 32


def _conditioning(tts, speaker_wav: str):
    """Отпечаток голоса из референса, с кэшем (LRU по числу спикеров)."""
    key = str(speaker_wav)
    cached = _LATENTS.pop(key, None)
    if cached is None:
        model = tts.synthesizer.tts_model
        conf = tts.synthesizer.tts_config
        cached = model.get_conditioning_latents(
            audio_path=[key],
            gpt_cond_len=conf.gpt_cond_len,
            gpt_cond_chunk_len=conf.gpt_cond_chunk_len,
            max_ref_length=conf.max_ref_len,
            sound_norm_refs=conf.sound_norm_refs,
        )
    _LATENTS[key] = cached
    while len(_LATENTS) > _LATENTS_MAX:
        _LATENTS.pop(next(iter(_LATENTS)))
    return cached


def _tts_cached_ref(tts, text: str, lang: str, path: Path, speaker_wav: str,
                    speed: float) -> None:
    """Синтез с готовым отпечатком голоса.

    Параметры генерации берутся из конфига модели — те же, что подставляет
    штатный synthesize(): иначе звук поедет (например, штраф за повторы
    по умолчанию вдвое строже конфигурационного).
    """
    import numpy as np
    import soundfile as sf

    model = tts.synthesizer.tts_model
    conf = tts.synthesizer.tts_config
    gpt_latent, speaker_emb = _conditioning(tts, speaker_wav)
    out = model.inference(
        text=text, language=lang,
        gpt_cond_latent=gpt_latent, speaker_embedding=speaker_emb,
        temperature=conf.temperature, length_penalty=conf.length_penalty,
        repetition_penalty=float(conf.repetition_penalty),
        top_k=conf.top_k, top_p=conf.top_p, speed=speed,
        enable_text_splitting=False,
    )
    wav = np.asarray(out["wav"], dtype=np.float32)
    if not wav.size:
        raise RuntimeError("XTTS вернул пустой звук")
    sf.write(str(path), wav, 24000)


def synthesize(text: str, lang: str, out_path: str | Path, cfg,
               speaker_wav: str | None = None, preset: str | None = None,
               speed: float = 1.0) -> Path:
    """Синтезирует text в out_path. Возвращает путь к wav (24 кГц mono)."""
    from core.normalize import assert_no_digits

    assert_no_digits(text)  # финальная гарантия: цифры в TTS не попадают
    tts = _get_tts(cfg)
    out_path = Path(out_path)
    kwargs = {
        "language": lang,
        "speed": max(0.9, min(1.3, speed)),
    }
    if speaker_wav:
        kwargs["speaker_wav"] = speaker_wav
    else:
        kwargs["speaker"] = preset or PRESET_VOICES["male"][0]

    use_cache = bool(cfg.y("tts", "cache_speaker_latents", default=True))
    fast_ref = speaker_wav if (use_cache and speaker_wav) else None

    def _one(text_part: str, path: Path) -> None:
        if fast_ref:
            try:
                _tts_cached_ref(tts, text_part, lang, path,
                                fast_ref, kwargs["speed"])
                return
            except Exception:  # noqa: BLE001 — откат на штатный путь синтеза
                log.exception("Синтез с кэшем отпечатка не удался, иду штатным путём")
        _tts_to_file(tts, text_part, path, kwargs)

    chunks = split_for_tts(text, lang)
    if len(chunks) == 1:
        _one(chunks[0], out_path)
        return out_path

    log.info("Реплика длиннее лимита XTTS — синтезирую %d частями", len(chunks))
    import numpy as np
    import soundfile as sf

    pieces, sr = [], 24000
    for i, chunk in enumerate(chunks):
        part = out_path.with_name(f"{out_path.stem}_p{i}{out_path.suffix}")
        _one(chunk, part)
        data, sr = sf.read(str(part), dtype="float32")
        pieces.append(data)
        part.unlink(missing_ok=True)
    sf.write(str(out_path), np.concatenate(pieces), sr)
    return out_path


def _tts_to_file(tts, text: str, path: Path, kwargs: dict) -> None:
    """Обёртка синтеза: внутренние assert'ы XTTS не должны валить задачу."""
    try:
        tts.tts_to_file(text=text, file_path=str(path), **kwargs)
    except AssertionError as e:  # «maximum of 400 tokens» и подобное
        raise RuntimeError(f"XTTS не смог синтезировать реплику: {e}") from e
