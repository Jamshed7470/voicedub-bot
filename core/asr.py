"""Распознавание речи с пословными таймингами (WhisperX)."""
from __future__ import annotations

import gc
import logging

from core.errors import UserError

log = logging.getLogger(__name__)


def transcribe(analysis_wav: str, cfg, progress=None) -> dict:
    """Распознаёт речь: автоопределение языка + word-level alignment.

    Возвращает dict: {"language": str, "segments": [{start, end, text, words: [...]}]}.
    """
    import torch
    import whisperx

    device = cfg.device
    compute_type = (cfg.y("asr", "compute_type_gpu", default="int8_float16")
                    if device == "cuda"
                    else cfg.y("asr", "compute_type_cpu", default="int8"))
    batch_size = int(cfg.y("asr", "batch_size", default=8))

    log.info("WhisperX: модель %s, устройство %s, compute_type %s",
             cfg.whisper_model, device, compute_type)
    model = whisperx.load_model(cfg.whisper_model, device, compute_type=compute_type)
    audio = whisperx.load_audio(str(analysis_wav))

    if progress:
        progress(30)
    result = model.transcribe(audio, batch_size=batch_size)
    language = result.get("language")
    segments = result.get("segments") or []
    if not segments or not any((s.get("text") or "").strip() for s in segments):
        raise UserError(
            "В этом файле не нашлось речи — нечего дублировать. "
            "Проверьте, что в видео/аудио кто-то говорит."
        )

    if progress:
        progress(70)
    # word-level alignment — тайминги слов нужны для точной нарезки сегментов
    try:
        align_model, metadata = whisperx.load_align_model(language_code=language,
                                                          device=device)
        result = whisperx.align(segments, align_model, metadata, audio, device,
                                return_char_alignments=False)
        result["language"] = language
        del align_model
    except Exception:  # для редких языков alignment-модели может не быть
        log.exception("Alignment не удался, использую тайминги Whisper как есть")
        result = {"language": language, "segments": segments}

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result
