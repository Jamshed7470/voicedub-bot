"""XTTS-v2 как реализация TTSEngine.

Отличие от старого core/tts.py: модель получает готовые латенты профиля и
никогда — путь к референсу. Пересчёт тембра на лету физически невозможен.
"""
from __future__ import annotations

import logging
from pathlib import Path

from synth.engine_base import TTSEngine, split_text

log = logging.getLogger(__name__)

MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"

# лимиты XTTS-v2 на длину строки: длиннее — модель обрезает звук,
# а на одном сверхдлинном предложении падает
CHAR_LIMITS = {
    "en": 250, "de": 253, "fr": 273, "es": 239, "it": 213, "pt": 203,
    "pl": 224, "zh": 82, "zh-cn": 82, "ar": 166, "cs": 186, "ru": 182,
    "nl": 251, "tr": 226, "ja": 71, "hu": 224, "ko": 95, "hi": 150,
}


class XTTSEngine(TTSEngine):
    name = "xtts_v2"
    sample_rate = 24000

    def __init__(self, cfg):
        self.cfg = cfg
        self._tts = None

    # ---------- модель ----------

    @property
    def tts(self):
        if self._tts is None:
            import torch  # noqa: F401
            from TTS.api import TTS

            log.info("XTTS-v2: загружаю модель (устройство %s)…", self.cfg.device)
            self._tts = TTS(MODEL_ID).to(self.cfg.device)
        return self._tts

    @property
    def model(self):
        return self.tts.synthesizer.tts_model

    @property
    def conf(self):
        return self.tts.synthesizer.tts_config

    @property
    def model_version(self) -> str:
        return str(getattr(self.conf, "model", "xtts")) + "/" + MODEL_ID.rsplit("/", 1)[-1]

    # ---------- профиль ----------

    def build_conditioning(self, ref_wav: str | Path):
        """Референс → латенты. Единственный вызов get_conditioning_latents."""
        conf = self.conf
        return self.model.get_conditioning_latents(
            audio_path=[str(ref_wav)],
            gpt_cond_len=conf.gpt_cond_len,
            gpt_cond_chunk_len=conf.gpt_cond_chunk_len,
            max_ref_length=conf.max_ref_len,
            sound_norm_refs=conf.sound_norm_refs,
        )

    # ---------- синтез ----------

    def synthesize(self, text: str, lang: str, profile, out_path: str | Path,
                   speed: float = 1.0, seed: int | None = None,
                   temperature: float | None = None) -> Path:
        import numpy as np
        import soundfile as sf
        import torch

        from core.normalize import assert_no_digits

        self.require_profile(profile)           # INV-2
        assert_no_digits(text)                  # цифры в TTS не попадают

        out_path = Path(out_path)
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        temp = float(temperature if temperature is not None
                     else self.cfg.y("synthesis", "temperature", default=0.55))
        limit = min(CHAR_LIMITS.get(lang, 200),
                    int(self.cfg.y("synthesis", "max_chars_per_call", default=220)))
        join = float(self.cfg.y("synthesis", "chunk_join_sec", default=0.12))
        chunks = split_text(text, limit)
        if not chunks:
            raise RuntimeError("Пустой текст для синтеза")

        pieces = []
        for chunk in chunks:
            pieces.append(self._infer(chunk, lang, profile, speed, temp))
            if len(chunks) > 1:
                pieces.append(np.zeros(int(join * self.sample_rate), dtype=np.float32))
        if len(chunks) > 1:
            pieces.pop()      # хвостовая пауза не нужна
            log.debug("Реплика длиннее лимита XTTS — синтезирована %d частями",
                      len(chunks))

        wav = np.concatenate(pieces).astype(np.float32)
        if not wav.size:
            raise RuntimeError("XTTS вернул пустой звук")
        sf.write(str(out_path), wav, self.sample_rate)
        return out_path

    def _infer(self, text: str, lang: str, profile, speed: float, temp: float):
        import numpy as np

        conf = self.conf
        try:
            out = self.model.inference(
                text=text, language=lang,
                gpt_cond_latent=profile.gpt_cond_latent,
                speaker_embedding=profile.speaker_embedding,
                temperature=temp,
                length_penalty=conf.length_penalty,
                repetition_penalty=float(conf.repetition_penalty),
                top_k=conf.top_k, top_p=conf.top_p,
                speed=max(0.9, min(1.3, float(speed))),
                enable_text_splitting=False,
            )
        except AssertionError as e:   # внутренние проверки XTTS не валят задачу
            raise RuntimeError(f"XTTS не смог синтезировать реплику: {e}") from e
        return np.asarray(out["wav"], dtype=np.float32)

    def unload(self) -> None:
        """Освобождает видеопамять между тяжёлыми стадиями."""
        self._tts = None


_engine: XTTSEngine | None = None


def get_engine(cfg) -> XTTSEngine:
    """Общий движок на процесс: модель XTTS весит несколько гигабайт."""
    global _engine
    if _engine is None:
        _engine = XTTSEngine(cfg)
    return _engine
