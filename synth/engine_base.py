"""Абстракция движка синтеза.

Смысл слоя — не в поддержке многих движков (сейчас он один, XTTS-v2), а в
том, что через него физически невозможно синтезировать реплику без
заблокированного профиля: `synthesize` принимает VoiceProfile, а не путь к
wav. Это INV-2, выраженный типом аргумента, а не соглашением.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path


class TTSEngine(ABC):
    """Движок синтеза с фиксированными профилями голоса."""

    name: str = "base"
    sample_rate: int = 24000

    @abstractmethod
    def build_conditioning(self, ref_wav: str | Path):
        """Референс → (gpt_cond_latent, speaker_embedding).

        Вызывается ТОЛЬКО из voices/profiles.py и scripts/build_voice_bank.py —
        один раз на спикера или голос банка.
        """

    @abstractmethod
    def synthesize(self, text: str, lang: str, profile, out_path: str | Path,
                   speed: float = 1.0, seed: int | None = None,
                   temperature: float | None = None) -> Path:
        """Синтезирует реплику голосом профиля."""

    # ---- общее для всех движков ----

    @staticmethod
    def make_seed(job_id: str, segment_id: int, attempt: int) -> int:
        """Детерминированный seed: один и тот же вход даёт один и тот же звук.

        Без этого повторный рендер после правки одной реплики перегенерировал
        бы все остальные чуть иначе, и сравнить два прогона было бы нечем.
        """
        raw = f"{job_id}:{segment_id}:{attempt}".encode()
        return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")

    @staticmethod
    def require_profile(profile) -> None:
        """INV-2: синтез без заблокированного профиля запрещён."""
        assert profile is not None, "синтез без профиля голоса запрещён (INV-2)"
        assert getattr(profile, "locked", False), (
            f"профиль {getattr(profile, 'speaker_id', '?')} не заблокирован — "
            "его нельзя использовать для синтеза (INV-2)")
        assert profile.gpt_cond_latent is not None, (
            "в профиле нет латентов голоса — профиль собран не полностью")


def split_text(text: str, limit: int) -> list[str]:
    """Режет длинную реплику по силе разрыва: предложения → запятые → слова.

    XTTS падает на одном сверхдлинном предложении («maximum of 400 tokens»),
    а зациклившийся перевод такие предложения даёт регулярно.
    """
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []

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

    chunks = cut([text], ("...", ".", "!", "?", "…", ";"))
    chunks = cut(chunks, (",", ":", "—", "–"))
    chunks = cut(chunks, (" ",))

    final: list[str] = []
    for c in chunks:
        while len(c) > limit:      # слово длиннее лимита — режем жёстко
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
