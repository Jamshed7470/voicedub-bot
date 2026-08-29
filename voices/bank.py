"""Банк пресет-голосов: voices/bank/<voice_id>/.

Каждый голос лежит вместе с уже посчитанными латентами XTTS и отпечатком
ECAPA — поэтому назначение голоса спикеру не требует ни синтеза, ни
пересчёта тембра, а профиль пресета так же заблокирован, как и клон.

Банк собирается скриптом scripts/build_voice_bank.py: из встроенных
голосов XTTS-v2 или из своих записей. Старая папка voice_db/ (голоса,
размеченные префиксом имени файла) читается как запасной источник, чтобы
уже собранные пользователем 55 голосов не пропали.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.config import ROOT

log = logging.getLogger(__name__)

BANK_DIR = ROOT / "voices" / "bank"
LEGACY_DIRS = [ROOT / "voice_db"]
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".opus"}


@dataclass
class BankVoice:
    """Голос банка вместе с метаданными."""
    id: str
    display_name: str
    gender: str = "unknown"
    age_group: str = "adult"
    timbre_tags: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    source: str = "user"
    source_id: str | None = None
    license_note: str = ""
    dir: Path | None = None
    sample_path: Path | None = None
    identity: np.ndarray | None = None
    f0_hz: float | None = None

    @property
    def is_child(self) -> bool:
        return self.age_group == "child"

    @property
    def has_profile(self) -> bool:
        return bool(self.dir and (self.dir / "profile_xtts.pt").exists())

    def to_dict(self) -> dict:
        return {
            "id": self.id, "display_name": self.display_name,
            "gender": self.gender, "age_group": self.age_group,
            "timbre_tags": self.timbre_tags, "languages": self.languages,
            "source": self.source, "source_id": self.source_id,
            "has_profile": self.has_profile,
            "f0_hz": self.f0_hz,
        }


class VoiceBank:
    """Читает банк с диска и отдаёт готовые профили голосов."""

    def __init__(self, bank_dir: Path | None = None):
        self.dir = Path(bank_dir or BANK_DIR)
        self._voices: dict[str, BankVoice] | None = None
        self._profiles: dict[str, object] = {}

    # ---------- чтение ----------

    def load(self, refresh: bool = False) -> dict[str, BankVoice]:
        if self._voices is not None and not refresh:
            return self._voices

        voices: dict[str, BankVoice] = {}
        if self.dir.exists():
            for entry in sorted(self.dir.iterdir()):
                if not entry.is_dir():
                    continue
                voice = self._read_voice(entry)
                if voice:
                    voices[voice.id] = voice

        if not voices:
            voices = self._read_legacy()
        self._voices = voices
        log.info("Банк голосов: %d голосов%s", len(voices),
                 "" if voices else " — пусто, соберите: python -m scripts.build_voice_bank")
        return voices

    def _read_voice(self, entry: Path) -> BankVoice | None:
        meta_path = entry / "meta.json"
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.warning("Банк: не читается %s", meta_path)
            return None

        identity = None
        ident_path = entry / "identity.npy"
        if ident_path.exists():
            try:
                identity = np.load(str(ident_path)).astype(np.float32)
            except (OSError, ValueError):
                log.warning("Банк: не читается отпечаток голоса %s", entry.name)

        sample = entry / "sample.wav"
        return BankVoice(
            id=meta.get("id", entry.name),
            display_name=meta.get("display_name", entry.name),
            gender=meta.get("gender", "unknown"),
            age_group=meta.get("age_group", "adult"),
            timbre_tags=meta.get("timbre_tags", []),
            languages=meta.get("languages", []),
            source=meta.get("source", "user"),
            source_id=meta.get("source_id"),
            license_note=meta.get("license_note", ""),
            dir=entry,
            sample_path=sample if sample.exists() else None,
            identity=identity,
            f0_hz=meta.get("f0_hz"),
        )

    def _read_legacy(self) -> dict[str, BankVoice]:
        """Старая папка voice_db/: пол и возраст берутся из имени файла.

        Профилей у этих голосов нет — они собираются при первом обращении
        и складываются рядом, чтобы второй раз не считать.
        """
        from core import voicebank as legacy

        voices: dict[str, BankVoice] = {}
        try:
            for rec in legacy.scan():
                vid = Path(rec["path"]).stem
                voices[vid] = BankVoice(
                    id=vid,
                    display_name=rec["name"].replace("_", " ").strip().capitalize(),
                    gender=rec["gender"],
                    age_group="child" if rec.get("is_child") else "adult",
                    languages=["ru"],
                    source="user",
                    dir=self.dir / vid,
                    sample_path=Path(rec["path"]),
                    f0_hz=rec.get("f0_hz"),
                )
        except Exception:  # noqa: BLE001 — банк не должен ронять задачу
            log.exception("Не удалось прочитать старый банк voice_db/")
        return voices

    # ---------- доступ ----------

    def all(self) -> list[BankVoice]:
        return list(self.load().values())

    def get(self, voice_id: str) -> BankVoice | None:
        return self.load().get(voice_id)

    def by_gender(self, gender: str, child: bool | None = None) -> list[BankVoice]:
        out = [v for v in self.all() if gender in ("unknown", v.gender)]
        if child is not None:
            same_age = [v for v in out if v.is_child == child]
            if same_age:
                out = same_age
        return out

    def identities(self, voices: list[BankVoice], embedder=None) -> np.ndarray:
        """Матрица отпечатков. Недостающие считаются по сэмплу и кэшируются."""
        vecs = []
        for v in voices:
            if v.identity is None and embedder is not None and v.sample_path:
                try:
                    v.identity = embedder.embed_file(v.sample_path, 30.0)
                    self._store_identity(v)
                except Exception:  # noqa: BLE001
                    log.exception("Банк: не посчитать отпечаток %s", v.id)
            vecs.append(v.identity if v.identity is not None
                        else np.zeros(192, dtype=np.float32))
        return np.stack(vecs)

    def _store_identity(self, voice: BankVoice) -> None:
        if not voice.dir:
            return
        try:
            voice.dir.mkdir(parents=True, exist_ok=True)
            np.save(str(voice.dir / "identity.npy"), voice.identity)
        except OSError:
            pass

    # ---------- профили ----------

    def load_profile(self, voice_id: str):
        """Профиль пресета: готовый .pt либо сборка из сэмпла при первом вызове."""
        from voices.profiles import VoiceProfile, load_profile, save_profile

        if voice_id in self._profiles:
            return self._profiles[voice_id]

        voice = self.get(voice_id)
        if voice is None:
            raise KeyError(f"голос {voice_id} не найден в банке")

        profile_path = (voice.dir / "profile_xtts.pt") if voice.dir else None
        if profile_path and profile_path.exists():
            profile = load_profile(profile_path,
                                   voice.dir / "identity.npy", verify_ref=False)
        else:
            profile = self._build_from_sample(voice)

        profile.mode = "preset"
        profile.preset_id = voice_id
        profile.locked = True
        self._profiles[voice_id] = profile
        return profile

    def _build_from_sample(self, voice: BankVoice):
        """Голос из старой папки: считаем латенты один раз и сохраняем."""
        from core.config import load_config
        from identity.embeddings import get_embedder
        from synth.xtts_engine import get_engine
        from voices.profiles import build_profile, save_profile

        if not voice.sample_path or not Path(voice.sample_path).exists():
            raise FileNotFoundError(f"у голоса {voice.id} нет образца звука")

        cfg = load_config()
        log.info("Банк: собираю профиль голоса «%s» (первое обращение)…",
                 voice.display_name)
        profile = build_profile(voice.id, Path(voice.sample_path),
                                get_engine(cfg), get_embedder(cfg), mode="preset")
        profile.preset_id = voice.id
        profile.meta = {"display_name": voice.display_name,
                        "gender": voice.gender, "age_group": voice.age_group}
        if voice.dir:
            voice.dir.mkdir(parents=True, exist_ok=True)
            save_profile(profile, voice.dir)
            (voice.dir / "profile_xtts.pt").write_bytes(
                (voice.dir / "voice_profile.pt").read_bytes())
            if not (voice.dir / "meta.json").exists():
                (voice.dir / "meta.json").write_text(
                    json.dumps(voice.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8")
        return profile


_bank: VoiceBank | None = None


def get_bank() -> VoiceBank:
    global _bank
    if _bank is None:
        _bank = VoiceBank()
    return _bank
