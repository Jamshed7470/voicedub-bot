"""Тесты инвариантов синтеза: один профиль на спикера, Identity QC, отчёт.

Здесь проверяется главное обещание версии 2: тембр голоса вычисляется один
раз и больше никогда. Движок и эмбеддер подменены заглушками — нас
интересует не качество звука, а число вызовов и поведение при отказах.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from synth import qc as qc_mod
from synth.engine_base import TTSEngine, split_text
from synth.render import ProfileCache
from voices.profiles import VoiceProfile

ROOT = Path(__file__).resolve().parent.parent
DIM = 192
RNG = np.random.default_rng(7)


# ------------------------------------------------------------------ заглушки

class FakeEngine(TTSEngine):
    """Движок, который считает свои вызовы и пишет тишину нужной длины."""

    name = "fake"
    sample_rate = 24000

    def __init__(self, identity: np.ndarray | None = None):
        self.conditioning_calls = 0
        self.synth_calls = 0
        self.identity = identity

    def build_conditioning(self, ref_wav):
        self.conditioning_calls += 1
        return ("gpt_latent", "speaker_emb")

    def synthesize(self, text, lang, profile, out_path, speed=1.0, seed=None,
                   temperature=None):
        self.require_profile(profile)
        self.synth_calls += 1
        dur = max(0.3, len(text) / 15.0 / max(0.5, speed))
        sf.write(str(out_path), np.zeros(int(dur * self.sample_rate), dtype=np.float32),
                 self.sample_rate)
        return Path(out_path)


class FakeEmbedder:
    """Возвращает заранее заданный вектор на файл, иначе — постоянный."""

    def __init__(self, vectors: dict[str, np.ndarray] | None = None,
                 default: np.ndarray | None = None):
        self.vectors = vectors or {}
        self.default = default if default is not None else _unit(1)
        self.calls = 0

    def embed_file(self, path, max_seconds=60.0):
        self.calls += 1
        return self.vectors.get(Path(path).name, self.default)

    def embed_windows(self, y, spans, sr=16000):
        return np.stack([self.default for _ in spans])


class FakeCfg:
    def __init__(self, **over):
        self.device = "cpu"
        self.values = {
            "temperature": 0.55, "max_chars_per_call": 220,
            "identity_qc_min_clone": 0.70, "identity_qc_min_preset": 0.75,
            "duration_ratio_range": [0.5, 2.5], "asr_backcheck": False,
            "asr_backcheck_max_cer": 0.35, "max_retries": 3,
            "chunk_join_sec": 0.12,
            **over,
        }

    def y(self, *keys, default=None):
        if keys[0] == "synthesis" and len(keys) > 1:
            return self.values.get(keys[1], default)
        return default


def _unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def make_profile(sid="S1", mode="clone", identity=None) -> VoiceProfile:
    return VoiceProfile(speaker_id=sid, mode=mode, gpt_cond_latent="latent",
                        speaker_embedding="emb",
                        identity=identity if identity is not None else _unit(1),
                        locked=True)


# ------------------------------------------------- INV-2: нет синтеза без профиля

def test_synthesis_without_profile_is_impossible():
    """Незаблокированный или отсутствующий профиль → синтез не начнётся."""
    engine = FakeEngine()
    with pytest.raises(AssertionError):
        engine.synthesize("текст", "ru", None, "out.wav")

    unlocked = make_profile()
    unlocked.locked = False
    with pytest.raises(AssertionError):
        engine.synthesize("текст", "ru", unlocked, "out.wav")

    empty = make_profile()
    empty.gpt_cond_latent = None
    with pytest.raises(AssertionError):
        engine.synthesize("текст", "ru", empty, "out.wav")


def test_profile_built_once_per_speaker(tmp_path):
    """get_conditioning_latents вызывается ровно по разу на спикера (INV-1).

    Раньше тембр пересчитывался почти на каждой реплике — отсюда и брались
    «три-четыре голоса на одного человека».
    """
    from voices import profiles as prof_mod

    engine, embedder = FakeEngine(), FakeEmbedder()
    for sid in ("S1", "S2", "S3"):
        ref = tmp_path / f"{sid}.wav"
        sf.write(str(ref), np.zeros(24000, dtype=np.float32), 24000)
        profile = prof_mod.build_profile(sid, ref, engine, embedder)
        assert profile.locked

    assert engine.conditioning_calls == 3

    # ...и сколько бы реплик потом ни синтезировалось, счётчик не растёт
    profile = make_profile()
    for i in range(50):
        engine.synthesize(f"реплика {i}", "ru", profile, tmp_path / f"s{i}.wav")
    assert engine.conditioning_calls == 3, "тембр пересчитан внутри цикла синтеза"
    assert engine.synth_calls == 50


def test_profile_cache_loads_each_speaker_once(tmp_path, monkeypatch):
    """ProfileCache держит профиль в памяти: повторной загрузки нет."""
    from voices import profiles as prof_mod

    loaded = []

    def fake_load(path, identity_path=None, verify_ref=True):
        loaded.append(str(path))
        return make_profile("S1")

    monkeypatch.setattr(prof_mod, "load_profile", fake_load)

    p = tmp_path / "voice_profile.pt"
    p.write_bytes(b"x")
    speakers = {"S1": {"voice": {"mode": "clone", "profile_path": str(p)}}}
    cache = ProfileCache(speakers)

    for _ in range(20):
        cache.get("S1")
    assert cache.load_calls == 1
    assert len(loaded) == 1


# ------------------------------------------------- запрет старых кодовых путей

def test_no_per_segment_reference():
    """В рабочем коде не осталось расчёта тембра вне сборки профилей.

    Ищутся ВЫЗОВЫ, а не упоминания: документация и сообщения об ошибках
    обязаны называть удалённые функции по имени, чтобы человек понимал,
    что искать, — иначе тест запрещал бы объяснять собственный запрет.
    """
    allowed = {
        Path("voices/profiles.py"),          # сборка профиля спикера
        Path("voices/bank.py"),              # сборка банка голосов
        Path("synth/xtts_engine.py"),        # реализация build_conditioning
        Path("scripts/build_voice_bank.py"),
    }
    skip_dirs = {".venv", "__pycache__", "tests", "node_modules", ".git",
                 "studio"}

    offenders = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if set(rel.parts) & skip_dirs or rel in allowed:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"get_conditioning_latents\s*\(", text):
            offenders.append(str(rel))
        if re.search(r"choose_style_ref\s*\(", text):
            offenders.append(f"{rel} (choose_style_ref)")

    assert not offenders, ("тембр считается вне сборки профиля: "
                           + ", ".join(offenders))


# ------------------------------------------------------------------ Identity QC

def test_qc_passes_matching_voice(tmp_path):
    identity = _unit(5)
    profile = make_profile(identity=identity)
    wav = tmp_path / "seg.wav"
    sf.write(str(wav), np.zeros(24000, dtype=np.float32), 24000)

    embedder = FakeEmbedder(default=identity)
    result = qc_mod.check(wav, "привет", "ru", profile, 1.0, embedder, FakeCfg())
    assert result.ok and result.identity_sim > 0.99
    assert result.status == "ok"


def test_qc_flags_drifted_voice(tmp_path):
    """Синтез чужим тембром не проходит проверку — это и есть ловушка бага."""
    profile = make_profile(identity=_unit(5))
    wav = tmp_path / "seg.wav"
    sf.write(str(wav), np.zeros(24000, dtype=np.float32), 24000)

    embedder = FakeEmbedder(default=_unit(99))     # совсем другой голос
    result = qc_mod.check(wav, "привет", "ru", profile, 1.0, embedder, FakeCfg())
    assert not result.ok
    assert any("тембр" in r for r in result.reasons)


def test_qc_flags_wrong_duration(tmp_path):
    profile = make_profile(identity=_unit(5))
    wav = tmp_path / "seg.wav"
    sf.write(str(wav), np.zeros(24000 * 10, dtype=np.float32), 24000)  # 10 с

    embedder = FakeEmbedder(default=_unit(5))
    result = qc_mod.check(wav, "привет", "ru", profile, 1.0, embedder, FakeCfg())
    assert not result.ok
    assert any("длительность" in r for r in result.reasons)


def test_qc_retry_and_flag(tmp_path):
    """Плохой эмбеддер → 3 попытки, статус qc_failed, реплика всё же есть."""
    from synth.render import synthesize_segment

    profile = make_profile(identity=_unit(5))
    engine = FakeEngine()
    embedder = FakeEmbedder(default=_unit(99))
    seg = {"id": 1, "text": "тестовая реплика", "speaker": "S1"}

    result = synthesize_segment(seg, profile, engine, embedder, FakeCfg(),
                                "job1", "ru", tmp_path / "seg_1.wav",
                                speed=1.0, expected_dur=1.0)
    assert engine.synth_calls == 3, "не сделаны все попытки пересинтеза"
    assert not result.ok and result.status == "qc_failed"
    assert (tmp_path / "seg_1.wav").exists(), "реплика должна остаться в миксе"
    # временные файлы попыток убраны
    assert not list(tmp_path.glob("*_try*.wav"))


def test_qc_stops_early_when_passed(tmp_path):
    """Прошло с первой попытки — лишних прогонов модели нет."""
    from synth.render import synthesize_segment

    identity = _unit(5)
    profile = make_profile(identity=identity)
    engine = FakeEngine()
    embedder = FakeEmbedder(default=identity)
    seg = {"id": 2, "text": "короткая реплика", "speaker": "S1"}

    result = synthesize_segment(seg, profile, engine, embedder, FakeCfg(),
                                "job1", "ru", tmp_path / "seg_2.wav",
                                speed=1.0, expected_dur=1.2)
    assert engine.synth_calls == 1
    assert result.ok


def test_seed_is_deterministic():
    """Один и тот же вход даёт тот же seed, разные попытки — разный."""
    a = TTSEngine.make_seed("job", 12, 0)
    b = TTSEngine.make_seed("job", 12, 0)
    c = TTSEngine.make_seed("job", 12, 1)
    d = TTSEngine.make_seed("job", 13, 0)
    assert a == b and a != c and a != d


# ------------------------------------------------------------------ backcheck

def test_cer_basic():
    assert qc_mod.cer("привет мир", "привет мир") == 0.0
    assert qc_mod.cer("привет мир", "Привет, Мир!") == 0.0     # регистр и знаки
    assert qc_mod.cer("привет", "") == 1.0
    assert 0.0 < qc_mod.cer("привет мир", "привет миф") < 0.2


def test_backcheck_catches_gibberish():
    """«Каша» на выходе XTTS даёт высокий CER и уходит на пересинтез."""
    expected = "сегодня прекрасная погода и мы идём гулять"
    gibberish = "ща ща ща ща ща ща ща ща"
    assert qc_mod.cer(expected, gibberish) > 0.35


# ------------------------------------------------------------------ стабильность

def test_pairwise_identity_stable_profile():
    """30 реплик одного профиля → средняя попарная схожесть ≥ 0.75.

    Критерий приёмки Фазы 1: именно это число показывает, что голос
    больше не плывёт.
    """
    base = _unit(11)
    embs = []
    for _ in range(30):
        ortho = RNG.normal(size=DIM).astype(np.float32)
        ortho -= float(ortho @ base) * base
        ortho /= np.linalg.norm(ortho)
        cos_t = 0.93                     # разброс синтеза из одного профиля
        embs.append(cos_t * base + np.sqrt(1 - cos_t ** 2) * ortho)

    assert qc_mod.mean_pairwise_identity(embs) >= 0.75


def test_pairwise_identity_detects_drift():
    """Реплики, синтезированные разными тембрами, дают низкую схожесть."""
    embs = [_unit(i) for i in range(30)]
    assert qc_mod.mean_pairwise_identity(embs) < 0.5


def test_build_report_counts():
    per_speaker = {
        "S1": {"voice": "клон", "segments": 10, "passed": 9,
               "embeddings": [_unit(1), _unit(1)]},
        "S2": {"voice": "Мужской низкий 1", "segments": 4, "passed": 4,
               "embeddings": []},
    }
    report = qc_mod.build_report(per_speaker)
    assert report["S1"]["passed"] == 9
    assert report["S1"]["mean_pairwise_identity"] == pytest.approx(1.0, abs=1e-3)
    # у S2 образцов нет — устойчивость не измерена, и это не «идеально»
    import math
    assert math.isnan(report["S2"]["mean_pairwise_identity"])


# ------------------------------------------------------------------ разбиение

def test_split_text_respects_limit():
    text = "Слово. " * 100
    parts = split_text(text, 100)
    assert all(len(p) <= 100 for p in parts)
    assert "".join(p.replace(" ", "") for p in parts).startswith("Слово.Слово.")


def test_split_text_handles_single_long_word():
    parts = split_text("а" * 500, 100)
    assert all(len(p) <= 100 for p in parts)
    assert sum(len(p) for p in parts) == 500


def test_split_text_short_stays_whole():
    assert split_text("Короткая реплика.", 200) == ["Короткая реплика."]


def test_stability_reports_no_data_instead_of_perfect():
    """Меньше двух реплик — сравнивать нечего, и это не «идеально».

    Раньше возвращалась 1.0, и в ночном отчёте два ролика получили
    идеальную оценку там, где её никто не измерял: у эпизодического
    персонажа с одной репликой сравнивать не с чем. Ложная уверенность
    хуже пропуска — по такому отчёту нельзя понять, где смотреть.
    """
    import math

    v = _unit(3)
    assert math.isnan(qc_mod.mean_pairwise_identity([]))
    assert math.isnan(qc_mod.mean_pairwise_identity([v]))
    assert qc_mod.mean_pairwise_identity([v, v]) == pytest.approx(1.0)


def test_report_marks_unmeasured_speaker():
    """Спикер с одной репликой попадает в отчёт как неизмеренный."""
    import math

    report = qc_mod.build_report({
        "S1": {"voice": "клон", "segments": 1, "passed": 1,
               "embeddings": [_unit(1)]},
        "S2": {"voice": "клон", "segments": 9, "passed": 9,
               "embeddings": [_unit(2), _unit(2), _unit(2)]},
    })
    assert math.isnan(report["S1"]["mean_pairwise_identity"])
    assert not math.isnan(report["S2"]["mean_pairwise_identity"])
