"""Темп речи: измеряется у голоса, а не берётся из таблицы.

Замер показал, что таблица завышена (14 симв/с против реальных 12.5), а
разброс между голосами больше разницы между языками (9.2–13.9). Бюджет
длины перевода, посчитанный по таблице, разрешает фразу, которая физически
не помещается в слот, — и её потом ускоряют до неразборчивости.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import soundfile as sf
import numpy as np

from voices import rate as rate_mod


class FakeProfile:
    def __init__(self, vid="v1"):
        self.preset_id = vid
        self.speaker_id = vid
        self.locked = True
        self.gpt_cond_latent = "x"
        self.mode = "preset"


class FakeCfg:
    def speech_rate(self, lang):
        return {"ru": 12.0, "en": 15.0}.get(lang, 12.0)


class FakeEngine:
    """Синтезирует тишину заданной длительности: важен только темп."""

    def __init__(self, chars_per_sec=10.0):
        self.rate = chars_per_sec
        self.calls = 0

    def synthesize(self, text, lang, profile, out_path, speed=1.0, seed=None):
        self.calls += 1
        seconds = len(text) / self.rate
        sf.write(str(out_path), np.zeros(int(seconds * 24000), dtype=np.float32),
                 24000)
        return Path(out_path)


@pytest.fixture(autouse=True)
def clear_cache():
    rate_mod._cache.clear()
    yield
    rate_mod._cache.clear()


def test_measures_actual_rate():
    """Темп берётся из синтеза, а не из таблицы."""
    engine = FakeEngine(chars_per_sec=10.0)
    got = rate_mod.measure(engine, FakeProfile(), "ru", FakeCfg())
    assert got == pytest.approx(10.0, rel=0.05)
    assert got != 12.0, "вернулось табличное значение вместо измеренного"


def test_result_is_cached():
    """Голосов в ролике единицы, обращений к темпу — по одному на реплику."""
    engine = FakeEngine()
    for _ in range(20):
        rate_mod.measure(engine, FakeProfile(), "ru", FakeCfg())
    assert engine.calls == 1


def test_different_voices_measured_separately():
    engine = FakeEngine(chars_per_sec=10.0)
    rate_mod.measure(engine, FakeProfile("slow"), "ru", FakeCfg())
    rate_mod.measure(engine, FakeProfile("fast"), "ru", FakeCfg())
    assert engine.calls == 2


def test_absurd_result_is_clamped():
    """Сбой синтеза не должен дать бюджет, оторванный от реальности."""
    insane = rate_mod.measure(FakeEngine(chars_per_sec=100.0), FakeProfile(),
                              "ru", FakeCfg())
    assert insane <= 12.0 * 1.5

    rate_mod._cache.clear()
    crawling = rate_mod.measure(FakeEngine(chars_per_sec=1.0), FakeProfile(),
                                "ru", FakeCfg())
    assert crawling >= 12.0 * 0.5


def test_falls_back_to_table_when_synthesis_fails():
    class Broken:
        def synthesize(self, *a, **kw):
            raise RuntimeError("нет видеопамяти")

    assert rate_mod.measure(Broken(), FakeProfile(), "ru", FakeCfg()) == 12.0


def test_unknown_language_uses_table():
    engine = FakeEngine()
    assert rate_mod.measure(engine, FakeProfile(), "xx", FakeCfg()) == 12.0
    assert engine.calls == 0, "нет калибровочной фразы — синтезировать нечего"


def test_measure_all_covers_every_speaker():
    class Cache:
        def get(self, sid):
            return FakeProfile(sid)

    rates = rate_mod.measure_all(FakeEngine(11.0), Cache(),
                                 {"S1": {}, "S2": {}}, "ru", FakeCfg())
    assert set(rates) == {"S1", "S2"}
    assert all(r == pytest.approx(11.0, rel=0.05) for r in rates.values())


def test_speaker_without_profile_is_skipped():
    class Cache:
        def get(self, sid):
            if sid == "S2":
                raise KeyError("нет профиля")
            return FakeProfile(sid)

    rates = rate_mod.measure_all(FakeEngine(), Cache(), {"S1": {}, "S2": {}},
                                 "ru", FakeCfg())
    assert set(rates) == {"S1"}, "спикер без профиля не должен ронять замер"
