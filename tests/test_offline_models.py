"""Уже скачанная модель не должна зависеть от сети.

Локальный переводчик включается ровно тогда, когда облачные недоступны —
то есть при проблемах со связью. Если он в этот момент пойдёт на
HuggingFace за уже скачанной моделью, задача встанет в повторах на часы.
Ночная очередь так и остановилась молча.
"""
from __future__ import annotations

import sys
import types

import pytest


class Recorder:
    """Подменяет transformers и запоминает, с чем звали from_pretrained."""

    def __init__(self, local_available=True):
        self.local_available = local_available
        self.calls: list[dict] = []

    def _from_pretrained(self, name, **kw):
        self.calls.append(kw)
        if kw.get("local_files_only") and not self.local_available:
            raise OSError("модель не найдена локально")

        class Loaded:
            src_lang = None

            def to(self, device):
                return self
        return Loaded()

    def install(self, monkeypatch):
        mod = types.ModuleType("transformers")
        mod.AutoTokenizer = types.SimpleNamespace(
            from_pretrained=self._from_pretrained)
        mod.AutoModelForSeq2SeqLM = types.SimpleNamespace(
            from_pretrained=self._from_pretrained)
        monkeypatch.setitem(sys.modules, "transformers", mod)


class FakeCfg:
    device = "cpu"
    xai_api_key = ""
    nllb_model = "facebook/nllb-200-distilled-1.3B"

    def y(self, *a, **kw):
        return kw.get("default")


def test_cached_model_loads_without_network(monkeypatch):
    """При наличии локальной копии сеть не спрашивается вовсе."""
    from core.translate import NLLBTranslator

    rec = Recorder(local_available=True)
    rec.install(monkeypatch)

    t = NLLBTranslator(FakeCfg())
    t._load("eng_Latn")

    assert rec.calls, "модель не загружалась"
    assert all(c.get("local_files_only") for c in rec.calls), \
        "был поход в сеть за уже скачанной моделью"


def test_missing_model_falls_back_to_download(monkeypatch):
    """Копии нет — качаем, но только после честной попытки взять локально."""
    from core.translate import NLLBTranslator

    rec = Recorder(local_available=False)
    rec.install(monkeypatch)

    t = NLLBTranslator(FakeCfg())
    t._load("eng_Latn")

    assert rec.calls[0].get("local_files_only") is True, \
        "локальную копию даже не попробовали"
    assert any(not c.get("local_files_only") for c in rec.calls), \
        "загрузка не началась, хотя копии нет"


def test_second_load_reuses_model(monkeypatch):
    """Повторный вызов не трогает диск и сеть заново."""
    from core.translate import NLLBTranslator

    rec = Recorder(local_available=True)
    rec.install(monkeypatch)

    t = NLLBTranslator(FakeCfg())
    t._load("eng_Latn")
    before = len(rec.calls)
    t._load("tur_Latn")
    assert len(rec.calls) == before, "модель загрузилась повторно"
