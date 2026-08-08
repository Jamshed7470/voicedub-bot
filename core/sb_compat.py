"""Совместимость со speechbrain 1.1+ на Windows.

speechbrain объявляет опциональные интеграции (k2_fsa, nlp/flair/spacy и др.)
ленивыми модулями. Библиотеки, сканирующие модули (whisperx/pyannote/TTS,
lightning), трогают эти атрибуты, ленивый импорт пытается загрузить
неустановленный опциональный пакет и роняет ВЕСЬ процесс ImportError'ом.

Патч: LazyModule.ensure_module при ImportError возвращает пустую заглушку
вместо исключения. На функциональность не влияет — опциональные интеграции
в проекте не используются.
"""
from __future__ import annotations

import logging
import types

log = logging.getLogger(__name__)


def patch_speechbrain_lazy_imports() -> None:
    """Вызывать ДО (или сразу после) первого импорта speechbrain."""
    try:
        from speechbrain.utils import importutils as sb_imp
    except Exception:  # speechbrain не установлен — патчить нечего
        return
    if getattr(sb_imp, "_voicedub_patched", False):
        return

    orig = sb_imp.LazyModule.ensure_module

    def safe_ensure_module(self, stacklevel: int = 1):
        try:
            return orig(self, stacklevel + 1)
        except ImportError:
            log.debug("Опциональный модуль speechbrain недоступен: %r", self)
            return types.ModuleType("speechbrain_optional_unavailable")

    sb_imp.LazyModule.ensure_module = safe_ensure_module
    sb_imp._voicedub_patched = True
    log.debug("speechbrain: ленивые импорты сделаны безопасными")
