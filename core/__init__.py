"""Пакет core.

Здесь же — обязательная заглушка модуля k2 для Windows: пакет k2 (k2-fsa)
на Windows недоступен, а ленивый импорт speechbrain.integrations.k2_fsa
падает при сканировании модулей (его провоцируют whisperx/pyannote/TTS)
и роняет весь процесс. Сам k2 в проекте не используется — заглушка делает
`import k2` успешным и безвредным.
"""
import sys
import types

try:  # если настоящий k2 вдруг установлен — используем его
    import k2  # noqa: F401
except ImportError:
    _stub = types.ModuleType("k2")
    _stub.__version__ = "0.0.0-stub"
    _stub.__doc__ = "Заглушка k2 для Windows (см. core/__init__.py VoiceDub Bot)"
    sys.modules["k2"] = _stub
