"""Устаревший модуль синтеза. Оставлен как заглушка совместимости.

Логика переехала в пакет synth/: движок получает ЗАБЛОКИРОВАННЫЙ профиль
голоса, а не путь к референсу. Прежние функции этого модуля —
`choose_style_ref` (выбор референса на каждый сегмент) и `_conditioning`
(расчёт тембра внутри цикла синтеза) — были причиной того, что один человек
звучал несколькими голосами, и удалены намеренно.

Кто пришёл сюда за синтезом:
    from synth.xtts_engine import get_engine
    engine = get_engine(cfg)
    engine.synthesize(text, lang, profile, out_path, speed=..., seed=...)
"""
from __future__ import annotations

MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"


def synthesize(*args, **kwargs):
    raise RuntimeError(
        "core.tts.synthesize удалён: синтез идёт через synth.xtts_engine и "
        "требует заблокированный профиль голоса (INV-2). "
        "См. docs/AUDIT.md, раздел 3."
    )
