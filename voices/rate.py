"""Реальный темп речи голоса: измеряем, а не берём из таблицы.

Бюджет длины перевода считается как «сколько символов уместится в слот».
Умножать длительность на табличный темп языка — значит ошибаться дважды:

* таблица завышена. Замер на четырёх голосах банка: медиана 12.5 симв/с
  против 14 в конфиге;
* темп зависит от ГОЛОСА, а не только от языка. Разброс на тех же
  четырёх голосах — от 9.2 до 13.9 симв/с, то есть в полтора раза.

Завышенный бюджет означает, что переводчику разрешают длинную фразу,
которая потом не влезает, и её приходится ускорять до неразборчивости.
Одна калибровочная фраза на голос стоит секунду синтеза и снимает эту
ошибку целиком.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

# Нейтральные фразы средней длины: короткая даёт большую погрешность из-за
# пауз в начале и конце, длинная — лишнюю секунду ожидания на каждый голос.
CALIBRATION = {
    "ru": "Сегодня в городе тепло, и до вечера обещают ясную погоду.",
    "en": "The weather in the city is warm today and clear until evening.",
    "tr": "Bugün şehirde hava sıcak ve akşama kadar açık olacak.",
    "de": "Das Wetter in der Stadt ist heute warm und bis zum Abend klar.",
    "es": "Hoy hace calor en la ciudad y el cielo estará despejado.",
    "fr": "Il fait chaud en ville aujourd'hui et le ciel restera dégagé.",
    "it": "Oggi in città fa caldo e il cielo resterà sereno fino a sera.",
    "pt": "Hoje está calor na cidade e o céu ficará limpo até a noite.",
    "pl": "Dziś w mieście jest ciepło, a niebo pozostanie pogodne.",
    "nl": "Het is vandaag warm in de stad en de lucht blijft helder.",
    "cs": "Dnes je ve městě teplo a obloha zůstane jasná až do večera.",
    "hu": "Ma meleg van a városban, és az ég estig derült marad.",
}

_cache: dict[tuple[str, str], float] = {}


def measure(engine, profile, lang: str, cfg=None) -> float:
    """Символов в секунду у ЭТОГО голоса на ЭТОМ языке.

    Результат кэшируется на процесс: голосов в ролике единицы, а обращений
    к темпу — по одному на реплику.
    """
    import soundfile as sf

    key = (str(profile.preset_id or profile.speaker_id), lang)
    if key in _cache:
        return _cache[key]

    text = CALIBRATION.get(lang)
    if not text:
        return _table_rate(cfg, lang)

    tmp = Path(tempfile.gettempdir()) / f"rate_{abs(hash(key))}.wav"
    try:
        engine.synthesize(text, lang, profile, tmp, speed=1.0, seed=1)
        info = sf.info(str(tmp))
        seconds = info.frames / info.samplerate
        if seconds < 0.5:
            raise ValueError("слишком короткий результат")
        rate = len(text) / seconds
    except Exception:  # noqa: BLE001 — без замера работаем по таблице
        log.exception("Не удалось измерить темп голоса %s — беру табличный", key[0])
        rate = _table_rate(cfg, lang)
    finally:
        tmp.unlink(missing_ok=True)

    # защита от нелепого результата: голос, «говорящий» вдвое быстрее или
    # медленнее любого разумного, скорее означает сбой синтеза
    table = _table_rate(cfg, lang)
    rate = float(min(max(rate, table * 0.5), table * 1.5))
    _cache[key] = rate
    log.info("Темп голоса %s (%s): %.1f симв/с (в таблице %.1f)",
             key[0], lang, rate, table)
    return rate


def measure_all(engine, cache, speakers: dict, lang: str, cfg) -> dict[str, float]:
    """Темп для каждого спикера задачи: {speaker_id: символов в секунду}."""
    rates: dict[str, float] = {}
    for sid in speakers:
        try:
            profile = cache.get(sid)
        except Exception:  # noqa: BLE001 — у спикера может не быть профиля
            continue
        rates[sid] = measure(engine, profile, lang, cfg)
    return rates


def _table_rate(cfg, lang: str) -> float:
    if cfg is not None:
        return float(cfg.speech_rate(lang))
    return 12.0
