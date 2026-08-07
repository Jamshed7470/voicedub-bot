"""Нормализация текста перед TTS: все числа — словами.

Гарантия модуля: в тексте, который уходит в TTS, нет ни одной цифры
и символов %, №, $, €.
"""
from __future__ import annotations

import logging
import re

from num2words import num2words

log = logging.getLogger(__name__)

DIGIT_RE = re.compile(r"[0-9]")
FORBIDDEN_RE = re.compile(r"[0-9%№$€]")

# ---------------------------------------------------------------------------
# Русский язык — собственные правила дат, порядковых, единиц и валют
# ---------------------------------------------------------------------------

MONTHS_GEN = {
    1: "января", 2: "февраля", 3: "марта", 4: "апреля", 5: "мая", 6: "июня",
    7: "июля", 8: "августа", 9: "сентября", 10: "октября", 11: "ноября", 12: "декабря",
}
MONTH_NAME_TO_NUM = {v: k for k, v in MONTHS_GEN.items()}
MONTHS_ALT = "|".join(MONTHS_GEN.values())


def _plural_ru(n: int, one: str, few: str, many: str) -> str:
    n = abs(int(n)) % 100
    if 11 <= n <= 14:
        return many
    d = n % 10
    if d == 1:
        return one
    if 2 <= d <= 4:
        return few
    return many


# единицы измерения: сокращение → (одна, две-четыре, пять)
UNITS_RU = {
    "км": ("километр", "километра", "километров"),
    "мм": ("миллиметр", "миллиметра", "миллиметров"),
    "см": ("сантиметр", "сантиметра", "сантиметров"),
    "мл": ("миллилитр", "миллилитра", "миллилитров"),
    "кг": ("килограмм", "килограмма", "килограммов"),
    "м": ("метр", "метра", "метров"),
    "г": ("грамм", "грамма", "граммов"),
    "т": ("тонна", "тонны", "тонн"),
    "л": ("литр", "литра", "литров"),
    "ч": ("час", "часа", "часов"),
    "мин": ("минута", "минуты", "минут"),
    "сек": ("секунда", "секунды", "секунд"),
}
# от длинных сокращений к коротким, чтобы «мин» не съелось как «м»
UNITS_RU_ALT = "|".join(sorted(UNITS_RU, key=len, reverse=True))

CURRENCY_RU = {
    "$": ("доллар", "доллара", "долларов"),
    "€": ("евро", "евро", "евро"),
}


def _card_ru(n) -> str:
    """Количественное числительное по-русски."""
    try:
        if isinstance(n, str):
            n = n.replace(",", ".")
            n = float(n) if "." in n else int(n)
        return num2words(n, lang="ru")
    except Exception:
        # последний резерв — цифра за цифрой
        return _digits_spelled(str(n), "ru")


def _ord_ru(n: int) -> str:
    """Порядковое числительное (именительный падеж): 2026 → «две тысячи двадцать шестой»."""
    return num2words(int(n), to="ordinal", lang="ru")


def _inflect_ordinal_ru(phrase: str, case: str) -> str:
    """Склоняет ПОСЛЕДНЕЕ слово порядкового числительного.

    case: "gen" (родительный: девятый→девятого), "prep" (предложный: шестой→шестом).
    """
    words = phrase.split()
    last = words[-1]
    if case == "gen":
        if last == "третий":
            last = "третьего"
        elif last.endswith(("ый", "ой")):
            last = last[:-2] + "ого"
        elif last.endswith("ий"):
            last = last[:-2] + "его"
    elif case == "prep":
        if last == "третий":
            last = "третьем"
        elif last.endswith(("ый", "ой")):
            last = last[:-2] + "ом"
        elif last.endswith("ий"):
            last = last[:-2] + "ем"
    words[-1] = last
    return " ".join(words)


def _digits_spelled(s: str, lang: str) -> str:
    """Резерв: каждая цифра отдельным словом."""
    out = []
    for ch in s:
        if ch.isdigit():
            try:
                out.append(num2words(int(ch), lang=lang))
            except Exception:
                out.append(num2words(int(ch), lang="en"))
        else:
            out.append(ch)
    return "".join(out)


def _normalize_ru(text: str) -> str:
    # --- даты вида 17.05.2026 / 17.05 ---
    def date_numeric(m: re.Match) -> str:
        day, month = int(m.group(1)), int(m.group(2))
        year = m.group(3) if (m.lastindex or 0) >= 3 else None
        if not (1 <= day <= 31 and 1 <= month <= 12):
            return m.group(0)
        s = f"{_inflect_ordinal_ru(_ord_ru(day), 'gen')} {MONTHS_GEN[month]}"
        if year:
            s += f" {_inflect_ordinal_ru(_ord_ru(int(year)), 'gen')} года"
        return s

    text = re.sub(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", date_numeric, text)
    text = re.sub(r"\b(\d{1,2})\.(\d{1,2})\b(?!\.)",
                  lambda m: date_numeric(m) if 1 <= int(m.group(2)) <= 12 else m.group(0),
                  text)

    # --- даты вида «9 мая» ---
    text = re.sub(
        rf"\b(\d{{1,2}})\s+({MONTHS_ALT})\b",
        lambda m: f"{_inflect_ordinal_ru(_ord_ru(int(m.group(1))), 'gen')} {m.group(2)}",
        text,
    )

    # --- годы: «в 2026 году», «2026 года», «2026 год» ---
    text = re.sub(
        r"\b[вВ]\s+(\d{3,4})\s+году\b",
        lambda m: f"в {_inflect_ordinal_ru(_ord_ru(int(m.group(1))), 'prep')} году",
        text,
    )
    text = re.sub(
        r"\b(\d{3,4})\s+года\b",
        lambda m: f"{_inflect_ordinal_ru(_ord_ru(int(m.group(1))), 'gen')} года",
        text,
    )
    text = re.sub(
        r"\b(\d{3,4})\s+год\b",
        lambda m: f"{_ord_ru(int(m.group(1)))} год",
        text,
    )

    # --- проценты ---
    def percent(m: re.Match) -> str:
        raw = m.group(1).replace(",", ".")
        if "." in raw:
            return f"{_card_ru(raw)} процента"
        n = int(raw)
        return f"{_card_ru(n)} {_plural_ru(n, 'процент', 'процента', 'процентов')}"

    text = re.sub(r"(\d+(?:[.,]\d+)?)\s*%", percent, text)

    # --- валюты: $5 / 5$ / €10 ---
    def currency(m: re.Match) -> str:
        sym = m.group("sym")
        raw = m.group("num").replace(",", ".")
        one, few, many = CURRENCY_RU[sym]
        if "." in raw:
            return f"{_card_ru(raw)} {few}"
        n = int(raw)
        return f"{_card_ru(n)} {_plural_ru(n, one, few, many)}"

    text = re.sub(r"(?P<sym>[$€])\s*(?P<num>\d+(?:[.,]\d+)?)", currency, text)
    text = re.sub(r"(?P<num>\d+(?:[.,]\d+)?)\s*(?P<sym>[$€])", currency, text)

    # --- № ---
    text = re.sub(r"№\s*(\d+)", lambda m: f"номер {_card_ru(int(m.group(1)))}", text)
    text = text.replace("№", "номер ")

    # --- единицы измерения: «10 км» ---
    def unit(m: re.Match) -> str:
        n = int(m.group(1))
        one, few, many = UNITS_RU[m.group(2)]
        return f"{_card_ru(n)} {_plural_ru(n, one, few, many)}"

    text = re.sub(rf"\b(\d+)\s*({UNITS_RU_ALT})\b\.?", unit, text)

    # --- порядковые с наращением: «5-й», «3-го», «10-му», «2-х» ---
    def ordinal_suffix(m: re.Match) -> str:
        n = int(m.group(1))
        suf = m.group(2)
        base = _ord_ru(n)
        if suf in ("го", "х"):
            return _inflect_ordinal_ru(base, "gen")
        if suf in ("м",):
            return _inflect_ordinal_ru(base, "prep")
        if suf == "я":  # пятая
            w = base.split()
            if w[-1].endswith(("ый", "ой")):
                w[-1] = w[-1][:-2] + "ая"
            elif w[-1] == "третий":
                w[-1] = "третья"
            return " ".join(w)
        if suf == "е":  # пятое
            w = base.split()
            if w[-1].endswith(("ый", "ой")):
                w[-1] = w[-1][:-2] + "ое"
            elif w[-1] == "третий":
                w[-1] = "третье"
            return " ".join(w)
        return base

    text = re.sub(r"\b(\d+)-(й|я|е|го|му|м|х)\b", ordinal_suffix, text)

    # --- диапазоны «5-10» ---
    text = re.sub(
        r"\b(\d+)\s*[-–—]\s*(\d+)\b",
        lambda m: f"{_card_ru(int(m.group(1)))} — {_card_ru(int(m.group(2)))}",
        text,
    )

    # --- остальные числа (включая десятичные) ---
    text = re.sub(r"\d+(?:[.,]\d+)?", lambda m: _card_ru(m.group(0)), text)

    return re.sub(r"\s{2,}", " ", text).strip()


# ---------------------------------------------------------------------------
# Прочие языки — num2words + карта слов для символов
# ---------------------------------------------------------------------------

# соответствие кодов языков бота кодам num2words
NUM2WORDS_LANG = {
    "ru": "ru", "en": "en", "tr": "tr", "ar": "ar", "es": "es", "fr": "fr",
    "de": "de", "it": "it", "pt": "pt", "pl": "pl", "nl": "nl", "cs": "cz",
    "hu": "hu", "ja": "ja", "ko": "ko", "hi": "hi", "zh-cn": "en",
}

PERCENT_WORD = {
    "en": "percent", "de": "Prozent", "fr": "pour cent", "es": "por ciento",
    "it": "per cento", "pt": "por cento", "pl": "procent", "nl": "procent",
    "cs": "procent", "hu": "százalék", "tr": "yüzde", "ar": "بالمئة",
    "ja": "パーセント", "ko": "퍼센트", "hi": "प्रतिशत", "zh-cn": "percent",
}
CURRENCY_WORD = {
    "$": {"en": "dollars", "de": "Dollar", "fr": "dollars", "es": "dólares",
          "it": "dollari", "pt": "dólares", "pl": "dolarów", "default": "dollars"},
    "€": {"en": "euros", "de": "Euro", "fr": "euros", "es": "euros",
          "it": "euro", "pt": "euros", "pl": "euro", "default": "euros"},
}
NUMBER_WORD = {"en": "number", "de": "Nummer", "fr": "numéro", "es": "número",
               "it": "numero", "pt": "número", "pl": "numer", "default": "number"}


def _card_generic(raw: str, lang: str) -> str:
    n2w = NUM2WORDS_LANG.get(lang, "en")
    raw = raw.replace(",", ".")
    try:
        val = float(raw) if "." in raw else int(raw)
        return num2words(val, lang=n2w)
    except Exception:
        try:
            return _digits_spelled(raw, n2w)
        except Exception:
            return _digits_spelled(raw, "en")


def _normalize_generic(text: str, lang: str) -> str:
    pct = PERCENT_WORD.get(lang, "percent")
    text = re.sub(r"(\d+(?:[.,]\d+)?)\s*%",
                  lambda m: f"{_card_generic(m.group(1), lang)} {pct}", text)
    for sym, words in CURRENCY_WORD.items():
        w = words.get(lang, words["default"])
        text = re.sub(rf"\{sym}\s*(\d+(?:[.,]\d+)?)",
                      lambda m, w=w: f"{_card_generic(m.group(1), lang)} {w}", text)
        text = re.sub(rf"(\d+(?:[.,]\d+)?)\s*\{sym}",
                      lambda m, w=w: f"{_card_generic(m.group(1), lang)} {w}", text)
    nw = NUMBER_WORD.get(lang, NUMBER_WORD["default"])
    text = re.sub(r"№\s*(\d+)", lambda m: f"{nw} {_card_generic(m.group(1), lang)}", text)
    text = text.replace("№", f"{nw} ")
    text = re.sub(r"\d+(?:[.,]\d+)?", lambda m: _card_generic(m.group(0), lang), text)
    return re.sub(r"\s{2,}", " ", text).strip()


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------

def normalize_for_tts(text: str, lang: str) -> str:
    """Приводит текст к виду, пригодному для TTS: без цифр и символов %, №, $, €."""
    if not FORBIDDEN_RE.search(text):
        return text

    if lang == "ru":
        text = _normalize_ru(text)
    else:
        text = _normalize_generic(text, lang)

    # последний резерв: если цифры всё ещё остались — цифра за цифрой
    if DIGIT_RE.search(text):
        log.warning("После нормализации остались цифры, применяю резерв: %r", text)
        text = _digits_spelled(text, NUM2WORDS_LANG.get(lang, "en"))

    # убрать оставшиеся запрещённые символы
    text = text.replace("%", " процентов " if lang == "ru" else " percent ")
    text = text.replace("$", " ").replace("€", " ").replace("№", " ")
    text = re.sub(r"\s{2,}", " ", text).strip()

    assert not DIGIT_RE.search(text), f"В тексте для TTS остались цифры: {text!r}"
    return text


def assert_no_digits(text: str) -> None:
    """Финальная проверка перед синтезом."""
    assert not DIGIT_RE.search(text), f"В тексте для TTS остались цифры: {text!r}"
