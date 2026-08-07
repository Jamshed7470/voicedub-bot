"""Тесты нормализации текста для TTS (обязательные кейсы из спецификации)."""
import re

import pytest

from core.normalize import assert_no_digits, normalize_for_tts


def test_date_9_may():
    assert normalize_for_tts("9 мая", "ru") == "девятого мая"


def test_year_2026_prepositional():
    out = normalize_for_tts("в 2026 году", "ru")
    assert out == "в две тысячи двадцать шестом году"


def test_percent_5():
    assert normalize_for_tts("5%", "ru") == "пять процентов"


def test_somoni_100():
    assert normalize_for_tts("100 сомони", "ru") == "сто сомони"


def test_km_10():
    assert normalize_for_tts("10 км", "ru") == "десять километров"


def test_percent_forms():
    assert normalize_for_tts("1%", "ru") == "один процент"
    assert normalize_for_tts("2%", "ru") == "два процента"
    assert normalize_for_tts("11%", "ru") == "одиннадцать процентов"
    assert normalize_for_tts("21%", "ru") == "двадцать один процент"


def test_sentence_with_date_and_year():
    out = normalize_for_tts("Встретимся 9 мая в 2026 году", "ru")
    assert "девятого мая" in out
    assert "в две тысячи двадцать шестом году" in out
    assert not re.search(r"\d", out)


def test_currency_and_number_sign():
    out = normalize_for_tts("Это стоило $250, дом №4", "ru")
    assert "долларов" in out
    assert "номер четыре" in out
    assert not re.search(r"[0-9%№$€]", out)


def test_no_digits_mixed():
    out = normalize_for_tts(
        "Скидка 50% до 17.05.2026, цена 1999 € за 2 кг и 10 км доставки №7",
        "ru",
    )
    assert not re.search(r"[0-9%№$€]", out)


def test_no_digits_english():
    out = normalize_for_tts("It costs $25 and 10 km, 5% off in 2026", "en")
    assert not re.search(r"[0-9%№$€]", out)


def test_no_change_without_digits():
    text = "Привет, как дела?"
    assert normalize_for_tts(text, "ru") == text


def test_assert_no_digits_raises():
    with pytest.raises(AssertionError):
        assert_no_digits("осталось 5 минут")
    assert_no_digits("осталось пять минут")  # не должно бросить
