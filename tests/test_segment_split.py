"""Разрезание реплик по смене говорящего.

WhisperX ставит спикера на каждое слово, но выдаёт их блоками по несколько
фраз. Раньше блок целиком получал одного спикера: на фильме это давало
реплики по двадцать пять секунд, внутри которых говорят трое. Озвучить
такое одним голосом нельзя.
"""
from __future__ import annotations

from core.diarize import split_by_word_speakers


def words(*spec):
    """spec: (слово, начало, конец, спикер)"""
    return [{"word": w, "start": a, "end": b, "speaker": s}
            for w, a, b, s in spec]


def test_splits_where_speaker_changes():
    seg = {
        "start": 0.0, "end": 6.0, "speaker": "SPEAKER_00",
        "text": "Привет как дела Хорошо спасибо",
        "words": words(
            ("Привет", 0.0, 0.6, "SPEAKER_00"),
            ("как", 0.7, 1.0, "SPEAKER_00"),
            ("дела", 1.1, 1.6, "SPEAKER_00"),
            ("Хорошо", 3.0, 3.7, "SPEAKER_01"),
            ("спасибо", 3.8, 4.5, "SPEAKER_01"),
        ),
    }
    parts = split_by_word_speakers(seg, max_dur=12.0)

    assert len(parts) == 2, "реплика двух людей осталась одной"
    assert parts[0]["speaker"] == "SPEAKER_00"
    assert parts[0]["text"] == "Привет как дела"
    assert parts[0]["start"] == 0.0 and parts[0]["end"] == 1.6
    assert parts[1]["speaker"] == "SPEAKER_01"
    assert parts[1]["text"] == "Хорошо спасибо"
    assert parts[1]["start"] == 3.0


def test_three_speakers_in_one_block():
    """Ровно случай из жалобы: в одном блоке говорят трое."""
    seg = {
        "start": 0.0, "end": 9.0, "speaker": "SPEAKER_00", "text": "…",
        "words": words(
            ("Раз", 0.0, 0.5, "SPEAKER_00"),
            ("два", 0.6, 1.0, "SPEAKER_00"),
            ("три", 2.0, 2.5, "SPEAKER_01"),
            ("четыре", 4.0, 4.6, "SPEAKER_02"),
            ("пять", 4.7, 5.2, "SPEAKER_02"),
        ),
    }
    parts = split_by_word_speakers(seg, max_dur=12.0)
    assert [p["speaker"] for p in parts] == ["SPEAKER_00", "SPEAKER_01",
                                             "SPEAKER_02"]


def test_single_speaker_stays_whole():
    seg = {
        "start": 0.0, "end": 3.0, "speaker": "SPEAKER_00", "text": "Раз два три",
        "words": words(("Раз", 0.0, 0.5, "SPEAKER_00"),
                       ("два", 0.6, 1.0, "SPEAKER_00"),
                       ("три", 1.1, 1.6, "SPEAKER_00")),
    }
    parts = split_by_word_speakers(seg, max_dur=12.0)
    assert len(parts) == 1
    assert parts[0]["text"] == "Раз два три"


def test_long_monologue_is_cut_at_biggest_pauses():
    """Двадцать секунд одного голоса тоже дробятся: в слот такое не влезет."""
    spec = []
    t = 0.0
    for i in range(20):
        # каждое пятое слово с большой паузой перед ним — граница фразы
        if i and i % 5 == 0:
            t += 1.5
        spec.append((f"слово{i}", t, t + 0.8, "SPEAKER_00"))
        t += 0.9
    parts = split_by_word_speakers(
        {"start": 0.0, "end": t, "speaker": "SPEAKER_00", "text": "x",
         "words": words(*spec)}, max_dur=8.0)

    assert len(parts) > 1, "длинный монолог не разрезан"
    assert all(p["end"] - p["start"] <= 12.0 for p in parts)
    # текст не потерян
    joined = " ".join(p["text"] for p in parts).split()
    assert len(joined) == 20


def test_segment_without_words_survives():
    """Без пословных таймингов реплика остаётся как была, а не пропадает."""
    seg = {"start": 1.0, "end": 4.0, "speaker": "SPEAKER_00",
           "text": "Без разметки слов"}
    parts = split_by_word_speakers(seg, max_dur=12.0)
    assert len(parts) == 1
    assert parts[0]["text"] == "Без разметки слов"
    assert parts[0]["start"] == 1.0 and parts[0]["end"] == 4.0


def test_words_without_timings_do_not_collapse():
    """У чисел и знаков whisperx иногда нет таймингов — кусок не должен
    получить нулевую длину и выпасть из микса."""
    seg = {
        "start": 0.0, "end": 3.0, "speaker": "SPEAKER_00", "text": "x",
        "words": [
            {"word": "Было", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
            {"word": "25", "speaker": "SPEAKER_00"},          # без таймингов
            {"word": "человек", "start": 1.2, "end": 1.9, "speaker": "SPEAKER_00"},
        ],
    }
    parts = split_by_word_speakers(seg, max_dur=12.0)
    assert len(parts) == 1
    assert parts[0]["end"] > parts[0]["start"]
    assert "25" in parts[0]["text"]


def test_word_without_speaker_joins_current():
    """Слово без метки не должно порождать лишнюю реплику."""
    seg = {
        "start": 0.0, "end": 3.0, "speaker": "SPEAKER_00", "text": "x",
        "words": [
            {"word": "Раз", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
            {"word": "два", "start": 0.6, "end": 1.0},        # без спикера
            {"word": "три", "start": 1.1, "end": 1.5, "speaker": "SPEAKER_00"},
        ],
    }
    parts = split_by_word_speakers(seg, max_dur=12.0)
    assert len(parts) == 1, "слово без метки разорвало реплику"
    assert parts[0]["text"] == "Раз два три"


def test_empty_pieces_are_dropped():
    seg = {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "",
           "words": [{"word": "  ", "start": 0.0, "end": 0.3,
                      "speaker": "SPEAKER_00"}]}
    assert split_by_word_speakers(seg, max_dur=12.0) == []
