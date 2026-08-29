"""Тесты кастинга: уникальность голосов, пол, кандидаты.

Главное требование спецификации — два спикера никогда не получают один
голос, пока голоса в банке есть. Жадный подбор «каждому ближайшего» это
требование нарушает, поэтому задача решается целиком.
"""
from __future__ import annotations

import numpy as np
import pytest

from voices import casting


DIM = 192


def unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeVoice:
    def __init__(self, vid, gender="male", child=False, langs=("ru",), f0=120.0):
        self.id = vid
        self.display_name = vid
        self.gender = gender
        self.languages = list(langs)
        self.f0_hz = f0
        self._child = child

    @property
    def is_child(self):
        return self._child


class FakeCfg:
    def __init__(self, **over):
        self.values = {"unique_voices": True, "language_bonus": 0.05,
                       "candidates_top_n": 3, **over}

    def y(self, *keys, default=None):
        if keys[0] == "casting" and len(keys) > 1:
            return self.values.get(keys[1], default)
        return default


class FakeBank:
    def __init__(self, voices, identities):
        self._voices = voices
        self._ident = identities

    def all(self):
        return self._voices

    def identities(self, voices, embedder=None):
        return np.stack([self._ident[v.id] for v in voices])


class FakeEmbedder:
    def embed_file(self, path, max_seconds=60.0):
        return unit(0)


# ------------------------------------------------------------------ тесты

def test_casting_unique():
    """Три спикера одного пола получают три РАЗНЫХ пресета."""
    # все три спикера ближе всего к одному и тому же голосу v1:
    # жадный подбор отдал бы его первому, а остальным — что попало
    target = unit(1)
    speakers = {
        f"S{i}": {"gender": "male", "centroid": target + 0.01 * unit(100 + i),
                  "speech_total_s": 100 - i,
                  "voice": {"mode": "preset", "preset_id": None,
                            "casting_candidates": []}}
        for i in range(1, 4)
    }
    voices = [FakeVoice("v1"), FakeVoice("v2"), FakeVoice("v3")]
    identities = {"v1": target, "v2": unit(2), "v3": unit(3)}

    casting.assign_voices(speakers, FakeCfg(), FakeBank(voices, identities),
                          FakeEmbedder(), lang="ru")

    assigned = [speakers[s]["voice"]["preset_id"] for s in speakers]
    assert len(set(assigned)) == 3, f"голоса повторились: {assigned}"
    assert None not in assigned


def test_casting_respects_gender():
    """Женскому спикеру не достаётся мужской голос, пока есть женские."""
    speakers = {
        "S1": {"gender": "female", "centroid": unit(5), "speech_total_s": 50,
               "voice": {"mode": "preset", "casting_candidates": []}},
        "S2": {"gender": "male", "centroid": unit(5), "speech_total_s": 40,
               "voice": {"mode": "preset", "casting_candidates": []}},
    }
    voices = [FakeVoice("m1", "male"), FakeVoice("f1", "female")]
    # оба голоса одинаково близки по тембру — решать должен пол
    identities = {"m1": unit(5), "f1": unit(5)}

    casting.assign_voices(speakers, FakeCfg(), FakeBank(voices, identities),
                          FakeEmbedder(), lang="ru")

    assert speakers["S1"]["voice"]["preset_id"] == "f1"
    assert speakers["S2"]["voice"]["preset_id"] == "m1"


def test_casting_keeps_top_candidates():
    """Для вкладки «Кастинг» сохраняются три лучших варианта с оценками."""
    speakers = {"S1": {"gender": "male", "centroid": unit(7),
                       "speech_total_s": 10,
                       "voice": {"mode": "preset", "casting_candidates": []}}}
    voices = [FakeVoice(f"v{i}") for i in range(5)]
    identities = {f"v{i}": unit(7 + i) for i in range(5)}

    casting.assign_voices(speakers, FakeCfg(), FakeBank(voices, identities),
                          FakeEmbedder(), lang="ru")

    cands = speakers["S1"]["voice"]["casting_candidates"]
    assert len(cands) == 3
    scores = [c["score"] for c in cands]
    assert scores == sorted(scores, reverse=True), "кандидаты не по убыванию"
    assert cands[0]["preset_id"] == speakers["S1"]["voice"]["preset_id"]


def test_casting_more_speakers_than_voices():
    """Голосов меньше, чем спикеров — работа не падает, повтор неизбежен."""
    speakers = {
        f"S{i}": {"gender": "male", "centroid": unit(i), "speech_total_s": 10,
                  "voice": {"mode": "preset", "casting_candidates": []}}
        for i in range(1, 5)
    }
    voices = [FakeVoice("v1"), FakeVoice("v2")]
    identities = {"v1": unit(1), "v2": unit(2)}

    casting.assign_voices(speakers, FakeCfg(), FakeBank(voices, identities),
                          FakeEmbedder(), lang="ru")

    assigned = [speakers[s]["voice"]["preset_id"] for s in speakers]
    assert all(a is not None for a in assigned)
    # пока голоса свободны, они не повторяются
    assert len(set(assigned)) == 2


def test_casting_skips_clone_speakers():
    """Спикер с клоном своего голоса кастингу не подлежит."""
    speakers = {
        "S1": {"gender": "male", "centroid": unit(1), "speech_total_s": 90,
               "voice": {"mode": "clone", "preset_id": None,
                         "casting_candidates": []}},
        "S2": {"gender": "male", "centroid": unit(2), "speech_total_s": 10,
               "voice": {"mode": "preset", "casting_candidates": []}},
    }
    voices = [FakeVoice("v1"), FakeVoice("v2")]
    identities = {"v1": unit(1), "v2": unit(2)}

    casting.assign_voices(speakers, FakeCfg(), FakeBank(voices, identities),
                          FakeEmbedder(), lang="ru")

    assert speakers["S1"]["voice"]["preset_id"] is None, "клон затронут кастингом"
    assert speakers["S2"]["voice"]["preset_id"] is not None


def test_casting_respects_user_choice():
    """Голос, выбранный человеком, автоматика не переназначает."""
    speakers = {"S1": {"gender": "male", "centroid": unit(1),
                       "speech_total_s": 10,
                       "voice": {"mode": "preset", "preset_id": "v2",
                                 "edited_by_user": True,
                                 "casting_candidates": []}}}
    voices = [FakeVoice("v1"), FakeVoice("v2")]
    identities = {"v1": unit(1), "v2": unit(9)}

    casting.assign_voices(speakers, FakeCfg(), FakeBank(voices, identities),
                          FakeEmbedder(), lang="ru")

    assert speakers["S1"]["voice"]["preset_id"] == "v2"


def test_casting_marks_narrator():
    """Спикер с наибольшим временем речи помечается как основной."""
    speakers = {
        "S1": {"gender": "male", "centroid": unit(1), "speech_total_s": 600,
               "voice": {"mode": "preset", "casting_candidates": []}},
        "S2": {"gender": "male", "centroid": unit(2), "speech_total_s": 60,
               "voice": {"mode": "preset", "casting_candidates": []}},
    }
    voices = [FakeVoice("v1"), FakeVoice("v2")]
    identities = {"v1": unit(1), "v2": unit(2)}

    casting.assign_voices(speakers, FakeCfg(), FakeBank(voices, identities),
                          FakeEmbedder(), lang="ru")
    assert speakers["S1"].get("role") == "narrator"
    assert speakers["S2"].get("role") != "narrator"


def test_empty_bank_returns_none():
    """Пустой банк — не падение, а честный None: рендер поймёт и объяснит."""
    speakers = {"S1": {"gender": "male", "centroid": unit(1),
                       "speech_total_s": 10,
                       "voice": {"mode": "preset", "casting_candidates": []}}}
    assert casting.assign_voices(speakers, FakeCfg(), FakeBank([], {}),
                                 FakeEmbedder()) is None


def test_language_bonus_applied():
    """При равном тембре предпочтение голосу, знающему язык задачи."""
    speakers = [{"gender": "male", "centroid": unit(4), "id": "S1"}]
    voices = [FakeVoice("no_ru", langs=("en",)), FakeVoice("with_ru", langs=("ru",))]
    identities = np.stack([unit(4), unit(4)])

    scores = casting.score_matrix(speakers, voices, identities, "ru", 0.05)
    assert scores[0, 1] > scores[0, 0]


def test_greedy_fallback_is_deterministic():
    """Запасной жадный путь даёт одинаковый ответ на одинаковых данных."""
    scores = np.array([[0.9, 0.8, 0.1], [0.85, 0.84, 0.2], [0.1, 0.2, 0.9]],
                      dtype=np.float32)
    first = casting._greedy_with_reuse(scores)
    second = casting._greedy_with_reuse(scores.copy())
    assert first == second
    assert len(set(first)) == 3


def test_bank_returned_when_all_voices_chosen_by_hand():
    """Все голоса выбраны человеком — банк всё равно нужен рендеру.

    Раньше при пустом списке назначений возвращался None, и озвучка
    падала с «банк недоступен» ровно в том случае, когда человек честно
    выбрал все голоса в студии: работы для автоматики не осталось, но
    загружать выбранное всё равно нужно.
    """
    speakers = {
        "S1": {"gender": "male", "centroid": unit(1), "speech_total_s": 10,
               "voice": {"mode": "preset", "preset_id": "v1",
                         "edited_by_user": True, "casting_candidates": []}},
    }
    voices = [FakeVoice("v1"), FakeVoice("v2")]
    bank = FakeBank(voices, {"v1": unit(1), "v2": unit(2)})

    result = casting.assign_voices(speakers, FakeCfg(), bank, FakeEmbedder())
    assert result is bank, "банк не отдан, рендер останется без голосов"
    assert speakers["S1"]["voice"]["preset_id"] == "v1", "выбор человека затёрт"


def test_no_bank_needed_for_pure_clones():
    """Одни клоны — банк не нужен и не запрашивается."""
    speakers = {
        "S1": {"gender": "male", "centroid": unit(1), "speech_total_s": 10,
               "voice": {"mode": "clone", "casting_candidates": []}},
    }
    assert casting.assign_voices(speakers, FakeCfg(), None, FakeEmbedder()) is None
