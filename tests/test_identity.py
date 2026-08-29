"""Тесты Speaker Identity Engine.

Работают на синтетических эмбеддингах: логика слияния и переприсвоения не
зависит от того, чем получены векторы, а прогон ECAPA на каждом тесте
превратил бы набор в минуты ожидания и потребовал GPU. Тесты с реальными
моделями помечены gpu и живут отдельно.
"""
from __future__ import annotations

import numpy as np
import pytest

from identity import clustering
from identity.embeddings import centroid, cosine, l2

RNG = np.random.default_rng(20260829)
DIM = 192


# ------------------------------------------------------------------ фикстуры

def voice(seed: int) -> np.ndarray:
    """Устойчивый «голос» — направление в пространстве эмбеддингов."""
    rng = np.random.default_rng(seed)
    return l2(rng.normal(size=DIM).astype(np.float32))


def sample(base: np.ndarray, similarity: float = 0.75) -> np.ndarray:
    """Одна реплика этого голоса с ЗАДАННОЙ похожестью на его центр.

    Похожесть задаётся углом, а не величиной шума: в 192 измерениях
    гауссов шум со scale=0.3 имеет норму 4.2 против 1.0 у самого вектора и
    полностью его затирает — «реплики одного человека» получаются
    разными людьми (cos ≈ 0.27). Ориентир по реальному ECAPA: один
    диктор в разных условиях 0.6–0.85, разные дикторы 0.0–0.3.
    """
    ortho = RNG.normal(size=DIM).astype(np.float32)
    ortho -= float(ortho @ base) * base          # перпендикуляр к голосу
    ortho = l2(ortho)
    cos_t = float(np.clip(similarity + RNG.normal(scale=0.05), 0.05, 0.99))
    return l2(cos_t * base + np.sqrt(1 - cos_t ** 2) * ortho)


def make_segments(plan: list[tuple[str, float, float]]) -> list[dict]:
    """plan: [(label, start, end)] → сегменты в формате пайплайна."""
    return [
        {"id": i + 1, "start": s, "end": e, "text": f"реплика {i + 1}",
         "speaker": label, "flags": []}
        for i, (label, s, e) in enumerate(plan)
    ]


class FakeCfg:
    """Конфиг с дефолтами спецификации."""

    defaults = {
        "merge_threshold": 0.72,
        "overlap_merge_block_sec": 2.0,
        "min_assign_sim": 0.55,
        "reassign_margin": 0.08,
        "rounds": 2,
        "registry_match_threshold": 0.75,
        "isolated_max_sec": 1.5,
        "isolated_margin": 0.05,
        "overlap_ratio": 0.3,
    }

    def __init__(self, **over):
        self.values = {**self.defaults, **over}

    def y(self, *keys, default=None):
        if keys[0] == "speaker_identity" and len(keys) > 1:
            return self.values.get(keys[1], default)
        return default


# ------------------------------------------------------- один длинный спикер

def test_identity_single_speaker_long():
    """8 минут одного голоса, разбитые диаризацией на 3 кластера → 1 спикер.

    Это ровно тот случай, на который жалуется пользователь: pyannote дробит
    человека на части (сменилась эмоция, отошёл от микрофона), и каждая
    часть получает свой голос.
    """
    base = voice(1)
    plan, embs = [], []
    t = 0.0
    for k in range(60):
        # три «условия записи» одного и того же человека
        label = f"SPEAKER_{k % 3:02d}"
        plan.append((label, t, t + 6.0))
        embs.append(sample(base, similarity=0.72))
        t += 8.0

    segments = make_segments(plan)
    clusters = clustering.run(segments, np.stack(embs), FakeCfg())

    assert len(clusters) == 1, f"один человек разъехался на {len(clusters)} спикеров"
    assert len({s["speaker"] for s in segments}) == 1


def test_identity_two_speakers_dialog():
    """Диалог двух голосов: ровно 2 спикера, привязка сегментов ≥ 95%."""
    a, b = voice(2), voice(3)
    plan, embs, truth = [], [], []
    t = 0.0
    for i in range(40):
        who = i % 2
        plan.append((f"SPEAKER_{who:02d}", t, t + 2.5))
        embs.append(sample(a if who == 0 else b, similarity=0.75))
        truth.append(who)
        t += 3.0

    segments = make_segments(plan)
    clusters = clustering.run(segments, np.stack(embs), FakeCfg())
    assert len(clusters) == 2

    # точность: сегменты одного истинного голоса должны оказаться вместе
    groups: dict[int, list[str]] = {0: [], 1: []}
    for seg, who in zip(segments, truth):
        groups[who].append(seg["speaker"])
    accuracy = sum(
        max(g.count(x) for x in set(g)) for g in groups.values()
    ) / len(segments)
    assert accuracy >= 0.95, f"точность привязки {accuracy:.2%}"


def test_identity_respects_num_speakers_hint():
    """4 голоса: подсказка 4 → 4 спикера, подсказка 2 → сливаются ближайшие."""
    voices = [voice(10), voice(11), voice(12), voice(13)]
    plan, embs = [], []
    t = 0.0
    for i in range(48):
        who = i % 4
        plan.append((f"SPEAKER_{who:02d}", t, t + 2.0))
        embs.append(sample(voices[who], similarity=0.80))
        t += 2.5
    emb = np.stack(embs)

    four = clustering.run(make_segments(plan), emb.copy(), FakeCfg(),
                          num_speakers=4)
    assert len(four) == 4

    two = clustering.run(make_segments(plan), emb.copy(), FakeCfg(),
                         num_speakers=2)
    assert len(two) == 2


def test_identity_no_merge_when_overlapping():
    """Два похожих голоса, говорящих ОДНОВРЕМЕННО, не сливаются.

    Похожесть тембра — не доказательство, что это один человек. Перебивать
    сам себя человек не может, и это сильнее любого косинуса.
    """
    base = voice(20)
    # второй голос очень похож на первый — именно такие пары pyannote и
    # склонен путать, а слияние обязано их различить по одновременности
    similar = sample(base, similarity=0.92)

    plan, embs = [], []
    t = 0.0
    for i in range(20):
        # оба говорят на одном и том же интервале — сплошное наложение
        plan.append(("SPEAKER_00", t, t + 4.0))
        embs.append(sample(base, similarity=0.85))
        plan.append(("SPEAKER_01", t + 0.5, t + 4.5))
        embs.append(sample(similar, similarity=0.85))
        t += 6.0

    segments = make_segments(plan)
    emb = np.stack(embs)
    assert cosine(centroid(emb[0::2]), centroid(emb[1::2])) > 0.72, \
        "голоса должны быть похожи, иначе тест ничего не проверяет"

    clusters = clustering.run(segments, emb, FakeCfg())
    assert len(clusters) == 2, "одновременно говорящие слились в одного"


def test_low_confidence_segment_is_flagged_not_guessed():
    """Сегмент между двумя одинаково вероятными спикерами помечается флагом.

    Спецификация запрещает угадывать: такой сегмент идёт в студию.
    """
    a, b = voice(30), voice(31)
    plan, embs = [], []
    t = 0.0
    for i in range(20):
        who = i % 2
        plan.append((f"SPEAKER_{who:02d}", t, t + 2.0))
        embs.append(sample(a if who == 0 else b, similarity=0.80))
        t += 2.5
    # спорная реплика ровно между голосами
    plan.append(("SPEAKER_00", t, t + 2.0))
    embs.append(l2((a + b) / 2))

    segments = make_segments(plan)
    clustering.run(segments, np.stack(embs), FakeCfg())
    assert "low_speaker_conf" in segments[-1]["flags"]


def test_overlap_marked_from_raw_turns():
    """Интервал с двумя говорящими получает overlap и доминирующего спикера."""
    segments = make_segments([("SPEAKER_00", 0.0, 4.0), ("SPEAKER_00", 5.0, 8.0)])
    raw_turns = [
        {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 3.5, "speaker": "SPEAKER_01"},   # 2.5 с из 4 → 62%
        {"start": 5.0, "end": 8.0, "speaker": "SPEAKER_00"},
    ]
    marked = clustering.mark_overlaps(segments, raw_turns, ratio=0.3)
    assert marked == 1
    assert segments[0]["overlap"] is True
    assert "overlap" in segments[0]["flags"]
    assert segments[1].get("overlap") is not True


def test_overlap_seconds_two_pointer():
    """Пересечение интервалов считается корректно и без квадратичного перебора."""
    a = [(0.0, 2.0), (5.0, 9.0)]
    b = [(1.0, 3.0), (8.0, 12.0)]
    assert clustering.overlap_seconds(a, b) == pytest.approx(2.0)
    assert clustering.overlap_seconds([], b) == 0.0


# ------------------------------------------------------------------ реестр

def test_registry_keeps_ids_after_rediarize(tmp_path):
    """Повторная диаризация не меняет ID: правки пользователя выживают (INV-3)."""
    from identity import registry

    a, b = voice(40), voice(41)
    plan = [("SPEAKER_00", 0, 3), ("SPEAKER_01", 4, 7), ("SPEAKER_00", 8, 11)]
    embs = np.stack([sample(a, 0.85), sample(b, 0.85), sample(a, 0.85)])

    segments = make_segments(plan)
    clusters = clustering.run(segments, embs, FakeCfg())
    mapping = registry.assign_ids(clusters, segments)
    registry.save(tmp_path, clusters, mapping)
    first = {seg["id"]: seg["speaker"] for seg in segments}

    # повторный прогон: метки диаризации другие, порядок кластеров другой
    plan2 = [("SPEAKER_07", 0, 3), ("SPEAKER_03", 4, 7), ("SPEAKER_07", 8, 11)]
    segments2 = make_segments(plan2)
    clusters2 = clustering.run(segments2, embs.copy(), FakeCfg())
    previous = registry.load(tmp_path)
    assert previous, "реестр не сохранился"

    mapping2 = registry.assign_ids(clusters2, segments2, previous)
    second = {seg["id"]: seg["speaker"] for seg in segments2}
    assert second == first, f"ID разъехались: {first} → {second}"
    assert len(set(mapping2.values())) == len(mapping2), "ID достался двоим"


def test_registry_new_speaker_gets_free_id(tmp_path):
    """Появился новый человек — он получает свободный номер, чужой не отбирает."""
    from identity import registry

    a, b, c = voice(50), voice(51), voice(52)
    segs = make_segments([("SPEAKER_00", 0, 3), ("SPEAKER_01", 4, 7)])
    clusters = clustering.run(segs, np.stack([sample(a, .85), sample(b, .85)]),
                              FakeCfg())
    mapping = registry.assign_ids(clusters, segs)
    registry.save(tmp_path, clusters, mapping)

    segs2 = make_segments([("X", 0, 3), ("Y", 4, 7), ("Z", 8, 11)])
    clusters2 = clustering.run(
        segs2, np.stack([sample(a, .85), sample(b, .85), sample(c, .85)]), FakeCfg())
    mapping2 = registry.assign_ids(clusters2, segs2, registry.load(tmp_path))
    assert len(set(mapping2.values())) == 3
    assert {"S1", "S2"} <= set(mapping2.values())


# ------------------------------------------------------------------ качество

def test_score_candidate_prefers_long_clean_similar():
    from identity import quality

    good = quality.score_candidate(duration=10.0, snr=30.0, similarity=0.95)
    short = quality.score_candidate(duration=1.6, snr=30.0, similarity=0.95)
    noisy = quality.score_candidate(duration=10.0, snr=6.0, similarity=0.95)
    alien = quality.score_candidate(duration=10.0, snr=30.0, similarity=0.2)
    assert good > short and good > noisy and good > alien


def test_loudness_normalize_no_clipping():
    from identity import quality

    y = (RNG.normal(scale=0.01, size=48000)).astype(np.float32)
    out = quality.loudness_normalize(y, 24000, -23.0)
    assert np.max(np.abs(out)) <= 0.995
    assert np.sqrt(np.mean(out ** 2)) > np.sqrt(np.mean(y ** 2))
