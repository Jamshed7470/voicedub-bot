"""Регрессия на фикстурах с известным ответом.

Здесь и только здесь можно честно измерить качество: материал синтезирован
известными голосами, поэтому видно не «сколько спикеров нашлось», а
сколько раз алгоритм ошибся.

Тесты требуют реальных моделей и заметного времени, поэтому помечены `gpu`
и пропускаются, если фикстуры не собраны:

    python -m scripts.make_fixtures
    pytest -m gpu
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"

pytestmark = pytest.mark.gpu


def load_fixture(name: str) -> dict:
    truth = FIXTURES / name / "truth.json"
    audio = FIXTURES / name / "audio16.wav"
    if not truth.exists() or not audio.exists():
        pytest.skip(f"фикстура «{name}» не собрана: python -m scripts.make_fixtures")
    data = json.loads(truth.read_text(encoding="utf-8"))
    data["audio"] = audio
    return data


def run_sie(fixture: dict, hint: str = "auto") -> tuple[list[dict], dict]:
    """Прогоняет Speaker Identity Engine по фикстуре.

    Диаризация pyannote намеренно НЕ вызывается: на вход подаются идеальные
    границы реплик, а каждая реплика объявляется отдельным «кластером».
    Это худший из возможных входов — полное дробление, — и именно его
    движок обязан свести обратно.
    """
    import identity as sie
    from core.config import load_config

    cfg = load_config()
    segments = [
        {"id": s["id"], "start": s["start"], "end": s["end"],
         "text": s["text"], "speaker": f"RAW{s['id']:03d}", "flags": []}
        for s in fixture["segments"]
    ]
    result = sie.analyze(fixture["audio"], segments, cfg, raw_turns=None,
                         job_dir=None, speakers_hint=hint)
    return segments, result


def accuracy(segments: list[dict], fixture: dict) -> float:
    """Доля реплик, попавших в «правильную» группу.

    Метка алгоритма (S1, S2…) не обязана совпадать с истинной (TRUE0…) —
    важно, что реплики одного человека оказались вместе. Поэтому каждой
    истинной группе сопоставляется её самая частая метка.
    """
    truth = {s["id"]: s["speaker"] for s in fixture["segments"]}
    groups: dict[str, Counter] = {}
    for seg in segments:
        groups.setdefault(truth[seg["id"]], Counter())[seg["speaker"]] += 1

    used: set[str] = set()
    correct = 0
    # группы разбираются по убыванию размера: крупная роль получает свою
    # метку первой и не отдаёт её эпизодической
    for true_id, counter in sorted(groups.items(), key=lambda kv: -sum(kv[1].values())):
        for label, count in counter.most_common():
            if label not in used:
                used.add(label)
                correct += count
                break
    return correct / max(1, len(segments))


# ------------------------------------------------------------------ тесты

def test_fixture_monologue_stays_one_speaker():
    """8 минут одного голоса, поданные как 60 отдельных кластеров → 1 спикер.

    Это главный сценарий исходной жалобы, доведённый до крайности.
    """
    fixture = load_fixture("monologue")
    segments, _ = run_sie(fixture)
    found = len({s["speaker"] for s in segments})
    assert found == 1, f"один человек разъехался на {found} спикеров"


def test_fixture_dialog_two_speakers():
    """Диалог двух голосов: ровно два спикера, точность привязки ≥ 95 %."""
    fixture = load_fixture("dialog")
    segments, _ = run_sie(fixture)
    found = len({s["speaker"] for s in segments})
    acc = accuracy(segments, fixture)
    assert found == 2, f"найдено {found} спикеров вместо 2"
    assert acc >= 0.95, f"точность привязки {acc:.1%}"


def test_fixture_crowd_four_speakers_under_music():
    """Четыре голоса под музыкой: музыка не должна их склеивать."""
    fixture = load_fixture("crowd")
    segments, _ = run_sie(fixture)
    found = len({s["speaker"] for s in segments})
    acc = accuracy(segments, fixture)
    assert found == 4, f"найдено {found} спикеров вместо 4"
    assert acc >= 0.90, f"точность привязки под музыкой {acc:.1%}"


def test_fixture_respects_num_speakers_hint():
    """Подсказка о числе голосов выполняется точно."""
    fixture = load_fixture("crowd")
    segments, _ = run_sie(fixture, hint="4")
    assert len({s["speaker"] for s in segments}) == 4

    segments2, _ = run_sie(fixture, hint="2")
    assert len({s["speaker"] for s in segments2}) == 2


def test_fixture_flags_are_a_minority():
    """Флаг «требует проверки» обязан оставаться редким.

    Если помечена половина фильма, человек перестаёт их читать, и точка
    проверки теряет смысл. На чистом материале порог строгий.
    """
    fixture = load_fixture("dialog")
    segments, _ = run_sie(fixture)
    flagged = sum(1 for s in segments if "low_speaker_conf" in s.get("flags", []))
    share = flagged / len(segments)
    assert share <= 0.25, f"помечено {share:.0%} реплик — флаг обесценился"


def test_fixture_registry_survives_second_pass(tmp_path):
    """Повторный прогон сохраняет ID спикеров (INV-3)."""
    import identity as sie
    from core.config import load_config

    fixture = load_fixture("dialog")
    cfg = load_config()

    def once():
        segs = [{"id": s["id"], "start": s["start"], "end": s["end"],
                 "text": s["text"], "speaker": f"RAW{s['id']:03d}", "flags": []}
                for s in fixture["segments"]]
        sie.analyze(fixture["audio"], segs, cfg, raw_turns=None,
                    job_dir=tmp_path, speakers_hint="auto")
        return {s["id"]: s["speaker"] for s in segs}

    first, second = once(), once()
    assert first == second, "ID спикеров изменились между прогонами"


def test_fixture_profiles_built_once_per_speaker(tmp_path):
    """На фикстуре из 2 голосов латенты считаются ровно дважды (INV-1)."""
    import soundfile as sf

    from core.config import load_config
    from identity.embeddings import get_embedder
    from voices import profiles as prof_mod

    fixture = load_fixture("dialog")
    cfg = load_config()
    segments, result = run_sie(fixture)

    calls = {"n": 0}

    class CountingEngine:
        name = "counting"

        def build_conditioning(self, ref_wav):
            calls["n"] += 1
            return ("latent", "emb")

    profiles = prof_mod.build_all(
        tmp_path, FIXTURES / "dialog" / "audio.wav", fixture["audio"],
        segments, result["speakers"], cfg, CountingEngine(),
        get_embedder(cfg), voice_mode="clone")

    speakers_with_clone = sum(
        1 for p in profiles.values() if p["voice"]["mode"] == "clone")
    assert calls["n"] == speakers_with_clone, (
        f"латенты посчитаны {calls['n']} раз при {speakers_with_clone} клонах")
    assert calls["n"] <= 2, "профилей больше, чем голосов в фикстуре"
