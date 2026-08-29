"""Кастинг: какой голос кому достанется.

Правило, которое отличает кастинг от простого «ближайшего по тону»:
уникальность. Два спикера не должны получить один голос, иначе зритель
перестаёт различать персонажей. Это задача о назначениях, и решается она
венгерским алгоритмом целиком, а не жадным перебором по одному спикеру:
жадность отдаёт лучший голос первому попавшемуся и загоняет остальных в
худшие варианты.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def score_matrix(speakers: list[dict], voices: list, identities: np.ndarray,
                 lang: str | None, language_bonus: float) -> np.ndarray:
    """Оценка «спикер × голос»: близость тембра плюс поправки.

    Пол — не слагаемое, а запрет: женскому спикеру мужской голос не
    подходит ни при какой близости тембра. Реализуется большим штрафом,
    чтобы задача назначения осталась разрешимой, когда голосов нужного
    пола физически не хватает.
    """
    scores = np.zeros((len(speakers), len(voices)), dtype=np.float32)
    for i, spk in enumerate(speakers):
        centroid = spk.get("centroid")
        gender = spk.get("gender", "unknown")
        is_child = spk.get("age") == "child"

        for j, voice in enumerate(voices):
            if centroid is not None and identities[j].any():
                sim = float(np.dot(np.asarray(centroid, dtype=np.float32).ravel(),
                                   identities[j].ravel()))
            else:
                sim = _f0_similarity(spk.get("f0_median"), voice.f0_hz)
            score = sim

            if lang and voice.languages and lang in voice.languages:
                score += language_bonus
            if gender != "unknown" and voice.gender != "unknown" and voice.gender != gender:
                score -= 10.0                    # другой пол — крайняя мера
            if voice.is_child != is_child:
                score -= 0.5                     # возраст важен, но мягче пола
            scores[i, j] = score
    return scores


def _f0_similarity(speaker_f0: float | None, voice_f0: float | None) -> float:
    """Запасная мера, когда отпечатков нет: близость основного тона в октавах."""
    if not speaker_f0 or not voice_f0:
        return 0.0
    octaves = abs(np.log2(voice_f0 / speaker_f0))
    return float(max(0.0, 1.0 - octaves))


def assign_voices(profiles: dict, cfg, bank=None, embedder=None,
                  lang: str | None = None):
    """Назначает голоса банка спикерам в режиме preset.

    Меняет profiles на месте: заполняет voice.preset_id и
    voice.casting_candidates. Возвращает банк (или None, если он не нужен
    и не доступен) — рендер берёт из него профили пресетов.
    """
    from voices.bank import get_bank

    need = [sid for sid, p in profiles.items()
            if (p.get("voice") or {}).get("mode") == "preset"
            and not (p.get("voice") or {}).get("edited_by_user")]
    if not need:
        return bank

    bank = bank or get_bank()
    voices = bank.all()
    if not voices:
        log.warning("Банк голосов пуст: %d спикерам без клона нечего назначить. "
                    "Соберите банк: python -m scripts.build_voice_bank --from-xtts",
                    len(need))
        return None

    if embedder is None:
        from identity.embeddings import get_embedder
        embedder = get_embedder(cfg)
    identities = bank.identities(voices, embedder)

    unique = bool(cfg.y("casting", "unique_voices", default=True))
    bonus = float(cfg.y("casting", "language_bonus", default=0.05))
    top_n = int(cfg.y("casting", "candidates_top_n", default=3))

    speakers = [dict(profiles[sid], id=sid) for sid in need]
    scores = score_matrix(speakers, voices, identities, lang, bonus)

    # голосов может не хватить на всех — тогда часть спикеров неизбежно
    # делит голос; предупреждаем об этом явно, а не молча
    if unique and len(need) > len(voices):
        log.warning("Голосов в банке %d, а спикеров на озвучку %d — "
                    "некоторым достанется один и тот же голос",
                    len(voices), len(need))

    chosen = _solve(scores, unique)

    for row, sid in enumerate(need):
        j = chosen[row]
        voice = voices[j]
        entry = profiles[sid]
        entry["voice"]["preset_id"] = voice.id
        entry["voice"]["locked"] = True
        order = np.argsort(-scores[row])[:top_n]
        entry["voice"]["casting_candidates"] = [
            {"preset_id": voices[k].id, "display_name": voices[k].display_name,
             "score": round(float(scores[row, k]), 4)} for k in order
        ]
        entry["bank_voice"] = voice.display_name
        log.info("Кастинг: %s (%s) → «%s» (оценка %.3f)", sid,
                 entry.get("gender", "?"), voice.display_name, scores[row, j])

    # спикер с наибольшим временем речи получает метку рассказчика
    if profiles:
        lead = max(profiles, key=lambda s: profiles[s].get("speech_total_s", 0))
        profiles[lead]["role"] = "narrator"
    return bank


def _solve(scores: np.ndarray, unique: bool) -> list[int]:
    """Назначение голосов: венгерский алгоритм либо просто лучший каждому."""
    n_spk, n_voice = scores.shape
    if not unique or n_voice < n_spk:
        if unique:
            return _greedy_with_reuse(scores)
        return [int(np.argmax(scores[i])) for i in range(n_spk)]

    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-scores)
        out = [0] * n_spk
        for r, c in zip(rows, cols):
            out[int(r)] = int(c)
        return out
    except ImportError:
        log.warning("scipy недоступен — назначаю голоса жадно")
        return _greedy_with_reuse(scores)


def _greedy_with_reuse(scores: np.ndarray) -> list[int]:
    """Жадное назначение: свободные голоса раздаются в порядке уверенности.

    Спикеры сортируются по тому, насколько их лучший голос лучше второго:
    у кого выбор очевиднее, тот получает своё первым. Когда свободные
    кончились, начинается повтор — но не раньше.
    """
    n_spk, n_voice = scores.shape
    order = sorted(range(n_spk),
                   key=lambda i: -(np.sort(scores[i])[-1] - np.sort(scores[i])[-2]
                                   if n_voice > 1 else 0))
    used: set[int] = set()
    out = [0] * n_spk
    for i in order:
        ranking = np.argsort(-scores[i])
        pick = next((int(j) for j in ranking if int(j) not in used), int(ranking[0]))
        used.add(pick)
        out[i] = pick
    return out
