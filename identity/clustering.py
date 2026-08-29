"""Слияние кластеров pyannote и переприсвоение сегментов.

pyannote дробит одного человека на несколько кластеров: смена эмоции,
отход от микрофона, музыка под речью, телефонный голос. Каждый лишний
кластер в старой схеме получал собственный профиль голоса — отсюда
«три-четыре голоса на одного человека». Здесь кластеры сводятся обратно
по голосовым отпечаткам.

Ключевая защита от слипания разных людей: два кластера, которые заметное
время звучат ОДНОВРЕМЕННО, не сливаются никогда, как бы ни были похожи.
Один человек не может перебивать сам себя.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from identity.embeddings import Embedder, centroid, cosine_matrix, spread

log = logging.getLogger(__name__)


@dataclass
class Cluster:
    """Кластер = кандидат в спикеры."""
    label: str
    seg_idx: list[int] = field(default_factory=list)
    centroid: np.ndarray | None = None
    spread: float = 0.0
    speech_sec: float = 0.0
    first_sec: float = float("inf")
    merged_from: list[str] = field(default_factory=list)


def overlap_seconds(a: list[tuple[float, float]],
                    b: list[tuple[float, float]]) -> float:
    """Сколько секунд два набора интервалов звучат одновременно.

    Списки сортируются и проходятся двумя указателями: на полутора тысячах
    сегментов попарное сравнение всех со всеми стоило бы миллионы операций.
    """
    if not a or not b:
        return 0.0
    a = sorted(a)
    b = sorted(b)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def build_clusters(segments: list[dict], embeddings: np.ndarray) -> list[Cluster]:
    """Группирует сегменты по исходной метке диаризации."""
    by_label: dict[str, Cluster] = {}
    for idx, seg in enumerate(segments):
        label = seg["speaker"]
        cl = by_label.get(label)
        if cl is None:
            cl = by_label[label] = Cluster(label=label)
        cl.seg_idx.append(idx)
        cl.speech_sec += seg["end"] - seg["start"]
        cl.first_sec = min(cl.first_sec, seg["start"])

    for cl in by_label.values():
        vecs = embeddings[cl.seg_idx]
        cl.centroid = centroid(vecs)
        cl.spread = spread(vecs, cl.centroid)
    return sorted(by_label.values(), key=lambda c: c.first_sec)


def _spans(segments: list[dict], idx: list[int]) -> list[tuple[float, float]]:
    return [(segments[i]["start"], segments[i]["end"]) for i in idx]


def merge_clusters(clusters: list[Cluster], segments: list[dict],
                   embeddings: np.ndarray,
                   merge_threshold: float, overlap_block_sec: float,
                   num_speakers: int | None = None,
                   min_speakers: int | None = None,
                   max_speakers: int | None = None) -> list[Cluster]:
    """Агломеративное слияние кластеров по близости центроидов.

    На каждом шаге сливается самая близкая допустимая пара, центроид
    пересчитывается по всем сегментам объединения. Останов — когда ни одна
    пара не проходит порог (или когда достигнуто заданное число спикеров).
    """
    clusters = list(clusters)
    if len(clusters) < 2:
        return clusters

    floor = max(1, int(min_speakers or 1))
    if num_speakers:
        floor = max(1, int(num_speakers))

    while len(clusters) > floor:
        target = num_speakers or max_speakers
        forced = bool(target and len(clusters) > target)

        best_pair, best_sim = None, -1.0
        cents = np.stack([c.centroid for c in clusters])
        sims = cosine_matrix(cents, cents)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                sim = float(sims[i, j])
                if sim <= best_sim:
                    continue
                if not forced and sim < merge_threshold:
                    continue
                # два голоса, звучащие одновременно, — это два человека
                simul = overlap_seconds(_spans(segments, clusters[i].seg_idx),
                                        _spans(segments, clusters[j].seg_idx))
                if simul >= overlap_block_sec:
                    continue
                best_pair, best_sim = (i, j), sim

        if best_pair is None:
            break

        i, j = best_pair
        a, b = clusters[i], clusters[j]
        keep, drop = (a, b) if a.speech_sec >= b.speech_sec else (b, a)
        keep.seg_idx = sorted(keep.seg_idx + drop.seg_idx)
        keep.speech_sec += drop.speech_sec
        keep.first_sec = min(keep.first_sec, drop.first_sec)
        keep.merged_from.append(drop.label)
        keep.merged_from.extend(drop.merged_from)
        # метка сегмента обновляется сразу: между слиянием и выдачей ID
        # состояние должно быть согласованным, иначе повторный build_clusters
        # снова разнесёт их по исчезнувшим кластерам
        for idx in drop.seg_idx:
            segments[idx]["speaker"] = keep.label
        log.info("Слияние кластеров %s + %s (косинус %.3f)%s",
                 keep.label, drop.label, best_sim,
                 " — по заданному числу спикеров" if forced else "")
        clusters = [c for k, c in enumerate(clusters) if k != (i if keep is b else j)]
        # центроид объединения считается по всем его сегментам, а не как
        # среднее двух центроидов: у кластеров разный вес
        vecs = embeddings[keep.seg_idx]
        keep.centroid = centroid(vecs)
        keep.spread = spread(vecs, keep.centroid)

    return sorted(clusters, key=lambda c: c.first_sec)


def reliability(duration: float, short_sec: float, reliable_sec: float) -> float:
    """Насколько можно верить отпечатку сегмента такой длины: 0…1.

    Замер на реальном фильме (1565 реплик, ECAPA): близость сегмента к
    собственному центроиду — 0.66 при длине ≥ 2 с и всего 0.34 при длине
    < 1 с. Это шум измерения, а не неуверенность в спикере. Единый порог
    для тех и других помечал бы половину фильма как «требует проверки» —
    и флаг переставал бы что-либо значить.
    """
    if duration >= reliable_sec:
        return 1.0
    if duration <= short_sec:
        return 0.0
    return (duration - short_sec) / (reliable_sec - short_sec)


def reassign(segments: list[dict], embeddings: np.ndarray,
             clusters: list[Cluster], min_sim: float, margin: float,
             rounds: int = 2, min_sim_short: float | None = None,
             short_sec: float = 0.8, reliable_sec: float = 3.0) -> list[Cluster]:
    """Каждый сегмент уходит к ближайшему центроиду — если запас достаточен.

    Сегмент, у которого два кандидата почти равны, НЕ перекидывается:
    вместо этого он получает флаг low_speaker_conf и попадает в студию на
    проверку человеком. Угадывать здесь хуже, чем признать неуверенность.

    Планка похожести зависит от длины реплики: короткая даёт шумный
    отпечаток, и требовать от неё той же близости, что от четырёхсекундной,
    значит браковать её за длину, а не за сомнительность.
    """
    if min_sim_short is None:
        min_sim_short = min_sim * 0.6
    for _ in range(max(1, rounds)):
        cents = np.stack([c.centroid for c in clusters])
        sims = cosine_matrix(embeddings, cents)          # (n_seg, n_cluster)
        order = np.argsort(-sims, axis=1)

        buckets: dict[str, list[int]] = {c.label: [] for c in clusters}
        label_of = {k: c.label for k, c in enumerate(clusters)}
        current = {}
        for k, c in enumerate(clusters):
            for i in c.seg_idx:
                current[i] = k

        for i, seg in enumerate(segments):
            best = int(order[i, 0])
            best_sim = float(sims[i, best])
            second = float(sims[i, order[i, 1]]) if len(clusters) > 1 else -1.0
            gap = best_sim - second
            now = current.get(i, best)

            seg["speaker_confidence"] = round(float(sims[i, now]), 4)
            seg["speaker_margin"] = round(gap, 4)

            rel = reliability(seg["end"] - seg["start"], short_sec, reliable_sec)
            bar = min_sim_short + (min_sim - min_sim_short) * rel
            confident = best_sim >= bar and gap >= margin
            if confident:
                buckets[label_of[best]].append(i)
                seg["speaker"] = label_of[best]
                seg["speaker_confidence"] = round(best_sim, 4)
            else:
                # сегмент, у которого два кандидата почти равны, остаётся
                # там, где был, и помечается — угадывать здесь хуже, чем
                # признать неуверенность и показать её человеку.
                # Флаг ставится независимо от того, совпал ли argmax с
                # текущей меткой: ненадёжна сама привязка, а не только её смена
                buckets[label_of[now]].append(i)
                seg["speaker"] = label_of[now]
                _add_flag(seg, "low_speaker_conf")

        for c in clusters:
            c.seg_idx = sorted(buckets[c.label])
            if c.seg_idx:
                c.centroid = centroid(embeddings[c.seg_idx])
                c.spread = spread(embeddings[c.seg_idx], c.centroid)
                c.speech_sec = sum(segments[i]["end"] - segments[i]["start"]
                                   for i in c.seg_idx)
                c.first_sec = min(segments[i]["start"] for i in c.seg_idx)
        clusters = [c for c in clusters if c.seg_idx]

    return sorted(clusters, key=lambda c: c.first_sec)


def mark_isolated(segments: list[dict], max_sec: float, margin: float) -> int:
    """Помечает одиночную короткую реплику внутри чужой длинной серии.

    Такой сегмент автоматически не переворачивается: чаще это действительно
    короткая вставка другого человека. Но именно здесь ошибка привязки
    заметнее всего на слух, поэтому он идёт в студию с флагом.
    """
    count = 0
    for i in range(1, len(segments) - 1):
        seg, prev, nxt = segments[i], segments[i - 1], segments[i + 1]
        if seg["end"] - seg["start"] > max_sec:
            continue
        if prev["speaker"] != nxt["speaker"] or prev["speaker"] == seg["speaker"]:
            continue
        if abs(float(seg.get("speaker_margin", 1.0))) < margin:
            _add_flag(seg, "suspicious_isolated")
            count += 1
    return count


def mark_overlaps(segments: list[dict], raw_turns: list[dict],
                  ratio: float) -> int:
    """Отмечает сегменты, где на интервале говорят двое.

    Текст остаётся за доминирующим по времени спикером — синтезировать оба
    голоса в один слот нельзя, они наложатся друг на друга.
    """
    if not raw_turns:
        return 0
    count = 0
    for seg in segments:
        dur = max(1e-6, seg["end"] - seg["start"])
        per_speaker: dict[str, float] = {}
        for turn in raw_turns:
            lo = max(seg["start"], turn["start"])
            hi = min(seg["end"], turn["end"])
            if hi > lo:
                per_speaker[turn["speaker"]] = per_speaker.get(turn["speaker"], 0.0) + (hi - lo)
        if len(per_speaker) < 2:
            continue
        shared = sorted(per_speaker.values(), reverse=True)[1] / dur
        if shared >= ratio:
            seg["overlap"] = True
            _add_flag(seg, "overlap")
            others = sorted(per_speaker.items(), key=lambda kv: -kv[1])[1:]
            seg["overlap_with"] = [k for k, _ in others]
            count += 1
    return count


def _add_flag(seg: dict, flag: str) -> None:
    flags = seg.setdefault("flags", [])
    if flag not in flags:
        flags.append(flag)


def run(segments: list[dict], embeddings: np.ndarray, cfg,
        raw_turns: list[dict] | None = None,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None) -> list[Cluster]:
    """Полный проход SIE: кластеры → слияние → переприсвоение → флаги."""
    c = lambda k, d: cfg.y("speaker_identity", k, default=d)  # noqa: E731

    clusters = build_clusters(segments, embeddings)
    log.info("SIE: кластеров от диаризации — %d", len(clusters))

    clusters = merge_clusters(
        clusters, segments, embeddings,
        merge_threshold=float(c("merge_threshold", 0.72)),
        overlap_block_sec=float(c("overlap_merge_block_sec", 2.0)),
        num_speakers=num_speakers, min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    log.info("SIE: после слияния — %d", len(clusters))

    clusters = reassign(
        segments, embeddings, clusters,
        min_sim=float(c("min_assign_sim", 0.55)),
        margin=float(c("reassign_margin", 0.08)),
        rounds=int(c("rounds", 2)),
        min_sim_short=float(c("min_assign_sim_short", 0.33)),
        short_sec=float(c("min_segment_sec_for_embedding", 0.8)),
        reliable_sec=float(c("reliable_segment_sec", 3.0)),
    )

    isolated = mark_isolated(segments, float(c("isolated_max_sec", 1.5)),
                             float(c("isolated_margin", 0.05)))
    overlaps = mark_overlaps(segments, raw_turns or [],
                             float(c("overlap_ratio", 0.3)))
    low = sum(1 for s in segments if "low_speaker_conf" in s.get("flags", []))
    log.info("SIE: итог %d спикеров; флаги — низкая уверенность %d, "
             "одиночные %d, наложения %d", len(clusters), low, isolated, overlaps)
    return clusters
