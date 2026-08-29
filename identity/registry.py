"""Реестр спикеров: стабильные ID между прогонами.

Без реестра повторная диаризация выдаёт новые метки, и все правки
пользователя (имена, полы, назначенные голоса, переназначенные сегменты)
теряются — сегмент S3 в новом прогоне может оказаться совсем другим
человеком. Реестр сопоставляет новые центроиды со старыми по голосовому
отпечатку и сохраняет прежние ID.

Это прямая реализация INV-3: правки переживают повторную диаризацию.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from identity.embeddings import cosine_matrix

log = logging.getLogger(__name__)

REGISTRY_FILE = "speaker_registry.json"


def assign_ids(clusters, segments: list[dict],
               previous: dict[str, np.ndarray] | None = None,
               match_threshold: float = 0.75) -> dict[str, str]:
    """Кластерам выдаются ID S1, S2… по порядку первого появления.

    Если передан прошлый реестр — сначала переиспользуются старые ID для
    кластеров, узнавших себя по центроиду. Сопоставление жадное и
    взаимно-однозначное: один старый ID не достаётся двум новым кластерам.

    Возвращает {label кластера: speaker_id}.
    """
    ordered = sorted(clusters, key=lambda c: c.first_sec)
    mapping: dict[str, str] = {}
    taken: set[str] = set()

    if previous:
        old_ids = list(previous.keys())
        old_vecs = np.stack([previous[k] for k in old_ids])
        new_vecs = np.stack([c.centroid for c in ordered])
        sims = cosine_matrix(new_vecs, old_vecs)

        # пары в порядке убывания похожести: сильные совпадения занимают
        # свои ID первыми и не отбираются у них более слабыми
        pairs = [(float(sims[i, j]), i, j)
                 for i in range(len(ordered)) for j in range(len(old_ids))]
        pairs.sort(key=lambda t: -t[0])
        used_new: set[int] = set()
        for sim, i, j in pairs:
            if sim < match_threshold:
                break
            if i in used_new or old_ids[j] in taken:
                continue
            mapping[ordered[i].label] = old_ids[j]
            taken.add(old_ids[j])
            used_new.add(i)
            log.info("Реестр: кластер %s узнан как %s (косинус %.3f)",
                     ordered[i].label, old_ids[j], sim)

    # остальные получают свободные номера
    n = 1
    for cl in ordered:
        if cl.label in mapping:
            continue
        while f"S{n}" in taken:
            n += 1
        mapping[cl.label] = f"S{n}"
        taken.add(f"S{n}")

    for cl in ordered:
        for i in cl.seg_idx:
            segments[i]["speaker"] = mapping[cl.label]
    return mapping


def save(job_dir: str | Path, clusters, mapping: dict[str, str]) -> Path:
    """Сохраняет центроиды под финальными ID для будущих прогонов."""
    job_dir = Path(job_dir)
    spk_dir = job_dir / "speakers"
    spk_dir.mkdir(parents=True, exist_ok=True)

    index: dict[str, dict] = {}
    for cl in clusters:
        sid = mapping[cl.label]
        path = spk_dir / sid / "centroid.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), np.asarray(cl.centroid, dtype=np.float32))
        index[sid] = {
            "centroid": f"speakers/{sid}/centroid.npy",
            "raw_label": cl.label,
            "merged_from": cl.merged_from,
            "spread": round(float(cl.spread), 4),
            "segments": len(cl.seg_idx),
            "speech_sec": round(float(cl.speech_sec), 2),
            "first_sec": round(float(cl.first_sec), 2),
        }
    out = job_dir / REGISTRY_FILE
    out.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load(job_dir: str | Path) -> dict[str, np.ndarray]:
    """Читает прошлый реестр: {speaker_id: центроид}. Пусто — если его нет."""
    job_dir = Path(job_dir)
    path = job_dir / REGISTRY_FILE
    if not path.exists():
        return {}
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("Реестр спикеров повреждён — начинаю с чистого листа")
        return {}

    out: dict[str, np.ndarray] = {}
    for sid, rec in index.items():
        p = job_dir / rec["centroid"]
        if p.exists():
            try:
                out[sid] = np.load(str(p)).astype(np.float32)
            except (OSError, ValueError):
                log.warning("Реестр: не читается центроид %s", sid)
    return out
