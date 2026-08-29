"""Журнал операций пользователя и отмена последнего действия.

Хранятся последние 50 операций. Отмена работает по принципу «вернуть то,
что было записано в before» — это надёжнее обратных операций: не нужно
описывать, как отменить каждое действие, достаточно помнить предыдущее
состояние затронутых объектов.
"""
from __future__ import annotations

import logging
from typing import Any

from project.schema import Project

log = logging.getLogger(__name__)

HISTORY_LIMIT = 50


def snapshot_segments(proj: Project, seg_ids: list[int]) -> list[dict]:
    return [s.model_dump(mode="json") for s in proj.segments if s.id in set(seg_ids)]


def snapshot_speakers(proj: Project, sids: list[str]) -> list[dict]:
    return [s.model_dump(mode="json") for s in proj.speakers if s.id in set(sids)]


def record(proj: Project, op: str, before: Any = None, after: Any = None) -> None:
    proj.push_history(op, before, after, limit=HISTORY_LIMIT)


def undo(proj: Project) -> str | None:
    """Откатывает последнюю операцию. Возвращает её название или None."""
    from project.schema import Segment, Speaker

    while proj.history:
        op = proj.history.pop()
        before = op.before or {}
        if not isinstance(before, dict):
            continue

        restored = False
        for raw in before.get("segments", []) or []:
            seg = Segment.model_validate(raw)
            for i, cur in enumerate(proj.segments):
                if cur.id == seg.id:
                    proj.segments[i] = seg
                    restored = True
                    break
        for raw in before.get("speakers", []) or []:
            sp = Speaker.model_validate(raw)
            for i, cur in enumerate(proj.speakers):
                if cur.id == sp.id:
                    proj.speakers[i] = sp
                    restored = True
                    break
        # удалённые операцией сегменты (склейка) возвращаются целиком
        for raw in before.get("segments_removed", []) or []:
            seg = Segment.model_validate(raw)
            if not any(s.id == seg.id for s in proj.segments):
                proj.segments.append(seg)
                restored = True
        for seg_id in before.get("segments_added", []) or []:
            proj.segments = [s for s in proj.segments if s.id != seg_id]
            restored = True
        # удалённый спикер возвращается целиком, иначе его реплики повиснут
        # без владельца и рендер остановится
        for raw in before.get("speakers_removed", []) or []:
            sid = raw if isinstance(raw, str) else raw.get("id")
            snapshot = next((x for x in (before.get("speakers") or [])
                             if x.get("id") == sid), None)
            if snapshot and not any(s.id == sid for s in proj.speakers):
                proj.speakers.append(Speaker.model_validate(snapshot))
                restored = True
        for sid in before.get("speakers_added", []) or []:
            proj.speakers = [s for s in proj.speakers if s.id != sid]
            restored = True

        if restored:
            proj.segments.sort(key=lambda s: (s.start, s.id))
            proj.recompute_stats()
            log.info("Отменена операция «%s»", op.op)
            return op.op
    return None
