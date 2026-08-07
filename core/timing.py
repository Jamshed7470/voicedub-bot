"""Подгонка длительности синтезированных сегментов под исходные слоты.

Правила (из спецификации):
- каждый сегмент кладётся ТОЧНО на исходный start (абсолютная шкала времени);
- если синтез длиннее слота — ffmpeg atempo до ×1.35;
- если всё равно длиннее — вернуть статус need_compress (запросить сжатый перевод);
- если короче — оставить как есть (паузы заполняются тишиной/фоном).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from core.media import probe_duration, run

log = logging.getLogger(__name__)

STATUS_OK = "ok"            # влезает без изменений
STATUS_TEMPO = "tempo"      # ужат atempo
STATUS_TOO_LONG = "too_long"  # не влезает даже с atempo_max → нужен сжатый перевод


@dataclass
class FitResult:
    status: str
    path: Path
    duration: float
    tempo: float = 1.0


def fit_to_slot(in_wav: str | Path, out_wav: str | Path, slot_s: float,
                atempo_max: float = 1.35, tolerance_s: float = 0.05) -> FitResult:
    """Подгоняет длительность in_wav под слот slot_s (секунды)."""
    in_wav, out_wav = Path(in_wav), Path(out_wav)
    dur = probe_duration(in_wav)

    if dur <= slot_s + tolerance_s:
        if in_wav != out_wav:
            run(["ffmpeg", "-y", "-i", str(in_wav), "-c:a", "pcm_s16le", str(out_wav)],
                desc="ffmpeg copy")
        return FitResult(STATUS_OK, out_wav, dur, 1.0)

    tempo = dur / slot_s
    if tempo > atempo_max + 1e-6:
        return FitResult(STATUS_TOO_LONG, in_wav, dur, tempo)

    # atempo поддерживает 0.5–100 в одном звене; наш диапазон ≤1.35 — одно звено
    run(["ffmpeg", "-y", "-i", str(in_wav),
         "-filter:a", f"atempo={tempo:.6f}",
         "-c:a", "pcm_s16le", str(out_wav)], desc="ffmpeg atempo")
    new_dur = probe_duration(out_wav)
    log.debug("atempo %.3f: %.3fs -> %.3fs (слот %.3fs)", tempo, dur, new_dur, slot_s)
    return FitResult(STATUS_TEMPO, out_wav, new_dur, tempo)


def start_sample(start_s: float, sr: int) -> int:
    """Абсолютная позиция сегмента в сэмплах — сегменты никогда не сдвигают друг друга."""
    return max(0, round(start_s * sr))
