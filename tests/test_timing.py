"""Тесты подгонки таймингов: atempo <= 1.35, точная укладка на исходный start."""
import shutil
import subprocess

import numpy as np
import pytest

from core.timing import (STATUS_OK, STATUS_TEMPO, STATUS_TOO_LONG, fit_to_slot,
                         start_sample)

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                reason="нужен ffmpeg в PATH")

SR = 44100
TOL = 0.05  # ±50 мс из спецификации


def make_tone(path, duration_s: float) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", f"sine=frequency=440:duration={duration_s}",
         "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", str(path)],
        check=True, capture_output=True,
    )


def probe_duration(path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def test_segment_longer_than_slot_is_compressed(tmp_path):
    """Сегмент длиннее слота ужимается atempo <= 1.35 и попадает в слот ±50 мс."""
    src = tmp_path / "seg.wav"
    make_tone(src, 2.0)
    out = tmp_path / "fitted.wav"

    result = fit_to_slot(src, out, slot_s=1.6, atempo_max=1.35)

    assert result.status == STATUS_TEMPO
    assert result.tempo <= 1.35
    assert abs(probe_duration(out) - 1.6) <= TOL


def test_segment_within_slot_untouched(tmp_path):
    src = tmp_path / "seg.wav"
    make_tone(src, 1.0)
    out = tmp_path / "fitted.wav"

    result = fit_to_slot(src, out, slot_s=1.6, atempo_max=1.35)

    assert result.status == STATUS_OK
    assert abs(probe_duration(out) - 1.0) <= TOL


def test_segment_too_long_requests_compression(tmp_path):
    """Если даже atempo 1.35 не хватает — статус too_long (нужен сжатый перевод)."""
    src = tmp_path / "seg.wav"
    make_tone(src, 2.0)
    out = tmp_path / "fitted.wav"

    result = fit_to_slot(src, out, slot_s=1.0, atempo_max=1.35)

    assert result.status == STATUS_TOO_LONG


def test_segment_placed_exactly_at_start():
    """Сегмент ложится ТОЧНО на исходный start (абсолютная шкала, ±50 мс)."""
    start_s = 3.0
    timeline = np.zeros((SR * 5, 2), dtype=np.float32)
    seg = np.ones((SR, 2), dtype=np.float32)

    pos = start_sample(start_s, SR)
    timeline[pos:pos + len(seg)] += seg

    first_nonzero = int(np.argmax(np.abs(timeline[:, 0]) > 0))
    assert abs(first_nonzero / SR - start_s) <= TOL
    # сегменты никогда не сдвигают друг друга: позиция абсолютная
    assert pos == int(round(start_s * SR))
