"""Оценка качества звука: SNR, клиппинг, доля тишины.

Нужна там, где выбирается референс голоса. Референс, собранный из грязных
кусков, даёт нестабильный тембр — а профиль строится один раз и живёт весь
фильм, поэтому цена ошибки здесь выше, чем где-либо ещё в пайплайне.
"""
from __future__ import annotations

import numpy as np

FRAME_SEC = 0.05


def frame_rms(y: np.ndarray, sr: int, frame_sec: float = FRAME_SEC) -> np.ndarray:
    """RMS по коротким кадрам — основа всех оценок ниже."""
    frame = max(1, int(frame_sec * sr))
    if len(y) < frame:
        return np.array([float(np.sqrt(np.mean(y ** 2)) + 1e-9)], dtype=np.float32)
    n = len(y) // frame
    frames = y[:n * frame].reshape(n, frame)
    return (np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9).astype(np.float32)


def noise_floor(y: np.ndarray, sr: int) -> float:
    """Пол шума: 10-й перцентиль кадрового RMS по всей дорожке."""
    return float(np.percentile(frame_rms(y, sr), 10))


def snr_db(piece: np.ndarray, floor: float) -> float:
    """SNR-прокси: активная часть фрагмента против пола шума дорожки.

    Считается по громким кадрам (верхние 50 %), а не по всему фрагменту:
    иначе длинная пауза внутри реплики занижает оценку и хорошая запись
    проигрывает короткой выкрикнутой.
    """
    if not len(piece):
        return -60.0
    rms = frame_rms(piece, 16000)
    active = rms[rms >= np.median(rms)]
    level = float(np.mean(active)) if len(active) else float(np.mean(rms))
    return float(20 * np.log10(level / max(floor, 1e-9)))


def clipping_ratio(piece: np.ndarray, threshold: float = 0.99) -> float:
    """Доля отсчётов на пределе шкалы. Клиппинг слышен в клоне как хрип."""
    if not len(piece):
        return 0.0
    return float(np.mean(np.abs(piece) >= threshold))


def silence_ratio(piece: np.ndarray, sr: int, rel_threshold: float = 0.1) -> float:
    """Доля тихих кадров относительно пика фрагмента."""
    if not len(piece):
        return 1.0
    rms = frame_rms(piece, sr)
    peak = float(np.max(rms))
    if peak <= 1e-8:
        return 1.0
    return float(np.mean(rms < peak * rel_threshold))


def loudness_normalize(y: np.ndarray, sr: int, target_lufs: float = -23.0) -> np.ndarray:
    """Приводит фрагмент к целевой громкости.

    Используется pyloudnorm, если он есть; иначе — оценка по RMS
    (LUFS и RMS не одно и то же, но для референса важна не точность
    измерения, а одинаковый уровень всех кусков склейки).
    """
    y = np.asarray(y, dtype=np.float32)
    if not len(y):
        return y
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        current = meter.integrated_loudness(y.astype(np.float64))
        if np.isfinite(current):
            gain = 10 ** ((target_lufs - current) / 20)
        else:
            raise ValueError("тишина")
    except Exception:  # noqa: BLE001 — приближение по RMS
        rms = float(np.sqrt(np.mean(y ** 2)) + 1e-9)
        target_rms = 10 ** ((target_lufs + 3.0) / 20)   # эмпирический сдвиг
        gain = target_rms / rms

    out = y * gain
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 0.99:                       # нормализация не должна вносить клиппинг
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def trim_silence(y: np.ndarray, sr: int, top_db: float = 35.0) -> np.ndarray:
    """Обрезает тишину по краям и длинные паузы внутри."""
    import librosa

    if not len(y):
        return y
    intervals = librosa.effects.split(y, top_db=top_db)
    if not len(intervals):
        return y
    return np.concatenate([y[a:b] for a, b in intervals]).astype(np.float32)


def score_candidate(duration: float, snr: float, similarity: float,
                    dur_range: tuple[float, float] = (1.5, 12.0),
                    snr_range: tuple[float, float] = (5.0, 35.0)) -> float:
    """Оценка сегмента как кандидата в референс: 0.4·длина + 0.3·SNR + 0.3·тембр.

    Формула из спецификации. Длина и SNR нормируются в [0, 1] по разумным
    границам — без этого SNR в децибелах перевешивал бы всё остальное.
    """
    def norm(value: float, lo: float, hi: float) -> float:
        return float(min(1.0, max(0.0, (value - lo) / max(1e-6, hi - lo))))

    return (0.4 * norm(duration, *dur_range)
            + 0.3 * norm(snr, *snr_range)
            + 0.3 * float(max(0.0, min(1.0, similarity))))
