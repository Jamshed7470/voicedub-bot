"""Финальный микс: фон + дублированные голоса + скопированные невербальные события.

- background.wav — вся длина, оригинальная громкость;
- каждый синтезированный сегмент кладётся ТОЧНО на исходный start;
- невербальные события копируются из оригинального вокала с кроссфейдом 50 мс;
- громкость голосов нормализуется к уровню оригинальных голосов.

Микс идёт блоками. Держать дорожки целиком нельзя: час стерео 44.1 кГц —
это 1.2 ГБ на массив, а их в миксе четыре (фон, вокал, шина голосов, сумма).
На двухчасовом ролике это гарантированный MemoryError.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from core.timing import start_sample

log = logging.getLogger(__name__)

SR = 44100


def _to_stereo(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return data.astype(np.float32, copy=False)


def _load_seg_44k(path: str | Path) -> np.ndarray:
    """Синтезированный сегмент → stereo 44.1 кГц float32.

    XTTS отдаёт 24 кГц, и раньше каждый сегмент конвертировался отдельным
    вызовом ffmpeg — на длинном ролике это тысячи процессов. Пересчёт в
    памяти делает то же самое без порождения процессов.
    """
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR:
        import librosa
        data = librosa.resample(data.T, orig_sr=sr, target_sr=SR).T
    return _to_stereo(np.ascontiguousarray(data))


class _SegCache:
    """Маленький LRU: сегмент на границе блоков читается дважды, не больше."""

    def __init__(self, size: int = 8):
        self.size = size
        self._items: dict[str, np.ndarray] = {}

    def get(self, path: str) -> np.ndarray:
        arr = self._items.pop(path, None)
        if arr is None:
            arr = _load_seg_44k(path)
        self._items[path] = arr
        while len(self._items) > self.size:
            self._items.pop(next(iter(self._items)))
        return arr


def _fade(x: np.ndarray, fade_len: int) -> np.ndarray:
    """Кроссфейд-огибающая по краям фрагмента."""
    x = x.copy()
    n = min(fade_len, len(x) // 2)
    if n > 0:
        env_in = np.linspace(0.0, 1.0, n)[:, None]
        env_out = np.linspace(1.0, 0.0, n)[:, None]
        x[:n] *= env_in
        x[-n:] *= env_out
    return x


def _read_slice(path: str | Path, a: int, b: int) -> np.ndarray:
    import soundfile as sf
    with sf.SoundFile(str(path)) as f:
        a = max(0, min(a, len(f)))
        b = max(a, min(b, len(f)))
        if b == a:
            return np.zeros((0, 2), dtype=np.float32)
        f.seek(a)
        return _to_stereo(f.read(b - a, dtype="float32", always_2d=True))


def _sum_squares(path: str | Path, block: int) -> tuple[float, int]:
    """Сумма квадратов и число отсчётов дорожки — потоково, без загрузки целиком."""
    import soundfile as sf
    total_sq, n = 0.0, 0
    with sf.SoundFile(str(path)) as f:
        while True:
            chunk = f.read(block, dtype="float32", always_2d=True)
            if not len(chunk):
                break
            total_sq += float(np.sum(chunk.astype(np.float64) ** 2))
            n += chunk.size
    return total_sq, n


def _prepare_items(placed_segments: list[dict], events: list[dict],
                   vocals_wav: Path, total: int, vocals_len: int,
                   fade_len: int) -> list[dict]:
    """Что и куда класть в шину голосов: сегменты (файлами) и события (в памяти)."""
    import soundfile as sf

    items: list[dict] = []
    for item in placed_segments:
        pos = start_sample(item["start"], SR)
        if pos >= total:
            continue
        try:
            info = sf.info(str(item["path"]))
            length = int(round(info.frames * SR / info.samplerate))
        except Exception:  # noqa: BLE001 — битый сегмент не должен ронять микс
            log.warning("Микс: не читается сегмент %s", item.get("path"))
            continue
        items.append({"pos": pos, "end": pos + length, "path": str(item["path"])})

    for ev in events:
        a = start_sample(ev["start"], SR)
        b = min(total, vocals_len, start_sample(ev["end"], SR))
        if b <= a or a >= total:
            continue
        arr = _fade(_read_slice(vocals_wav, a, b), fade_len)
        items.append({"pos": a, "end": a + len(arr), "array": arr})

    items.sort(key=lambda it: it["pos"])
    return items


def _bus_block(items: list[dict], a: int, b: int, cache: _SegCache) -> np.ndarray:
    """Шина дублированных голосов на отрезке [a, b)."""
    bus = np.zeros((b - a, 2), dtype=np.float32)
    for it in items:
        if it["end"] <= a or it["pos"] >= b:
            continue
        arr = it["array"] if "array" in it else cache.get(it["path"])
        i0, i1 = max(a, it["pos"]), min(b, it["pos"] + len(arr))
        if i1 > i0:
            bus[i0 - a:i1 - a] += arr[i0 - it["pos"]:i1 - it["pos"]]
    return bus


def build_mix(job_dir: str | Path, background_wav: str | Path, vocals_wav: str | Path,
              placed_segments: list[dict], events: list[dict], cfg,
              keep_background: bool = True) -> Path:
    """Собирает финальную аудиодорожку dubbed.wav (44.1 кГц stereo).

    placed_segments: [{"start": float, "path": str}] — синтезированные сегменты.
    events: [{"start", "end"}] — отрезки, копируемые из оригинального вокала.
    """
    import soundfile as sf

    job_dir = Path(job_dir)
    background_wav, vocals_wav = Path(background_wav), Path(vocals_wav)
    block = max(1, int(float(cfg.y("mix", "block_s", default=30)) * SR))

    with sf.SoundFile(str(background_wav)) as bg:
        assert bg.samplerate == SR, f"Ожидалось {SR} Гц, получено {bg.samplerate}"
        total = len(bg)
    with sf.SoundFile(str(vocals_wav)) as voc:
        vocals_len = len(voc)

    fade_len = int(SR * float(cfg.y("events", "crossfade_ms", default=50)) / 1000)
    items = _prepare_items(placed_segments, events, vocals_wav, total,
                           vocals_len, fade_len)
    cache = _SegCache()

    # --- громкость голосов к уровню оригинальных (RMS) ---
    voc_sq, voc_n = _sum_squares(vocals_wav, block)
    orig_rms = float(np.sqrt(voc_sq / max(1, voc_n)) + 1e-12)
    # сумма квадратов шины — по её содержимому: пересечения реплик редки,
    # а полный проход по шине ради одного числа стоит лишних минут на длинном ролике
    bus_sq = 0.0
    for it in items:
        arr = it["array"] if "array" in it else cache.get(it["path"])
        # хвост реплики за концом дорожки в микс не попадёт — и в счёт тоже
        usable = arr[:max(0, total - it["pos"])]
        bus_sq += float(np.sum(usable.astype(np.float64) ** 2))
    dub_rms = float(np.sqrt(bus_sq / max(1, total * 2)) + 1e-12)

    gain = 1.0
    if dub_rms > 1e-9 and orig_rms > 1e-9:
        gain = float(np.clip(orig_rms / dub_rms, 0.25, 4.0))
        log.info("Микс: выравнивание громкости голосов, gain %.2f", gain)

    # --- проход 1: пиковый уровень (PCM_16 клиппинг необратим) ---
    peak = 1e-12
    with sf.SoundFile(str(background_wav)) as bg:
        for a in range(0, total, block):
            b = min(total, a + block)
            chunk = _to_stereo(bg.read(b - a, dtype="float32", always_2d=True))
            if not keep_background:
                chunk = np.zeros_like(chunk)
            mix = chunk + gain * _bus_block(items, a, b, cache)
            peak = max(peak, float(np.max(np.abs(mix))) if len(mix) else 0.0)

    peak_target = 10 ** (float(cfg.y("mix", "peak_dbfs", default=-1.0)) / 20)
    scale = gain * (peak_target / peak if peak > peak_target else 1.0)
    bg_scale = (peak_target / peak) if peak > peak_target else 1.0

    # --- проход 2: запись ---
    out = job_dir / "dubbed.wav"
    with sf.SoundFile(str(background_wav)) as bg, \
            sf.SoundFile(str(out), "w", samplerate=SR, channels=2,
                         subtype="PCM_16") as dst:
        for a in range(0, total, block):
            b = min(total, a + block)
            chunk = _to_stereo(bg.read(b - a, dtype="float32", always_2d=True))
            if not keep_background:
                chunk = np.zeros_like(chunk)
            mix = bg_scale * chunk + scale * _bus_block(items, a, b, cache)
            dst.write(mix)

    log.info("Микс собран: %.1f мин, реплик %d, событий %d",
             total / SR / 60, len(placed_segments), len(events))
    return out
