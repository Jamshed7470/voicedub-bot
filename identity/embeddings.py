"""Голосовые отпечатки (speaker embeddings) на ECAPA-TDNN.

Один и тот же эмбеддер обслуживает две задачи: кластеризацию спикеров
(identity/clustering.py) и контроль тембра после синтеза (synth/qc.py).
Это осознанно — пороги слияния 0.72 и порог QC 0.70 сравнимы только если
измеряются в одном пространстве.

Все векторы L2-нормированы, поэтому косинусная близость — это скалярное
произведение, и её можно считать матрично.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

from core.config import DATA_DIR

log = logging.getLogger(__name__)

MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
FALLBACK_MODEL_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"
MODEL_DIR = DATA_DIR / "models" / "ecapa"
EMB_SR = 16000
EMB_DIM = 192

_model = None
_model_device = None


def load_model(device: str = "cpu"):
    """Загружает ECAPA один раз на процесс."""
    global _model, _model_device
    if _model is not None and _model_device == device:
        return _model

    from core.sb_compat import patch_speechbrain_lazy_imports
    patch_speechbrain_lazy_imports()

    from speechbrain.inference.speaker import EncoderClassifier

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log.info("ECAPA: загружаю %s (устройство %s)…", MODEL_ID, device)
    kwargs = {"source": MODEL_ID, "savedir": str(MODEL_DIR),
              "run_opts": {"device": device}}
    try:
        # На Windows создание симлинка требует прав администратора или
        # режима разработчика, а speechbrain линкует веса из кэша HF по
        # умолчанию. Без этой стратегии загрузка падает с WinError 1314.
        from speechbrain.utils.fetching import LocalStrategy

        kwargs["local_strategy"] = LocalStrategy.COPY
    except ImportError:      # speechbrain < 1.0 — там симлинков нет
        pass
    _model = EncoderClassifier.from_hparams(**kwargs)
    _model_device = device
    return _model


def l2(vec: np.ndarray) -> np.ndarray:
    """L2-нормировка вектора или матрицы (по последней оси)."""
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.maximum(norm, 1e-9)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусная близость двух векторов (нормировка внутри)."""
    a, b = l2(a).ravel(), l2(b).ravel()
    return float(np.dot(a, b))


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Матрица близостей: (n, d) × (m, d) → (n, m)."""
    return l2(np.atleast_2d(a)) @ l2(np.atleast_2d(b)).T


def centroid(vectors: np.ndarray) -> np.ndarray:
    """Центроид набора эмбеддингов: среднее с повторной нормировкой."""
    vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
    if not len(vectors):
        return np.zeros(EMB_DIM, dtype=np.float32)
    return l2(vectors.mean(axis=0)).astype(np.float32)


def spread(vectors: np.ndarray, cent: np.ndarray | None = None) -> float:
    """Разброс кластера: средний косинус элементов к центроиду.

    Близко к 1 — плотный кластер (один голос). Заметно ниже — внутри
    кластера, скорее всего, разные люди либо очень разные условия записи.
    """
    vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
    if not len(vectors):
        return 0.0
    cent = centroid(vectors) if cent is None else cent
    return float(np.mean(l2(vectors) @ l2(cent).ravel()))


def _audio_key(y: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(y, dtype=np.float32).tobytes()).hexdigest()[:16]


class Embedder:
    """Считает эмбеддинги пачками. Кэширует по хэшу звука.

    Кэш важен: на полуторачасовом фильме полторы тысячи сегментов, и
    переприсвоение идёт в два раунда — без кэша модель прогонялась бы
    трижды по одним и тем же данным.
    """

    def __init__(self, device: str = "cpu", batch: int = 32):
        self.device = device
        self.batch = max(1, int(batch))
        self._cache: dict[str, np.ndarray] = {}

    # ---------- низкий уровень ----------

    def encode_batch(self, waves: list[np.ndarray]) -> np.ndarray:
        """Список моно-сигналов 16 кГц → матрица (n, 192), L2-нормированная."""
        import torch

        if not waves:
            return np.zeros((0, EMB_DIM), dtype=np.float32)

        model = load_model(self.device)
        out: list[np.ndarray] = []
        for i in range(0, len(waves), self.batch):
            chunk = waves[i:i + self.batch]
            lengths = [len(w) for w in chunk]
            width = max(lengths)
            # speechbrain принимает батч с паддингом + относительные длины,
            # иначе тишина в хвосте короткой записи попадёт в эмбеддинг
            padded = np.zeros((len(chunk), width), dtype=np.float32)
            for j, w in enumerate(chunk):
                padded[j, :len(w)] = w
            rel = torch.tensor([n / width for n in lengths], dtype=torch.float32)
            with torch.no_grad():
                emb = model.encode_batch(
                    torch.from_numpy(padded).to(self.device),
                    rel.to(self.device),
                )
            out.append(emb.squeeze(1).cpu().numpy().astype(np.float32))
        return l2(np.concatenate(out, axis=0))

    # ---------- прикладной уровень ----------

    def embed_windows(self, y: np.ndarray, spans: list[tuple[float, float]],
                      sr: int = EMB_SR) -> np.ndarray:
        """Эмбеддинги для интервалов (start, end) в секундах."""
        waves, keys, todo = [], [], []
        result = np.zeros((len(spans), EMB_DIM), dtype=np.float32)

        for idx, (start, end) in enumerate(spans):
            a = max(0, int(start * sr))
            b = min(len(y), int(end * sr))
            piece = y[a:b] if b > a else np.zeros(int(0.2 * sr), dtype=np.float32)
            if len(piece) < int(0.1 * sr):  # совсем пусто — модель выдаст мусор
                piece = np.pad(piece, (0, int(0.1 * sr) - len(piece)))
            key = _audio_key(piece)
            keys.append(key)
            cached = self._cache.get(key)
            if cached is None:
                todo.append(idx)
                waves.append(piece)
            else:
                result[idx] = cached

        if waves:
            fresh = self.encode_batch(waves)
            for pos, idx in enumerate(todo):
                result[idx] = fresh[pos]
                self._cache[keys[idx]] = fresh[pos]
        return result

    def embed_file(self, path: str | Path, max_seconds: float = 60.0) -> np.ndarray:
        """Эмбеддинг целого файла (референс, сэмпл банка, результат синтеза)."""
        import librosa

        y, _ = librosa.load(str(path), sr=EMB_SR, mono=True, duration=max_seconds)
        if not len(y):
            return np.zeros(EMB_DIM, dtype=np.float32)
        return self.encode_batch([y.astype(np.float32)])[0]


_shared: Embedder | None = None


def get_embedder(cfg=None, device: str | None = None) -> Embedder:
    """Общий эмбеддер на процесс: модель весит 80 МБ, второй экземпляр не нужен."""
    global _shared
    dev = device or (cfg.device if cfg is not None else "cpu")
    batch = int(cfg.y("speaker_identity", "embedding_batch", default=32)) if cfg else 32
    if _shared is None or _shared.device != dev:
        _shared = Embedder(dev, batch)
    return _shared
