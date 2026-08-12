"""Эмоция каждого речевого сегмента (speechbrain, IEMOCAP)."""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

LABEL_MAP = {"neu": "neutral", "hap": "happy", "sad": "sad", "ang": "angry"}

_classifier = None


def _get_classifier(cfg):
    global _classifier
    if _classifier is None:
        from core.sb_compat import patch_speechbrain_lazy_imports
        patch_speechbrain_lazy_imports()
        from speechbrain.inference.interfaces import foreign_class
        model = cfg.y("emotions", "model",
                      default="speechbrain/emotion-recognition-wav2vec2-IEMOCAP")
        _classifier = foreign_class(
            source=model,
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            run_opts={"device": cfg.device},
        )
    return _classifier


MAX_PIECE_S = 20.0   # длиннее модели не нужно: эмоция реплики слышна и по началу


def classify_segments(vocals16_path: str | Path, segments: list[dict],
                      cfg, tmp_dir: str | Path, progress=None) -> None:
    """Присваивает каждому сегменту seg["emotion"] и seg["emotion_conf"] (in-place).

    Фрагменты читаются прямо из wav по смещению. Раньше каждый сегмент
    вырезался отдельным вызовом ffmpeg — на часовом ролике это тысячи
    запусков процесса и столько же чтений дорожки с начала.
    """
    import soundfile as sf
    import torch

    clf = _get_classifier(cfg)
    n = len(segments)

    with sf.SoundFile(str(vocals16_path)) as f:
        sr = f.samplerate
        total = len(f)
        for i, seg in enumerate(segments):
            seg["emotion"] = "neutral"
            seg["emotion_conf"] = 0.0
            if seg["end"] - seg["start"] < 0.5:
                continue
            a = max(0, min(total, int(seg["start"] * sr)))
            b = max(a, min(total, int(min(seg["end"], seg["start"] + MAX_PIECE_S) * sr)))
            if b - a < int(0.5 * sr):
                continue
            try:
                # ВАЖНО: classify_file ломает Windows-пути (fetch() съедает "\"
                # после буквы диска) — грузим аудио сами и зовём classify_batch
                f.seek(a)
                y = f.read(b - a, dtype="float32", always_2d=False)
                if y.ndim > 1:
                    y = y.mean(axis=1)
                wav = torch.from_numpy(y).unsqueeze(0)
                _, score, _, text_lab = clf.classify_batch(wav)
                label = (text_lab[0] if isinstance(text_lab, (list, tuple))
                         else str(text_lab))
                seg["emotion"] = LABEL_MAP.get(label, "neutral")
                seg["emotion_conf"] = round(float(score), 3)
            except Exception:  # noqa: BLE001
                log.exception("Эмоции: ошибка на сегменте %s", seg["id"])
            if progress and n:
                progress(int(100 * (i + 1) / n))
