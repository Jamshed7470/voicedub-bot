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


def classify_segments(vocals16_path: str | Path, segments: list[dict],
                      cfg, tmp_dir: str | Path, progress=None) -> None:
    """Присваивает каждому сегменту seg["emotion"] и seg["emotion_conf"] (in-place)."""
    from core.media import cut_fragment

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    clf = _get_classifier(cfg)

    n = len(segments)
    for i, seg in enumerate(segments):
        seg["emotion"] = "neutral"
        seg["emotion_conf"] = 0.0
        if seg["end"] - seg["start"] < 0.5:
            continue
        piece = tmp_dir / f"emo_{seg['id']}.wav"
        try:
            cut_fragment(vocals16_path, piece, seg["start"], seg["end"],
                         sr=16000, mono=True)
            _, score, _, text_lab = clf.classify_file(str(piece))
            label = text_lab[0] if isinstance(text_lab, (list, tuple)) else str(text_lab)
            seg["emotion"] = LABEL_MAP.get(label, "neutral")
            seg["emotion_conf"] = round(float(score), 3)
        except Exception:  # noqa: BLE001
            log.exception("Эмоции: ошибка на сегменте %s", seg["id"])
        finally:
            piece.unlink(missing_ok=True)
        if progress and n:
            progress(int(100 * (i + 1) / n))
