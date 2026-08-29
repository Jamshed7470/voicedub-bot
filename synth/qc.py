"""Identity QC — проверка, что синтез звучит тем самым голосом.

Три независимые проверки на каждую реплику:

1. Тембр: косинус между отпечатком синтеза и эталоном профиля. Ловит
   главный баг — уплывший голос.
2. Длительность: отношение к ожидаемой. Ловит обрыв синтеза и зацикливание.
3. Обратное распознавание: whisper слушает то, что получилось. Ловит
   «кашу», которую XTTS иногда выдаёт на длинном тексте, — она проходит
   первые две проверки, потому что и тембр, и длина у неё правильные.

Ни одна из проверок не бракует реплику окончательно: после исчерпания
попыток берётся лучший вариант и ставится флаг для студии. Дырка в дубляже
хуже, чем неидеальная реплика.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class QCResult:
    ok: bool
    identity_sim: float = 0.0
    duration_ratio: float = 0.0
    backcheck_cer: float | None = None
    reasons: list[str] = field(default_factory=list)
    seed: int | None = None
    attempts: int = 1

    @property
    def status(self) -> str:
        return "ok" if self.ok else "qc_failed"


# ---------------------------------------------------------------- метрики

def identity_similarity(wav_path: str | Path, profile, embedder) -> float:
    """Косинус отпечатка синтеза с эталоном профиля."""
    if profile is None or profile.identity is None:
        return 0.0
    emb = embedder.embed_file(wav_path, max_seconds=30.0)
    return float(np.dot(emb.ravel(), np.asarray(profile.identity, dtype=np.float32).ravel()))


def audio_duration(path: str | Path) -> float:
    import soundfile as sf

    try:
        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:  # noqa: BLE001
        return 0.0


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate по расстоянию Левенштейна, 0 — идеально."""
    ref = _norm_text(reference)
    hyp = _norm_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return float(prev[-1]) / len(ref)


def _norm_text(s: str) -> str:
    s = (s or "").lower().replace("ё", "е")
    return re.sub(r"[^\w\s]", "", s, flags=re.UNICODE).strip()


# ---------------------------------------------------------------- backcheck

_asr_model = None


def _backcheck_model(cfg):
    """Маленький whisper для обратного распознавания. Грузится один раз."""
    global _asr_model
    if _asr_model is None:
        import faster_whisper

        name = str(cfg.y("synthesis", "asr_backcheck_model", default="small"))
        device = "cuda" if cfg.device == "cuda" else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        log.info("Обратное распознавание: загружаю whisper %s…", name)
        _asr_model = faster_whisper.WhisperModel(name, device=device,
                                                 compute_type=compute)
    return _asr_model


def backcheck(wav_path: str | Path, expected: str, lang: str, cfg) -> float | None:
    """Распознаёт синтез и возвращает CER к ожидаемому тексту."""
    try:
        model = _backcheck_model(cfg)
        segments, _ = model.transcribe(str(wav_path), language=_asr_lang(lang),
                                       beam_size=1, vad_filter=False)
        heard = " ".join(s.text for s in segments)
    except Exception:  # noqa: BLE001 — проверка не должна валить синтез
        log.exception("Обратное распознавание не удалось — пропускаю проверку")
        return None
    return cer(expected, heard)


def _asr_lang(lang: str) -> str:
    return {"zh-cn": "zh"}.get(lang, lang)


# ---------------------------------------------------------------- проверка

def check(wav_path: str | Path, text: str, lang: str, profile,
          expected_dur: float, embedder, cfg,
          with_backcheck: bool = True) -> QCResult:
    """Полная проверка одной синтезированной реплики."""
    reasons: list[str] = []

    threshold = float(cfg.y("synthesis", profile.qc_threshold_key,
                            default=0.70 if profile.mode == "clone" else 0.75))
    sim = identity_similarity(wav_path, profile, embedder)
    if sim < threshold:
        reasons.append(f"тембр {sim:.2f} < {threshold:.2f}")

    lo, hi = cfg.y("synthesis", "duration_ratio_range", default=[0.5, 2.5])
    dur = audio_duration(wav_path)
    ratio = dur / max(0.05, expected_dur)
    if not (float(lo) <= ratio <= float(hi)):
        reasons.append(f"длительность ×{ratio:.2f}")

    cer_value = None
    if with_backcheck and bool(cfg.y("synthesis", "asr_backcheck", default=True)):
        max_cer = float(cfg.y("synthesis", "asr_backcheck_max_cer", default=0.35))
        cer_value = backcheck(wav_path, text, lang, cfg)
        if cer_value is not None and cer_value > max_cer:
            reasons.append(f"распознано с ошибкой {cer_value:.2f}")

    return QCResult(ok=not reasons, identity_sim=round(sim, 4),
                    duration_ratio=round(ratio, 3),
                    backcheck_cer=None if cer_value is None else round(cer_value, 3),
                    reasons=reasons)


# ---------------------------------------------------------------- отчёт

def mean_pairwise_identity(embeddings: list[np.ndarray]) -> float:
    """Средняя попарная схожесть реплик одного спикера.

    Это и есть прямой ответ на жалобу «один человек говорит несколькими
    голосами»: близко к 1 — голос стабилен, заметно ниже — плывёт.
    """
    if len(embeddings) < 2:
        return 1.0
    m = np.stack([np.asarray(e, dtype=np.float32).ravel() for e in embeddings])
    m = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
    sims = m @ m.T
    iu = np.triu_indices(len(m), k=1)
    return float(np.mean(sims[iu]))


def build_report(per_speaker: dict[str, dict]) -> dict:
    """Сводка по спикерам для итогового сообщения и report.md."""
    report = {}
    for sid, data in per_speaker.items():
        embs = data.get("embeddings") or []
        report[sid] = {
            "voice": data.get("voice", "?"),
            "segments": data.get("segments", 0),
            "passed": data.get("passed", 0),
            "mean_pairwise_identity": round(mean_pairwise_identity(embs), 4),
        }
    return report
