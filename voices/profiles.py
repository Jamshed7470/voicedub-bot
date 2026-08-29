"""Заблокированный профиль голоса: строится ОДИН раз на спикера.

Это центральное место исправления главного бага. Раньше XTTS получал новый
референс почти на каждую реплику и пересчитывал по нему тембр — за фильм
один человек успевал прозвучать несколькими разными голосами. Теперь:

    один speaker_id → один ref_main.wav → один voice_profile.pt → весь фильм

Профиль содержит готовые латенты XTTS. После его сборки
`get_conditioning_latents` в пайплайне не вызывается больше нигде — за этим
следит тест `test_no_per_segment_reference`.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from identity import quality

log = logging.getLogger(__name__)

REF_SR = 24000          # частота референса для XTTS
EMB_SR = 16000          # частота для ECAPA и анализа пола
PROFILE_FORMAT = 2      # версия формата voice_profile.pt


@dataclass
class VoiceProfile:
    """Готовый к синтезу голос. Собирается один раз, дальше только читается."""
    speaker_id: str
    mode: str                       # clone | preset
    gpt_cond_latent: object = None  # torch.Tensor
    speaker_embedding: object = None
    identity: np.ndarray | None = None   # эталон ECAPA для Identity QC
    ref_path: str | None = None
    preset_id: str | None = None
    engine: str = "xtts_v2"
    ref_sha256: str = ""
    locked: bool = True
    meta: dict = field(default_factory=dict)

    @property
    def qc_threshold_key(self) -> str:
        return "identity_qc_min_clone" if self.mode == "clone" else "identity_qc_min_preset"


# ---------------------------------------------------------------- референс

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select_reference(segments: list[dict], y_ref: np.ndarray, y16: np.ndarray,
                     centroid: np.ndarray | None, embedder, cfg) -> dict:
    """Отбирает лучшие куски речи спикера и склеивает из них референс.

    Кандидат обязан быть чистым: без наложения, уверенно привязанный,
    достаточно длинный, не тише порога SNR, нейтральный по эмоции и без
    смеха/плача. Эмоциональный референс — прямой путь к «истеричному»
    тембру на весь фильм, поэтому такие куски отбрасываются.
    """
    c = lambda k, d: cfg.y("voice_profile", k, default=d)  # noqa: E731
    min_dur = float(c("min_candidate_sec", 1.5))
    min_conf = float(c("min_candidate_conf", 0.8))
    min_snr = float(c("min_snr_db", 15))
    target = float(c("target_ref_sec", 25))

    floor = quality.noise_floor(y16, EMB_SR)
    bad_events = {"laughter", "crying", "screaming", "baby cry, infant cry"}

    strict, relaxed = [], []
    for seg in segments:
        dur = float(seg["end"]) - float(seg["start"])
        if dur < min_dur:
            continue
        a16, b16 = int(seg["start"] * EMB_SR), int(seg["end"] * EMB_SR)
        piece16 = y16[a16:b16]
        if not len(piece16):
            continue

        snr = quality.snr_db(piece16, floor)
        clip = quality.clipping_ratio(piece16)
        emotion = str(seg.get("emotion", "neutral")).lower()
        events = {str(e).lower() for e in (seg.get("events") or [])}
        conf = float(seg.get("speaker_confidence", 1.0))

        record = {"seg": seg, "dur": dur, "snr": snr,
                  "a": int(seg["start"] * REF_SR), "b": int(seg["end"] * REF_SR)}

        if (not seg.get("overlap") and conf >= min_conf and snr >= min_snr
                and clip < 0.01 and emotion in ("neutral", "calm")
                and not (events & bad_events)):
            strict.append(record)
        elif not seg.get("overlap") and clip < 0.05:
            # запасной пул: если строгих кандидатов не набралось на 8 секунд,
            # лучше собрать референс из менее идеальной речи, чем отдать
            # спикеру чужой пресет
            relaxed.append(record)

    pool = strict if _total(strict) >= float(c("min_ref_sec_for_clone", 8)) else strict + relaxed
    if not pool:
        return {"clean_sec": 0.0, "snr_db": 0.0, "score": 0.0,
                "clone_allowed": False, "best_samples": [], "audio": None}

    # близость к центроиду спикера: кусок, непохожий на самого спикера,
    # в референс не годится, даже если он чистый и длинный
    sims = np.ones(len(pool), dtype=np.float32)
    if centroid is not None and embedder is not None:
        spans = [(r["seg"]["start"], r["seg"]["end"]) for r in pool]
        vecs = embedder.embed_windows(y16, spans, sr=EMB_SR)
        sims = np.clip(vecs @ np.asarray(centroid, dtype=np.float32).ravel(), 0, 1)

    for r, sim in zip(pool, sims):
        r["sim"] = float(sim)
        r["score"] = quality.score_candidate(r["dur"], r["snr"], float(sim))
    pool.sort(key=lambda r: -r["score"])

    chunks, chosen, total = [], [], 0.0
    gap = np.zeros(int(float(c("join_silence_sec", 0.2)) * REF_SR), dtype=np.float32)
    for r in pool:
        piece = y_ref[r["a"]:r["b"]]
        if not len(piece):
            continue
        chunks.append(piece)
        chunks.append(gap)
        chosen.append(r)
        total += r["dur"]
        if total >= target:
            break

    audio = np.concatenate(chunks).astype(np.float32) if chunks else None
    if audio is not None:
        audio = quality.trim_silence(audio, REF_SR)
        audio = quality.loudness_normalize(audio, REF_SR,
                                           float(c("loudness_lufs", -23)))
    clean_sec = len(audio) / REF_SR if audio is not None else 0.0

    return {
        "clean_sec": round(clean_sec, 2),
        "snr_db": round(float(np.mean([r["snr"] for r in chosen])), 1) if chosen else 0.0,
        "score": round(float(np.mean([r["score"] for r in chosen])), 3) if chosen else 0.0,
        "clone_allowed": clean_sec >= float(c("min_ref_sec_for_clone", 8)),
        "best_samples": [int(r["seg"]["id"]) for r in chosen[:3]],
        "strict_sec": round(_total(strict), 2),
        "audio": audio,
    }


def _total(records: list[dict]) -> float:
    return float(sum(r["dur"] for r in records))


# ---------------------------------------------------------------- профиль

def build_profile(speaker_id: str, ref_path: Path, engine, embedder,
                  mode: str = "clone") -> VoiceProfile:
    """Считает латенты XTTS и эталон ECAPA. ЕДИНСТВЕННОЕ место вызова
    get_conditioning_latents во всём пайплайне (кроме сборки банка)."""
    gpt_latent, speaker_emb = engine.build_conditioning(ref_path)
    identity = embedder.embed_file(ref_path)
    return VoiceProfile(
        speaker_id=speaker_id,
        mode=mode,
        gpt_cond_latent=gpt_latent,
        speaker_embedding=speaker_emb,
        identity=identity,
        ref_path=str(ref_path),
        engine=engine.name,
        ref_sha256=_sha256(ref_path),
    )


def save_profile(profile: VoiceProfile, spk_dir: Path) -> dict:
    """Кладёт профиль на диск. Отпечаток источника сохраняется вместе с ним."""
    import torch

    spk_dir.mkdir(parents=True, exist_ok=True)
    profile_path = spk_dir / "voice_profile.pt"
    identity_path = spk_dir / "identity_embedding.npy"

    torch.save({
        "format": PROFILE_FORMAT,
        "speaker_id": profile.speaker_id,
        "mode": profile.mode,
        "engine": profile.engine,
        "gpt_cond_latent": profile.gpt_cond_latent,
        "speaker_embedding": profile.speaker_embedding,
        "ref_path": profile.ref_path,
        "ref_sha256": profile.ref_sha256,
        "preset_id": profile.preset_id,
        "meta": profile.meta,
    }, str(profile_path))
    np.save(str(identity_path), np.asarray(profile.identity, dtype=np.float32))
    return {"profile_path": str(profile_path), "identity_path": str(identity_path)}


def load_profile(profile_path: str | Path,
                 identity_path: str | Path | None = None,
                 verify_ref: bool = True) -> VoiceProfile:
    """Читает профиль. Не совпал отпечаток референса — сообщаем, а не молчим.

    Урок прошлых разборов: кэш производного результата обязан проверять,
    тем ли входом он получен. Иначе после смены модели или пересборки
    референса используются чужие латенты, и это никак не заметно.
    """
    import torch

    profile_path = Path(profile_path)
    blob = torch.load(str(profile_path), map_location="cpu", weights_only=False)

    identity = None
    ipath = Path(identity_path) if identity_path else profile_path.with_name(
        "identity_embedding.npy")
    if ipath.exists():
        identity = np.load(str(ipath)).astype(np.float32)

    prof = VoiceProfile(
        speaker_id=blob.get("speaker_id", "?"),
        mode=blob.get("mode", "clone"),
        gpt_cond_latent=blob.get("gpt_cond_latent"),
        speaker_embedding=blob.get("speaker_embedding"),
        identity=identity,
        ref_path=blob.get("ref_path"),
        preset_id=blob.get("preset_id"),
        engine=blob.get("engine", "xtts_v2"),
        ref_sha256=blob.get("ref_sha256", ""),
        meta=blob.get("meta", {}) or {},
    )

    if verify_ref and prof.ref_path and prof.ref_sha256:
        ref = Path(prof.ref_path)
        if ref.exists() and _sha256(ref) != prof.ref_sha256:
            raise StaleProfile(
                f"Профиль {prof.speaker_id} собран с другого референса — "
                "нужно пересобрать")
    return prof


class StaleProfile(RuntimeError):
    """Профиль не соответствует своему референсу."""


# ---------------------------------------------------------------- сборка всех

def build_all(job_dir: str | Path, vocals_wav: str | Path,
              vocals16_wav: str | Path, segments: list[dict],
              speaker_summary: dict, cfg, engine, embedder,
              voice_mode: str = "auto", progress=None) -> dict:
    """Строит профили всех спикеров задачи и сохраняет speakers.json.

    Возвращает {speaker_id: dict} в формате, совместимом со старым
    speakers.json, плюс новые поля reference/voice.
    """
    import librosa
    import soundfile as sf

    from core.gender import detect_gender

    job_dir = Path(job_dir)
    speakers_dir = job_dir / "speakers"
    speakers_dir.mkdir(parents=True, exist_ok=True)

    y_ref = librosa.load(str(vocals_wav), sr=REF_SR, mono=True)[0].astype(np.float32)
    y16 = librosa.load(str(vocals16_wav), sr=EMB_SR, mono=True)[0].astype(np.float32)

    by_speaker: dict[str, list[dict]] = {}
    for seg in segments:
        by_speaker.setdefault(seg["speaker"], []).append(seg)

    ids = sorted(by_speaker, key=lambda s: speaker_summary.get(s, {}).get(
        "first_sec", float("inf")))
    profiles: dict[str, dict] = {}

    for idx, sid in enumerate(ids):
        segs = by_speaker[sid]
        spk_dir = speakers_dir / sid
        spk_dir.mkdir(parents=True, exist_ok=True)
        centroid = speaker_summary.get(sid, {}).get("centroid")

        ref = select_reference(segs, y_ref, y16, centroid, embedder, cfg)
        ref_path = spk_dir / "ref_main.wav"
        if ref["audio"] is not None and len(ref["audio"]):
            sf.write(str(ref_path), ref["audio"], REF_SR)
        else:
            ref_path = None

        # пол определяется ОДИН раз на спикера по его референсу,
        # а не голосованием по сегментам: это свойство человека, а не реплики
        gender_src = segs
        g = detect_gender(y16, EMB_SR, gender_src, cfg)

        clone_ok = bool(ref["clone_allowed"] and ref_path is not None)
        mode = "clone" if (voice_mode == "clone" or (voice_mode == "auto" and clone_ok)) else "preset"
        if mode == "clone" and not clone_ok:
            log.warning("Спикер %s: чистой речи всего %.1f с — клон невозможен, "
                        "будет голос из банка", sid, ref["clean_sec"])
            mode = "preset"

        entry = {
            "id": sid,
            "label": f"Спикер {sid[1:]}",
            "gender": g["gender"],
            "gender_confidence": g["confidence"],
            "f0_median": g["f0_median"],
            "age": g.get("age", "adult"),
            "segments": len(segs),
            "speech_total_s": round(sum(s["end"] - s["start"] for s in segs), 2),
            "first_sec": speaker_summary.get(sid, {}).get("first_sec", 0.0),
            "spread": speaker_summary.get(sid, {}).get("spread", 0.0),
            "merged_from": speaker_summary.get(sid, {}).get("merged_from", []),
            "reference": {
                "path": str(ref_path) if ref_path else None,
                "clean_sec": ref["clean_sec"],
                "snr_db": ref["snr_db"],
                "score": ref["score"],
                "clone_allowed": clone_ok,
                "best_samples": ref["best_samples"],
            },
            "voice": {
                "mode": mode,
                "preset_id": None,
                "profile_path": None,
                "identity_path": None,
                "locked": False,
                "casting_candidates": [],
                "edited_by_user": False,
            },
            # совместимость со старым форматом speakers.json
            "ref_main": str(ref_path) if ref_path else None,
            "ref_total_s": ref["clean_sec"],
            "ref_ok": clone_ok,
        }

        if mode == "clone" and ref_path is not None:
            profile = build_profile(sid, ref_path, engine, embedder, mode="clone")
            paths = save_profile(profile, spk_dir)
            entry["voice"].update(paths, locked=True)
            log.info("Спикер %s: профиль голоса зафиксирован (%.1f с чистой речи, "
                     "SNR %.0f дБ, %s)", sid, ref["clean_sec"], ref["snr_db"],
                     g["gender"])
        else:
            log.info("Спикер %s: голос будет назначен из банка (%.1f с чистой речи)",
                     sid, ref["clean_sec"])

        profiles[sid] = entry
        if progress:
            progress(int(100 * (idx + 1) / max(1, len(ids))))

    (job_dir / "speakers.json").write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    return profiles
