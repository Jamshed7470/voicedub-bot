"""A/B-стенд озвучки: один фрагмент ролика в нескольких вариантах синтеза.

Нужен, чтобы услышать разницу до многочасового прогона всего ролика.
Берёт готовый разбор задачи (transcript/translated/speakers) и озвучивает
выбранное окно времени разными способами, собирая по видеофайлу на вариант.

    python -m scripts.ab_test --job 88dd033fbc05 --start 6376 --dur 120

Варианты:
  current  — как сейчас в пайплайне: ускорение темпа до ×1.5 ради тайминга;
  natural  — без ускорения вообще (эталон естественности, слот не соблюдается);
  tuned    — XTTS с поднятой живостью генерации, без ускорения;
  silero   — Silero v4 (дикторский русский, без клонирования голоса).

Запускать из папки voicedub.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import JOBS_DIR, load_config  # noqa: E402
from core.media import run as ffrun  # noqa: E402

log = logging.getLogger("ab_test")

SR_OUT = 44100
SILERO_SPEAKERS = {"male": "eugene", "female": "kseniya"}


# ---------------------------------------------------------------------------
# подготовка данных
# ---------------------------------------------------------------------------

def load_job(job_id: str, cfg) -> tuple[Path, list[dict], dict]:
    job_dir = JOBS_DIR / job_id
    with open(job_dir / "translated.json", encoding="utf-8") as f:
        segments = json.load(f)["segments"]
    with open(job_dir / "speakers.json", encoding="utf-8") as f:
        profiles = json.load(f)

    from core import voicebank
    assigned = voicebank.assign(profiles)
    for spk, voice in (assigned or {}).items():
        profiles[spk]["ref_main"] = voice["path"]
        profiles[spk]["bank_voice"] = voice["name"]
        profiles[spk]["ref_ok"] = True
    return job_dir, segments, profiles


def window(segments: list[dict], start: float, dur: float) -> list[dict]:
    end = start + dur
    return [s for s in segments if s["start"] >= start and s["end"] <= end]


# ---------------------------------------------------------------------------
# синтез вариантов
# ---------------------------------------------------------------------------

def synth_xtts(segs: list[dict], profiles: dict, limits: dict, out_dir: Path,
               cfg, accelerate: bool, lively: bool) -> list[dict]:
    """Озвучка XTTS-v2. accelerate=True воспроизводит нынешнее поведение."""
    from core.normalize import normalize_for_tts
    from core.timing import STATUS_TOO_LONG, fit_to_slot
    from core.tts import synthesize

    if lively:
        _make_lively(cfg)

    atempo_max = cfg.atempo_max
    hard = float(cfg.y("timing", "atempo_hard_max", default=1.5))
    placed = []
    for i, seg in enumerate(segs, 1):
        raw = out_dir / f"seg_{seg['id']}_raw.wav"
        fitted = out_dir / f"seg_{seg['id']}.wav"
        text = normalize_for_tts(seg["text"], "ru")
        speaker_wav = profiles.get(seg["speaker"], {}).get("ref_main")
        try:
            synthesize(text, "ru", raw, cfg, speaker_wav=speaker_wav, speed=1.0)
            if not accelerate:
                raw.replace(fitted)
            else:
                slot = max(0.4, limits.get(seg["id"], seg["end"]) - seg["start"])
                fit = fit_to_slot(raw, fitted, slot, atempo_max)
                if fit.status == STATUS_TOO_LONG:
                    tempo = min(fit.tempo, hard)
                    ffrun(["ffmpeg", "-y", "-i", str(raw), "-filter:a",
                           f"atempo={tempo:.6f}", "-c:a", "pcm_s16le",
                           str(fitted)], desc="ab atempo")
            placed.append({"start": seg["start"], "path": str(fitted)})
        except Exception:  # noqa: BLE001 — одна реплика не должна ронять стенд
            log.exception("A/B: реплика %s не синтезировалась", seg["id"])
        finally:
            raw.unlink(missing_ok=True)
        log.info("  %d/%d", i, len(segs))
    return placed


def _make_lively(cfg) -> None:
    """Поднимает живость генерации XTTS: ровный монотон — половина «робота»."""
    from core.tts import _get_tts

    conf = _get_tts(cfg).synthesizer.tts_config
    conf.temperature = 0.85          # было 0.75 — шире интонационный разброс
    conf.repetition_penalty = 2.5    # было 5.0 — меньше «сглаживания» просодии
    conf.top_k = 60
    conf.top_p = 0.9
    conf.length_penalty = 0.9
    log.info("XTTS: живость поднята (temperature %.2f, rep_penalty %.1f)",
             conf.temperature, conf.repetition_penalty)


def synth_silero(segs: list[dict], profiles: dict, out_dir: Path) -> list[dict]:
    """Silero v4: дикторский русский без клонирования (голос по полу спикера)."""
    import torch

    # torch.hub тянет репозиторий с github, а он отсюда отваливается по DNS:
    # файл модели берём напрямую с серверов Silero (40 МБ, кладём рядом с данными)
    model_path = Path("data") / "models" / "silero_v4_ru.pt"
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Silero: скачиваю модель (40 МБ)…")
        torch.hub.download_url_to_file(
            "https://models.silero.ai/models/tts/ru/v4_ru.pt", str(model_path))
    model = torch.package.PackageImporter(str(model_path)).load_pickle(
        "tts_models", "model")
    model.to(torch.device("cpu"))

    from core.normalize import normalize_for_tts

    placed = []
    for i, seg in enumerate(segs, 1):
        gender = profiles.get(seg["speaker"], {}).get("gender", "male")
        speaker = SILERO_SPEAKERS.get(gender, "eugene")
        path = out_dir / f"seg_{seg['id']}.wav"
        text = normalize_for_tts(seg["text"], "ru")
        try:
            model.save_wav(text=text, speaker=speaker, sample_rate=48000,
                           audio_path=str(path))
            placed.append({"start": seg["start"], "path": str(path)})
        except Exception:  # noqa: BLE001
            log.exception("Silero: реплика %s не синтезировалась", seg["id"])
        log.info("  %d/%d", i, len(segs))
    return placed


# ---------------------------------------------------------------------------
# сборка фрагмента
# ---------------------------------------------------------------------------

def build_fragment(job_dir: Path, placed: list[dict], start: float, dur: float,
                   out_mp4: Path, tmp: Path) -> None:
    """Микс: фон ролика + озвучка, поверх — видеоряд того же куска."""
    import numpy as np
    import soundfile as sf

    bg = tmp / "bg.wav"
    ffrun(["ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
           "-i", str(job_dir / "background.wav"), "-ar", str(SR_OUT),
           "-ac", "2", "-c:a", "pcm_s16le", str(bg)], desc="ab bg")
    voc = tmp / "voc_ref.wav"
    ffrun(["ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
           "-i", str(job_dir / "vocals.wav"), "-ar", str(SR_OUT),
           "-ac", "1", "-c:a", "pcm_s16le", str(voc)], desc="ab voc")

    mix, _ = sf.read(str(bg), dtype="float32", always_2d=True)
    ref, _ = sf.read(str(voc), dtype="float32")
    ref_rms = float(np.sqrt(np.mean(ref ** 2))) or 0.05

    for item in placed:
        path = Path(item["path"])
        if not path.exists():
            continue
        res = tmp / f"r_{path.stem}.wav"
        ffrun(["ffmpeg", "-y", "-i", str(path), "-ar", str(SR_OUT), "-ac", "2",
               "-c:a", "pcm_s16le", str(res)], desc="ab resample")
        data, _ = sf.read(str(res), dtype="float32", always_2d=True)
        res.unlink(missing_ok=True)
        rms = float(np.sqrt(np.mean(data ** 2)))
        if rms > 1e-6:
            data = data * min(4.0, ref_rms / rms)
        pos = int(max(0.0, item["start"] - start) * SR_OUT)
        end = min(len(mix), pos + len(data))
        if end > pos:
            mix[pos:end] += data[:end - pos]

    peak = float(np.max(np.abs(mix))) or 1.0
    if peak > 0.99:
        mix = mix * (0.99 / peak)
    dubbed = tmp / "dubbed.wav"
    sf.write(str(dubbed), mix, SR_OUT)

    ffrun(["ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
           "-i", str(job_dir / "input.mp4"), "-i", str(dubbed),
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset",
           "veryfast", "-crf", "23", "-c:a", "aac", "-b:a", "192k",
           "-shortest", str(out_mp4)], desc="ab mux")


def main() -> None:
    p = argparse.ArgumentParser(description="A/B-стенд вариантов озвучки")
    p.add_argument("--job", required=True, help="id задачи в data/jobs")
    p.add_argument("--start", type=float, required=True, help="начало окна, с")
    p.add_argument("--dur", type=float, default=120.0, help="длина окна, с")
    p.add_argument("--variants", default="current,natural,tuned,silero")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    cfg = load_config()
    job_dir, segments, profiles = load_job(args.job, cfg)
    from core.pipeline import _slot_limits
    limits = _slot_limits(segments, cfg)
    segs = window(segments, args.start, args.dur)
    log.info("Окно %.0f–%.0f с: %d реплик, %d спикеров", args.start,
             args.start + args.dur, len(segs),
             len({s["speaker"] for s in segs}))

    # окно в имени папки: сравнения разных участков не затирают друг друга
    out_root = cfg.output_dir / "ab_test" / f"{int(args.start)}s"
    out_root.mkdir(parents=True, exist_ok=True)
    tmp = job_dir / "ab_tmp"
    tmp.mkdir(exist_ok=True)

    # оригинал для сравнения
    orig = out_root / "0_оригинал.mp4"
    if not orig.exists():
        ffrun(["ffmpeg", "-y", "-ss", f"{args.start:.3f}", "-t", f"{args.dur:.3f}",
               "-i", str(job_dir / "input.mp4"), "-c:v", "libx264", "-preset",
               "veryfast", "-crf", "23", "-c:a", "aac", "-b:a", "192k",
               str(orig)], desc="ab original")
        log.info("Готов: %s", orig.name)

    plan = {
        "current": lambda d: synth_xtts(segs, profiles, limits, d, cfg,
                                        accelerate=True, lively=False),
        "natural": lambda d: synth_xtts(segs, profiles, limits, d, cfg,
                                        accelerate=False, lively=False),
        "tuned": lambda d: synth_xtts(segs, profiles, limits, d, cfg,
                                      accelerate=False, lively=True),
        "silero": lambda d: synth_silero(segs, profiles, d),
    }
    names = {"current": "1_как_сейчас", "natural": "2_без_ускорения",
             "tuned": "3_xtts_живее", "silero": "4_silero"}

    for variant in args.variants.split(","):
        variant = variant.strip()
        if variant not in plan:
            log.warning("Неизвестный вариант: %s", variant)
            continue
        log.info("=== Вариант %s ===", variant)
        seg_dir = tmp / variant
        seg_dir.mkdir(exist_ok=True)
        try:
            placed = plan[variant](seg_dir)
            if not placed:
                log.error("Вариант %s: ни одной реплики — пропускаю", variant)
                continue
            out = out_root / f"{names[variant]}.mp4"
            build_fragment(job_dir, placed, args.start, args.dur, out, tmp)
            log.info("Готов: %s", out.name)
        except Exception:  # noqa: BLE001 — один вариант не должен ронять стенд
            log.exception("Вариант %s не собрался", variant)

    log.info("Файлы для прослушивания: %s", out_root)


if __name__ == "__main__":
    main()
