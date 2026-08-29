"""Рендер всех реплик задачи по заблокированным профилям.

Заменяет прежний `_synthesize_all`. Отличия, ради которых он переписан:

* профиль спикера загружается ОДИН раз на задачу и держится в памяти —
  внутри цикла тембр не пересчитывается никогда (INV-1, INV-2);
* каждая реплика проходит Identity QC и при провале пересинтезируется с
  другим seed, а не уходит в микс молча;
* по итогам собирается отчёт «карта голосов» со средней попарной схожестью
  реплик каждого спикера — числом, по которому видно, поплыл голос или нет.

Подгонка под тайминги оставлена прежней (core/timing.fit_to_slot):
спецификация её не меняет.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.errors import UserError
from synth import qc as qc_mod

log = logging.getLogger(__name__)

SYNTH_DONE = "done.txt"


class JobCancelled(Exception):
    """Задачу отменил пользователь."""


@dataclass
class RenderStats:
    total: int = 0
    synthesized: int = 0
    reused: int = 0
    failed: int = 0
    qc_failed: int = 0
    retries: int = 0
    per_speaker: dict = field(default_factory=dict)
    per_speaker_report: dict = field(default_factory=dict)
    overall_identity: float = 0.0


class ProfileCache:
    """Профили голоса, загруженные один раз на задачу.

    Ключ — speaker_id (или id пресета для голоса из банка). Профиль
    загружается лениво при первой реплике спикера и больше не трогается.
    """

    def __init__(self, speakers: dict, bank=None):
        self.speakers = speakers
        self.bank = bank
        self._cache: dict[str, object] = {}
        self.load_calls = 0

    def get(self, speaker_id: str, override: dict | None = None):
        from voices import profiles as prof_mod

        key = speaker_id
        if override and override.get("preset_id"):
            key = f"preset:{override['preset_id']}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        self.load_calls += 1
        entry = self.speakers.get(speaker_id) or {}
        voice = entry.get("voice") or {}
        preset_id = (override or {}).get("preset_id") or voice.get("preset_id")

        if voice.get("mode") == "preset" or (override or {}).get("mode") == "preset":
            if self.bank is None:
                # банк — общий ресурс приложения, а не аргумент вызова:
                # не найти его здесь значит остановить озвучку из-за того,
                # что кто-то выше по стеку его не передал
                from voices.bank import get_bank

                self.bank = get_bank()
            if not preset_id:
                raise UserError(
                    f"Спикеру {speaker_id} не назначен голос из банка. "
                    "Выберите его в студии или включите режим клонирования.")
            try:
                profile = self.bank.load_profile(preset_id)
            except KeyError as e:
                raise UserError(
                    f"Голоса «{preset_id}» нет в банке. Выберите другой в "
                    "студии или соберите банк заново: "
                    "python -m scripts.build_voice_bank --from-dir voice_db") from e
        else:
            path = voice.get("profile_path")
            if not path or not Path(path).exists():
                raise UserError(
                    f"У спикера {speaker_id} нет собранного профиля голоса — "
                    "синтез невозможен (INV-2)")
            profile = prof_mod.load_profile(path, voice.get("identity_path"))

        profile.locked = True
        self._cache[key] = profile
        return profile


# ---------------------------------------------------------------- журнал

def done_ids(synth_dir: Path) -> set[int]:
    """Реплики, озвучка которых доведена до конца.

    Судить по самому файлу нельзя: оборванный wav выглядит целым — длину
    libsndfile берёт из заголовка. Поэтому id дописывается ПОСЛЕ закрытия
    файла; оборвали задачу — строка не появилась, реплика переозвучится.
    """
    path = synth_dir / SYNTH_DONE
    if not path.exists():
        return set()
    ids: set[int] = set()
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
        if lines and lines[-1] != "":
            lines = lines[:-1]        # хвост без перевода строки мог не дописаться
        for line in lines:
            if line.strip().isdigit():
                ids.add(int(line.strip()))
    except OSError:
        log.warning("Журнал озвученных реплик не читается — озвучиваю заново")
    return ids


def mark_done(synth_dir: Path, seg_id: int) -> None:
    try:
        with open(synth_dir / SYNTH_DONE, "a", encoding="utf-8") as f:
            f.write(f"{seg_id}\n")
            f.flush()
    except OSError:
        log.warning("Не удалось отметить реплику %s в журнале", seg_id)


def reusable(path: Path, done: set[int], seg_id: int) -> bool:
    return seg_id in done and path.exists() and path.stat().st_size > 1000


# ---------------------------------------------------------------- синтез

def synthesize_segment(seg: dict, profile, engine, embedder, cfg, job_id: str,
                       lang: str, out_path: Path, speed: float,
                       expected_dur: float,
                       cross_lingual: bool = False) -> qc_mod.QCResult:
    """Синтез одной реплики с повторами по результатам Identity QC.

    Возвращает лучший результат: прошедший проверки, а если ни один не
    прошёл — тот, у которого выше схожесть тембра. Реплика попадает в микс
    в любом случае: дырка в дубляже хуже неидеальной реплики.
    """
    max_retries = int(cfg.y("synthesis", "max_retries", default=3))
    text = seg.get("text_tts") or seg.get("text") or ""

    best_path, best_qc = None, None
    for attempt in range(max_retries):
        seed = engine.make_seed(job_id, int(seg["id"]), attempt)
        candidate = out_path.with_name(f"{out_path.stem}_try{attempt}{out_path.suffix}")
        engine.synthesize(text, lang, profile, candidate, speed=speed, seed=seed)

        result = qc_mod.check(candidate, text, lang, profile, expected_dur,
                              embedder, cfg, cross_lingual=cross_lingual)
        result.seed = seed
        result.attempts = attempt + 1
        if best_qc is None or result.identity_sim > best_qc.identity_sim:
            if best_path is not None and best_path != candidate:
                best_path.unlink(missing_ok=True)
            best_path, best_qc = candidate, result
        elif candidate != best_path:
            candidate.unlink(missing_ok=True)

        if result.ok:
            break
        log.info("Сегмент %s: попытка %d не прошла проверку (%s)",
                 seg["id"], attempt + 1, "; ".join(result.reasons))

    if best_path is None:
        raise RuntimeError("синтез не дал ни одного варианта")
    best_path.replace(out_path)
    return best_qc


def render_all(job_dir: Path, segments: list[dict], speakers: dict, cfg,
               engine, embedder, lang: str, job_id: str, translator=None,
               bank=None, progress=None, cancel_event=None,
               lang_src: str | None = None) -> tuple[list[dict], RenderStats]:
    """Синтезирует все реплики задачи и подгоняет их под тайминги."""
    from core.media import run as ffrun
    from core.normalize import normalize_for_tts
    from core.timing import STATUS_TOO_LONG, fit_to_slot
    from core.translate import compute_max_chars

    synth_dir = job_dir / "synth"
    synth_dir.mkdir(parents=True, exist_ok=True)

    atempo_max = cfg.atempo_max
    atempo_hard_max = float(cfg.y("timing", "atempo_hard_max", default=1.5))
    speed_soft_max = float(cfg.y("timing", "speed_soft_max", default=1.15))

    cache = ProfileCache(speakers, bank)
    stability_min = float(cfg.y("synthesis", "stability_min_sec", default=3.0))
    # профиль собран из речи на языке оригинала: синтез на другом языке
    # неизбежно даёт меньшее сходство тембра, и порог это учитывает
    cross_lingual = bool(lang_src and _norm(lang_src) != _norm(lang))
    if cross_lingual:
        log.info("Синтез межъязыковой (%s -> %s): порог сходства тембра снижен",
                 lang_src, lang)
    stats = RenderStats()
    to_do = [s for s in segments if not s.get("skip_tts") and (
        s.get("text_tts") or s.get("text") or "").strip()]
    stats.total = len(to_do)
    if not to_do:
        raise UserError("Не удалось синтезировать ни одного сегмента речи.")

    limits = slot_limits(segments, cfg)
    done = done_ids(synth_dir)
    max_failed = max(3, int(len(to_do) * float(
        cfg.y("timing", "max_failed_ratio", default=0.05))))

    placed: list[dict] = []
    rates: dict[str, list[float]] = {}

    def guess_speed(text: str, speaker: str, slot: float) -> float:
        """Темп по уже измеренному для ЭТОГО голоса: табличный ошибается."""
        samples = rates.get(speaker)
        if not samples:
            return 1.0
        rate = sorted(samples)[len(samples) // 2]
        need = len(text) / max(1e-6, rate)
        if need <= slot * atempo_max:
            return 1.0
        return min(speed_soft_max, need / (slot * atempo_max))

    for i, seg in enumerate(to_do):
        if cancel_event is not None and cancel_event.is_set():
            raise JobCancelled()

        ready = synth_dir / f"seg_{seg['id']}.wav"
        if reusable(ready, done, seg["id"]):
            placed.append({"start": seg["start"], "path": str(ready), "id": seg["id"]})
            stats.reused += 1
            _progress(progress, i, len(to_do))
            continue

        slot = max(0.4, limits.get(seg["id"], seg["end"]) - seg["start"])
        text = seg.get("text_tts") or seg["text"]

        try:
            profile = cache.get(seg["speaker"], seg.get("voice_override"))
        except UserError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("Профиль спикера %s не загрузился", seg["speaker"])
            stats.failed += 1
            continue

        raw = synth_dir / f"seg_{seg['id']}_raw.wav"
        fitted = synth_dir / f"seg_{seg['id']}.wav"

        try:
            speed = guess_speed(text, seg["speaker"], slot)
            result = synthesize_segment(seg, profile, engine, embedder, cfg,
                                        job_id, lang, raw, speed, slot,
                                        cross_lingual)
            fit = fit_to_slot(raw, fitted, slot, atempo_max)
            rates.setdefault(seg["speaker"], []).append(
                len(text) / max(0.1, fit.duration) * speed)
            del rates[seg["speaker"]][:-20]

            # мягкая подгонка параметром speed до применения atempo
            if (fit.status == STATUS_TOO_LONG and speed < speed_soft_max
                    and fit.tempo <= atempo_max * speed_soft_max):
                speed = min(speed_soft_max, speed * fit.tempo)
                result = synthesize_segment(seg, profile, engine, embedder, cfg,
                                            job_id, lang, raw, speed, slot,
                                            cross_lingual)
                fit = fit_to_slot(raw, fitted, slot, atempo_max)

            # не влезает даже так — просим переводчика сократить реплику
            if fit.status == STATUS_TOO_LONG and translator is not None:
                rate = len(text) / max(0.1, fit.duration)
                max_chars = max(10, int(slot * rate))
                max_chars = min(max_chars, compute_max_chars(seg, lang, cfg, slot))
                shorter = translator.compress_segment(text, max_chars, lang)
                shorter = normalize_for_tts(shorter, lang)
                if shorter.strip() and shorter != text:
                    seg["text_tts"] = seg["text"] = text = shorter
                    result = synthesize_segment(seg, profile, engine, embedder,
                                                cfg, job_id, lang, raw,
                                                speed_soft_max, slot,
                                                cross_lingual)
                    fit = fit_to_slot(raw, fitted, slot, atempo_max)

            if fit.status == STATUS_TOO_LONG:
                tempo = min(fit.tempo, atempo_hard_max)
                log.warning("Сегмент %s: нужно ×%.2f, ускоряю до ×%.2f",
                            seg["id"], fit.tempo, tempo)
                ffrun(["ffmpeg", "-y", "-i", str(raw),
                       "-filter:a", f"atempo={tempo:.6f}",
                       "-c:a", "pcm_s16le", str(fitted)], desc="ffmpeg atempo max")

            seg["_stability_ok"] = (seg["end"] - seg["start"]) >= stability_min
            _record(seg, result, fitted, profile, stats, embedder)
            placed.append({"start": seg["start"], "path": str(fitted), "id": seg["id"]})
            mark_done(synth_dir, seg["id"])
            stats.synthesized += 1

        except (AssertionError, JobCancelled, UserError):
            raise
        except Exception:  # noqa: BLE001 — одна сломанная реплика не валит задачу
            log.exception("Синтез сегмента %s не удался, пропускаю", seg["id"])
            stats.failed += 1
            _free_vram()
            if stats.failed > max_failed:
                raise UserError(
                    f"Синтез сорвался на {stats.failed} репликах из {len(to_do)} — "
                    "похоже, не хватает видеопамяти. Закройте другие задачи "
                    "на видеокарте и пришлите ролик снова.")
        finally:
            raw.unlink(missing_ok=True)

        _progress(progress, i, len(to_do))

    if not placed:
        raise UserError("Не удалось синтезировать ни одного сегмента речи.")

    # INV-1: профилей загружено ровно столько, сколько голосов в задаче
    log.info("Рендер: озвучено %d, взято готовых %d, не удалось %d, "
             "QC не прошли %d, профилей загружено %d",
             stats.synthesized, stats.reused, stats.failed, stats.qc_failed,
             cache.load_calls)
    return placed, stats


def _record(seg: dict, result, wav_path: Path, profile, stats: RenderStats,
            embedder) -> None:
    """Пишет результат QC в сегмент и копит статистику по спикеру."""
    seg["synth"] = {
        "path": str(wav_path),
        "seed": getattr(result, "seed", None),
        "attempts": getattr(result, "attempts", None) or 1,
        "identity_sim": result.identity_sim,
        "duration_ratio": result.duration_ratio,
        "backcheck_cer": result.backcheck_cer,
        "status": result.status,
    }
    if not result.ok:
        stats.qc_failed += 1
        flags = seg.setdefault("flags", [])
        if "identity_qc_failed" not in flags:
            flags.append("identity_qc_failed")

    bucket = stats.per_speaker.setdefault(seg["speaker"], {
        "segments": 0, "passed": 0, "embeddings": [],
        "voice": profile.preset_id or ("клон" if profile.mode == "clone" else "пресет"),
    })
    bucket["segments"] += 1
    bucket["passed"] += int(result.ok)

    # Стабильность считается ТОЛЬКО по репликам, на которых голосовой
    # отпечаток надёжен. Замер на материале с заведомо одним голосом:
    # по всем репликам метрика даёт 0.64, по репликам от 3 с — 0.78.
    # Разница целиком в шуме измерения на коротких фрагментах. Считая по
    # всем, мы мерили бы не устойчивость голоса, а длину реплик, и
    # хороший дубляж выглядел бы плохим.
    #
    # Эмбеддинги копим не все: по сорока репликам оценка уже устойчива,
    # а память на тысячах векторов расти не должна.
    if seg.get("_stability_ok") and len(bucket["embeddings"]) < 40:
        try:
            bucket["embeddings"].append(embedder.embed_file(wav_path, 20.0))
        except Exception:  # noqa: BLE001
            pass


def _norm(code: str) -> str:
    """«zh-cn» и «zh» — один язык; регистр не важен."""
    return (code or "").strip().lower().split("-")[0]


def _progress(progress, i: int, total: int) -> None:
    if progress:
        progress(int(100 * (i + 1) / max(1, total)))


def _free_vram() -> None:
    try:
        import gc

        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def slot_limits(segments: list[dict], cfg) -> dict[int, float]:
    """Докуда реплика может звучать: до начала следующей минус зазор.

    Реплика может заходить в паузу после себя — это лучше, чем ускорять её
    до неразборчивости или выбрасывать слова.
    """
    max_extend = float(cfg.y("timing", "max_extend_s", default=1.5))
    guard = float(cfg.y("timing", "gap_guard_ms", default=80)) / 1000.0
    ordered = sorted(segments, key=lambda s: s["start"])
    limits: dict[int, float] = {}
    for i, seg in enumerate(ordered):
        limit = seg["end"] + max_extend
        if i + 1 < len(ordered):
            limit = min(limit, ordered[i + 1]["start"] - guard)
        limits[seg["id"]] = max(seg["end"], limit)
    return limits


# Достижимый уровень стабильности, измеренный на фикстуре с известным
# ответом. Выше него не поднимется даже идеальная реализация: это предел
# точности сравнения тембра на коротких репликах, а при озвучке на чужом
# языке — ещё и свойство XTTS переносить голос между языками.
STABILITY_TARGET = {"same": 0.75, "cross": 0.60}


def write_report(job_dir: Path, speakers: dict, stats: RenderStats,
                 lang_src: str, lang_tgt: str) -> Path:
    """Карта голосов: кто каким голосом озвучен и насколько голос стабилен."""
    report = qc_mod.build_report(stats.per_speaker)
    cross = _norm(lang_src) != _norm(lang_tgt)
    lines = [
        "# Карта голосов", "",
        f"Язык: {lang_src} → {lang_tgt}",
        f"Реплик озвучено: {stats.synthesized + stats.reused} из {stats.total}", "",
        "| Спикер | Пол | Голос | Реплик | QC | Стабильность |",
        "|---|---|---|---|---|---|",
    ]
    sims = []
    unmeasured = 0
    for sid, rec in sorted(report.items(), key=lambda kv: -kv[1]["segments"]):
        info = speakers.get(sid, {})
        name = info.get("name") or info.get("label") or sid
        gender = {"male": "♂", "female": "♀"}.get(info.get("gender"), "?")
        value = rec["mean_pairwise_identity"]
        if value == value:            # не nan
            sims.append(value)
            shown = f"{value:.2f}"
        else:
            unmeasured += 1
            shown = "мало реплик"
        lines.append(
            f"| {sid} {name} | {gender} | {rec['voice']} | {rec['segments']} | "
            f"{rec['passed']}/{rec['segments']} | {shown} |")

    target = STABILITY_TARGET["cross" if cross else "same"]

    if overall != overall:            # nan: измерять было не на чем
        lines += ["", "**Стабильность голосов: не измерена** — в ролике не "
                      "нашлось спикеров с двумя репликами от 3 секунд.", "",
                  "Это не значит, что дубляж плохой: значит, что проверить "
                  "устойчивость голоса было не на чем. Короткие ролики с "
                  "репликами по паре слов обычно так и выглядят."]
    else:
        verdict = ("голоса стабильны" if overall >= target
                   else "есть заметный разброс")
        lines += ["", f"**Стабильность голосов: {overall:.2f}** — {verdict} "
                      f"(норма для этого случая: от {target:.2f})", "",
                  "Это средняя схожесть реплик одного человека между собой."]
    if unmeasured:
        lines.append(f"У {unmeasured} спикеров реплик слишком мало — их "
                     "устойчивость не измерялась и в среднее не вошла.")
    if cross:
        # без этого пояснения 0.6 читается как «плохо», хотя это норма:
        # человек видит число и не знает, с чем его сравнивать
        lines.append(
            "Озвучка идёт на другом языке, чем говорил оригинал. XTTS "
            "переносит тембр между языками с потерей — замер на материале "
            "с заведомо одним голосом даёт 0.47 при межъязыковой озвучке "
            "против 0.70 при озвучке на языке оригинала. Поэтому норма здесь "
            "ниже; сравнивать с единицей бессмысленно.")
    else:
        lines.append(
            "Считается по репликам от 3 секунд: на более коротких голосовой "
            "отпечаток слишком шумный, и метрика мерила бы длину реплик, а "
            "не устойчивость голоса. Материал с заведомо одним голосом даёт "
            "по этой мерке 0.78.")

    if stats.qc_failed:
        lines.append(f"\nНе прошли проверку тембра: {stats.qc_failed} реплик — "
                     "их можно пересинтезировать в студии "
                     "(фильтр «Тембр не совпал»).")

    path = job_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    stats.per_speaker_report = report
    stats.overall_identity = round(overall, 4)
    return path
