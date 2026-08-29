"""Распознавание речи с пословными таймингами (WhisperX)."""
from __future__ import annotations

import gc
import logging
import os
import re
import sys
from pathlib import Path

from core.errors import UserError

log = logging.getLogger(__name__)


def _pick_compute_type(cfg) -> str:
    """float16 точнее квантованного int8, но требует ~5 ГБ VRAM для large-v3.

    Смотрим на СВОБОДНУЮ память, а не на общую: карту может занимать соседний
    процесс, и тогда «по паспорту подходит» заканчивается OOM в середине
    распознавания — то есть после часа уже сделанной работы.
    """
    requested = str(cfg.y("asr", "compute_type_gpu", default="auto"))
    if requested != "auto":
        return requested
    try:
        import torch
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        choice = "float16" if (free_gb >= 6 and total_gb >= 7) else "int8_float16"
        log.info("VRAM: свободно %.1f из %.1f ГБ → compute_type %s",
                 free_gb, total_gb, choice)
        return choice
    except Exception:  # noqa: BLE001
        return "int8_float16"


def speech_coverage(audio, segments, sr: int = 16000) -> tuple[float, list[tuple]]:
    """Какая доля речи попала в распознанное. Возвращает (доля, пропуски).

    Считаем по энергии дорожки: где заметно громче шумового пола — там речь
    (дорожка уже очищена от музыки). Участки без единого сегмента ASR —
    кандидаты в потери.
    """
    import numpy as np

    frame = int(sr * 0.1)
    if len(audio) < frame * 5:
        return 1.0, []
    frames = audio[: len(audio) // frame * frame].reshape(-1, frame)
    rms = np.sqrt((frames.astype(np.float32) ** 2).mean(axis=1))
    floor = float(np.percentile(rms, 20))
    loud = rms > max(floor * 3, 0.004)
    if not loud.any():
        return 1.0, []

    covered = np.zeros(len(rms), dtype=bool)
    for s in segments:
        a = max(0, int(s.get("start", 0) * 10))
        b = min(len(rms), int(s.get("end", 0) * 10) + 1)
        covered[a:b] = True

    missed = loud & ~covered
    gaps, run = [], None
    for i, flag in enumerate(missed):
        if flag and run is None:
            run = i
        elif not flag and run is not None:
            if (i - run) >= 8:  # пропуски короче 0.8 с не считаем
                gaps.append((run / 10, i / 10))
            run = None
    if run is not None and (len(missed) - run) >= 8:
        gaps.append((run / 10, len(missed) / 10))

    coverage = 1.0 - float(missed.sum()) / float(loud.sum())
    return coverage, gaps


# Артефакты Whisper на не-речи: модель «дописывает» титры субтитров.
HALLUCINATION_RE = re.compile(
    r"alt\s*yaz|subtitle|subs?\s+by|amara\.org|уou\s*tube|субтитр|"
    r"редактор\s+субтитров|翻訳|字幕|подписыв|thanks?\s+for\s+watching",
    re.IGNORECASE)


def _looks_like_speech(text: str, logprob: float, no_speech: float,
                       compression: float,
                       min_logprob: float = -0.85) -> tuple[bool, str]:
    """Отличает реальную реплику от выдумки модели на шуме и пении.

    Порогов уверенности мало: пение модель распознаёт уверенно, но выдаёт
    бессмыслицу, а короткий выкрик в драке — реальный, но с низкой оценкой.
    Поэтому смотрим ещё на форму текста. Планка намеренно строгая: вписать
    в дубляж выдуманную фразу хуже, чем потерять междометие.
    """
    t = text.strip()
    if len(t) < 2:
        return False, "пусто"
    if HALLUCINATION_RE.search(t):
        return False, "штамп субтитров"
    if no_speech > 0.5:
        return False, f"не речь ({no_speech:.2f})"
    if logprob < min_logprob:
        return False, f"низкая уверенность ({logprob:.2f})"
    if compression > 2.4:
        return False, "повторы"
    letters = [c for c in t if c.isalpha()]
    if len(letters) >= 10 and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        return False, "капсом (обычно пение)"
    return True, ""


def _recover_missed_speech(model, audio, segments: list[dict], language: str,
                           cfg) -> list[dict]:
    """Второй проход по участкам, где есть голос, но текста нет.

    Основной проход отсекается VAD, и реплики под музыку или крик в драке
    теряются целиком. Здесь такие места распознаются отдельно, без VAD,
    а результат проходит фильтр от галлюцинаций.
    """
    _, gaps = speech_coverage(audio, segments)
    if not gaps:
        return []
    max_gaps = int(cfg.y("asr", "recover_max_regions", default=25))
    min_len = float(cfg.y("asr", "recover_min_s", default=0.8))
    gaps = [g for g in gaps if g[1] - g[0] >= min_len][:max_gaps]

    min_logprob = float(cfg.y("asr", "recover_min_logprob", default=-0.85))
    recovered: list[dict] = []
    for start, end in gaps:
        piece = audio[int(start * 16000):int(end * 16000)]
        try:
            found, _ = model.transcribe(
                piece, language=language, vad_filter=False, beam_size=5,
                no_speech_threshold=0.9, log_prob_threshold=-2.0,
                condition_on_previous_text=False)
            found = list(found)
        except Exception:  # noqa: BLE001 — дораспознавание не должно ломать этап
            log.exception("Дораспознавание участка %.1f–%.1fс не удалось",
                          start, end)
            continue
        for s in found:
            ok, reason = _looks_like_speech(s.text, s.avg_logprob,
                                            s.no_speech_prob,
                                            s.compression_ratio, min_logprob)
            if not ok:
                log.info("Дораспознавание %.1f–%.1fс: отброшено (%s) — %r",
                         start, end, reason, s.text.strip()[:60])
                continue
            recovered.append({
                "start": round(start + s.start, 3),
                "end": round(min(end, start + s.end), 3),
                "text": s.text.strip(),
            })
            log.info("Дораспознавание %.1f–%.1fс: принято — %r",
                     start, end, s.text.strip()[:60])
    return recovered


def _detect_language(model, audio, cfg) -> str:
    """Определяет язык оригинала голосованием по всему ролику.

    Штатный whisperx смотрит ТОЛЬКО первые 30 секунд. У фильмов и сериалов
    там заставка, музыка или тишина — язык угадывается по шуму, и весь
    дальнейший разбор идёт вхолостую: текст выходит фонетической кашей,
    а ошибку видно только в конце, по бессмысленному дубляжу.
    """
    import numpy as np
    from whisperx.audio import N_SAMPLES, SAMPLE_RATE, log_mel_spectrogram

    forced = cfg.y("asr", "language", default=None)
    if forced:
        log.info("Язык оригинала задан в настройках: %s", forced)
        return str(forced)

    probes = int(cfg.y("asr", "lang_probes", default=12))
    min_prob = float(cfg.y("asr", "lang_min_prob", default=0.5))
    min_rms = float(cfg.y("asr", "lang_min_rms", default=0.005))

    total = int(audio.shape[0])
    if total <= N_SAMPLES:  # короткий файл — брать нечего, окно одно
        return model.detect_language(audio)

    n_mels = model.model.feat_kwargs.get("feature_size") or 80
    # равномерно по всей длине, с отступом от краёв: начало — заставка,
    # конец — титры, и то и другое речи обычно не содержит
    last_start = total - N_SAMPLES
    offsets = [int(last_start * (i + 0.5) / probes) for i in range(probes)]

    scores: dict[str, float] = {}
    votes: dict[str, int] = {}
    checked = 0
    for off in offsets:
        window = audio[off:off + N_SAMPLES]
        if float(np.sqrt(np.mean(window.astype(np.float64) ** 2))) < min_rms:
            continue  # тишина: детектор вернёт случайный язык с низкой верой
        try:
            segment = log_mel_spectrogram(window, n_mels=n_mels, padding=0)
            results = model.model.model.detect_language(model.model.encode(segment))
            token, prob = results[0][0]
        except Exception:  # noqa: BLE001 — одно окно не должно валить разбор
            log.exception("Определение языка: окно на %.0fс не удалось",
                          off / SAMPLE_RATE)
            continue
        checked += 1
        if prob < min_prob:
            continue
        lang = token[2:-2]
        scores[lang] = scores.get(lang, 0.0) + float(prob)
        votes[lang] = votes.get(lang, 0) + 1

    if not scores:
        # ни одно окно не дало уверенного ответа — честнее откатиться
        # на штатное поведение, чем молча выдумать язык
        log.warning("Язык не определился по %d окнам — беру первые 30 с", checked)
        return model.detect_language(audio)

    best = max(scores, key=scores.get)
    log.info("Язык оригинала: %s (%d из %d окон; распределение: %s)",
             best, votes[best], checked,
             ", ".join(f"{k}={v}" for k, v in sorted(votes.items(),
                                                     key=lambda kv: -kv[1])))
    return best


def _prefetch_align_model(language: str, cfg) -> None:
    """Скачивает модель выравнивания устойчивым загрузчиком.

    Штатный путь huggingface_hub на рвущейся сети виснет посреди файла
    и держит задачу часами — поэтому большие веса тянем сами, с докачкой.
    """
    try:
        from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF
    except ImportError:
        return
    repo = DEFAULT_ALIGN_MODELS_HF.get(language)
    if not repo:
        return  # язык идёт через torchaudio или модели нет вовсе
    try:
        from core.hfget import ensure_model
        ensure_model(repo, token=cfg.hf_token or None)
    except Exception:  # noqa: BLE001 — не смогли, пусть пробует whisperx
        log.exception("Не удалось предзагрузить модель выравнивания %s", repo)


def transcribe(analysis_wav: str, cfg, progress=None) -> dict:
    """Распознаёт речь: автоопределение языка + word-level alignment.

    Возвращает dict: {"language": str, "segments": [{start, end, text, words: [...]}]}.
    """
    import torch

    from core.sb_compat import patch_speechbrain_lazy_imports
    patch_speechbrain_lazy_imports()

    # Windows: ctranslate2 ищет cuDNN-DLL в PATH — подключаем папку torch\lib,
    # где лежат cudnn*_9.dll (иначе процесс молча падает на загрузке модели)
    if sys.platform == "win32":
        torch_lib = Path(torch.__file__).parent / "lib"
        if torch_lib.is_dir():
            os.add_dll_directory(str(torch_lib))
            os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")

    import whisperx

    device = cfg.device
    compute_type = (_pick_compute_type(cfg) if device == "cuda"
                    else cfg.y("asr", "compute_type_cpu", default="int8"))
    batch_size = int(cfg.y("asr", "batch_size", default=8))

    # VAD по умолчанию отсекает тихую речь под музыку — снижаем пороги;
    # no_speech_threshold ниже: иначе реплики в шуме считаются «тишиной»
    asr_options = {
        "no_speech_threshold": float(cfg.y("asr", "no_speech_threshold",
                                           default=0.45)),
        "log_prob_threshold": float(cfg.y("asr", "log_prob_threshold",
                                          default=-1.2)),
        "condition_on_previous_text": False,
    }
    vad_options = {
        "vad_onset": float(cfg.y("asr", "vad_onset", default=0.35)),
        "vad_offset": float(cfg.y("asr", "vad_offset", default=0.25)),
    }

    log.info("WhisperX: модель %s, устройство %s, compute_type %s",
             cfg.whisper_model, device, compute_type)
    try:
        model = whisperx.load_model(cfg.whisper_model, device,
                                    compute_type=compute_type,
                                    asr_options=asr_options,
                                    vad_options=vad_options)
    except (TypeError, ValueError):
        log.warning("WhisperX не принял настройки VAD/порогов — беру умолчания")
        model = whisperx.load_model(cfg.whisper_model, device,
                                    compute_type=compute_type)
    audio = whisperx.load_audio(str(analysis_wav))

    if progress:
        progress(30)
    # язык определяем сами и передаём явно: иначе whisperx возьмёт его из
    # первых 30 секунд, где у фильмов заставка, и распознает весь ролик
    # как чужой язык — молча, без единой ошибки в логе
    language = _detect_language(model, audio, cfg)
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    language = result.get("language") or language
    segments = result.get("segments") or []
    if not segments or not any((s.get("text") or "").strip() for s in segments):
        raise UserError(
            "В этом файле не нашлось речи — нечего дублировать. "
            "Проверьте, что в видео/аудио кто-то говорит."
        )

    if progress:
        progress(70)
    # word-level alignment — тайминги слов нужны для точной нарезки сегментов
    try:
        _prefetch_align_model(language, cfg)
        align_model, metadata = whisperx.load_align_model(language_code=language,
                                                          device=device)
        result = whisperx.align(segments, align_model, metadata, audio, device,
                                return_char_alignments=False)
        result["language"] = language

        # второй проход — только после выравнивания: до него границы реплик
        # растянуты широко и настоящие пропуски не видны
        if bool(cfg.y("asr", "recover_missed", default=True)):
            try:
                extra = _recover_missed_speech(model.model, audio,
                                               result["segments"], language, cfg)
                if extra:
                    extra = whisperx.align(extra, align_model, metadata, audio,
                                           device,
                                           return_char_alignments=False)["segments"]
                    result["segments"] = sorted(result["segments"] + extra,
                                                key=lambda s: s["start"])
                    log.info("Дораспознано реплик: %d", len(extra))
            except Exception:  # noqa: BLE001 — не должно ломать основной результат
                log.exception("Второй проход распознавания не удался")

        coverage, _ = speech_coverage(audio, result["segments"])
        log.info("Распознано %.0f%% речи", coverage * 100)
        del align_model
    except Exception:  # для редких языков alignment-модели может не быть
        log.exception("Alignment не удался, использую тайминги Whisper как есть")
        result = {"language": language, "segments": segments}

    result["segments"] = _drop_hallucinations(result.get("segments") or [])

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def _drop_hallucinations(segments: list[dict]) -> list[dict]:
    """Убирает штампы субтитров и реплики без единой буквы.

    На заставке и под музыку Whisper уверенно выдаёт «Altyazı M.K.»,
    «Subtitles by…», «... ... ...» — это артефакты обучающих данных, а не
    речь. Второй проход такое отсекает, основной — нет, и штамп доезжает
    до озвучки: голос посреди фильма произносит «Субтитры М. К.».
    """
    kept, dropped = [], 0
    for s in segments:
        t = (s.get("text") or "").strip()
        # без букв и цифр реплики не бывает: это точки, тире и многоточия
        if not t or not re.search(r"\w", t, re.UNICODE):
            dropped += 1
            continue
        if HALLUCINATION_RE.search(t):
            log.info("Распознавание: выброшен штамп субтитров на %.1fс — %r",
                     s.get("start", 0.0), t[:60])
            dropped += 1
            continue
        kept.append(s)
    if dropped:
        log.info("Распознавание: выброшено %d служебных реплик из %d",
                 dropped, len(segments))
    return kept
