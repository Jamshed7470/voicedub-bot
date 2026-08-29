"""Генерация тестовых аудио с известным ответом.

Пороги слияния и уверенности нельзя откалибровать на настоящем фильме:
там неизвестно, сколько на самом деле человек говорит. Поэтому материал
синтезируется голосами из банка — тогда точно известно, кто где говорит,
и можно измерить, сколько раз алгоритм ошибся.

Стороннего контента не используется: всё сгенерировано.

    python -m scripts.make_fixtures            # все сценарии
    python -m scripts.make_fixtures --only dialog
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import ROOT, load_config      # noqa: E402

log = logging.getLogger("fixtures")

FIXTURES_DIR = ROOT / "tests" / "fixtures"
SR = 24000

# Реплики нейтральные и разной длины: короткие проверяют поведение на
# ненадёжных отпечатках, длинные — на надёжных.
LINES = [
    "Сегодня в городе тепло, и до вечера обещают ясную погоду.",
    "Я думаю, нам стоит обсудить это ещё раз.",
    "Хорошо.",
    "На улице много людей, работают кафе, ходят автобусы.",
    "Подожди минуту, я сейчас проверю расписание и перезвоню тебе.",
    "Да, конечно.",
    "Мне кажется, ты немного преувеличиваешь значение этой встречи.",
    "Совсем нет.",
    "Вчера я закончил читать книгу, которую ты советовал прошлым летом.",
    "И как тебе?",
    "Очень понравилось, особенно вторая половина.",
    "Значит, договорились: встречаемся завтра в семь у входа.",
    "Не опаздывай.",
    "Постараюсь, но движение в это время бывает плотным.",
    "Тогда выезжай пораньше.",
    "Так и сделаю, спасибо за напоминание.",
]

SCENARIOS = {
    "monologue": {
        "title": "8 минут одного голоса",
        "speakers": 1,
        "turns": 60,
        "overlap": False,
        "music": False,
    },
    "dialog": {
        "title": "диалог двух голосов с наложениями",
        "speakers": 2,
        "turns": 44,
        "overlap": True,
        "music": False,
    },
    "crowd": {
        "title": "четыре голоса под музыкой",
        "speakers": 4,
        "turns": 48,
        "overlap": True,
        "music": True,
    },
}


def pick_voices(bank, count: int) -> list:
    """Берёт максимально непохожие голоса: разный пол, разный тон.

    Если взять соседние по тембру, тест будет проверять не алгоритм, а
    удачу — а нам нужен воспроизводимый результат.
    """
    voices = [v for v in bank.all() if v.has_profile and not v.is_child]
    if len(voices) < count:
        raise SystemExit(
            f"В банке {len(voices)} голосов с профилем, нужно {count}. "
            "Соберите банк: python -m scripts.build_voice_bank --from-dir voice_db")

    males = sorted((v for v in voices if v.gender == "male"),
                   key=lambda v: v.f0_hz or 999)
    females = sorted((v for v in voices if v.gender == "female"),
                     key=lambda v: v.f0_hz or 999)

    chosen: list = []
    pools = [males, females]
    # чередуем пол, внутри пола берём с разных концов диапазона тона
    while len(chosen) < count:
        pool = pools[len(chosen) % 2] or pools[(len(chosen) + 1) % 2]
        if not pool:
            break
        chosen.append(pool.pop(0 if len(chosen) % 4 < 2 else -1))
    return chosen[:count]


def build(name: str, cfg, bank, engine) -> dict:
    """Синтезирует один сценарий и возвращает разметку."""
    import numpy as np
    import soundfile as sf

    spec = SCENARIOS[name]
    out_dir = FIXTURES_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    voices = pick_voices(bank, spec["speakers"])
    log.info("%s: голоса %s", name,
             ", ".join(f"{v.display_name} ({v.gender})" for v in voices))

    pieces: list[tuple[float, np.ndarray, int]] = []
    truth: list[dict] = []
    t = 1.0
    for i in range(spec["turns"]):
        who = i % spec["speakers"]
        text = LINES[i % len(LINES)]
        profile = bank.load_profile(voices[who].id)

        wav = out_dir / f"_turn{i}.wav"
        engine.synthesize(text, "ru", profile, wav, speed=1.0,
                          seed=1000 + i)
        data, _ = sf.read(str(wav), dtype="float32")
        wav.unlink(missing_ok=True)
        dur = len(data) / SR

        # наложение: каждая шестая реплика начинается раньше конца прошлой
        start = t
        if spec["overlap"] and i > 0 and i % 6 == 0:
            start = max(0.0, t - min(1.2, dur * 0.4))

        pieces.append((start, data, who))
        truth.append({"id": i + 1, "start": round(start, 3),
                      "end": round(start + dur, 3), "speaker": f"TRUE{who}",
                      "text": text, "voice": voices[who].id})
        t = start + dur + 0.45

    total = int((t + 1.0) * SR)
    mix = np.zeros(total, dtype=np.float32)
    for start, data, _ in pieces:
        at = int(start * SR)
        width = min(len(data), total - at)
        if width > 0:
            mix[at:at + width] += data[:width]

    if spec["music"]:
        mix += _music_bed(total, SR) * 0.12

    peak = float(np.max(np.abs(mix))) or 1.0
    mix = mix * min(1.0, 0.89 / peak)

    audio_path = out_dir / "audio.wav"
    sf.write(str(audio_path), mix, SR)

    # версия 16 кГц — именно с ней работают анализаторы
    import librosa
    y16 = librosa.resample(mix, orig_sr=SR, target_sr=16000)
    sf.write(str(out_dir / "audio16.wav"), y16, 16000)

    meta = {
        "name": name, "title": spec["title"],
        "duration_sec": round(total / SR, 2),
        "true_speakers": spec["speakers"],
        "voices": [v.id for v in voices],
        "has_music": spec["music"],
        "segments": truth,
    }
    (out_dir / "truth.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("%s: %.1f мин, %d реплик, %d голосов → %s",
             name, total / SR / 60, len(truth), spec["speakers"], out_dir)
    return meta


def _music_bed(length: int, sr: int):
    """Тихая музыкальная подложка: аккорд с медленным дыханием громкости.

    Нужна, чтобы проверить, что голоса не слипаются под музыкой — это
    одна из причин, по которым pyannote дробит спикеров.
    """
    import numpy as np

    t = np.arange(length) / sr
    chord = sum(np.sin(2 * np.pi * f * t) for f in (146.8, 220.0, 293.7))
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 0.08 * t)
    return (chord / 3 * envelope).astype(np.float32)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(SCENARIOS),
                    help="собрать один сценарий")
    args = ap.parse_args()

    cfg = load_config()
    from synth.xtts_engine import get_engine
    from voices.bank import get_bank

    bank = get_bank()
    engine = get_engine(cfg)
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    names = [args.only] if args.only else list(SCENARIOS)
    built = [build(name, cfg, bank, engine) for name in names]

    index = {m["name"]: {k: v for k, v in m.items() if k != "segments"}
             for m in built}
    path = FIXTURES_DIR / "index.json"
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            pass
    existing.update(index)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=1),
                    encoding="utf-8")

    print("\nГотово:")
    for m in built:
        print(f"  {m['name']:10} {m['duration_sec'] / 60:5.1f} мин  "
              f"{len(m['segments']):3d} реплик  {m['true_speakers']} голосов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
