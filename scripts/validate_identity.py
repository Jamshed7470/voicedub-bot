"""Проверка Speaker Identity Engine на реальном закэшированном разборе.

Берёт транскрипт настоящего фильма (тот самый, где старый код нашёл
58 «спикеров»), прогоняет по нему SIE и печатает, во что превратились
кластеры. Это замер на боевых данных и боевых параметрах, а не на
синтетике: именно так проверяется, что исправление работает.

    python -m scripts.validate_identity data/cache/<key>/analysis_auto \\
        --vocals data/cache/<key>/vocals16.wav
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("validate")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis_dir", help="папка с transcript.json")
    ap.add_argument("--vocals", required=True, help="vocals16.wav")
    ap.add_argument("--speakers", default="auto", help="подсказка о числе голосов")
    ap.add_argument("--limit", type=int, default=0, help="взять первые N сегментов")
    args = ap.parse_args()

    cfg = load_config()
    data = json.loads((Path(args.analysis_dir) / "transcript.json").read_text(
        encoding="utf-8"))
    segments = data["segments"]
    if args.limit:
        segments = segments[:args.limit]
    for i, seg in enumerate(segments, 1):
        seg.setdefault("id", i)
        seg["flags"] = []

    before = Counter(s["speaker"] for s in segments)
    print(f"\nДО: {len(before)} кластеров на {len(segments)} сегментах")
    print("   крупнейшие:", ", ".join(
        f"{k}={v}" for k, v in before.most_common(8)))

    import identity as sie

    t0 = time.monotonic()
    result = sie.analyze(args.vocals, segments, cfg,
                         raw_turns=None, job_dir=None,
                         speakers_hint=args.speakers)
    elapsed = time.monotonic() - t0

    after = Counter(s["speaker"] for s in segments)
    speakers = result["speakers"]
    print(f"\nПОСЛЕ: {len(after)} спикеров (за {elapsed:.0f} с)")
    print(f"{'ID':<5}{'реплик':>8}{'речь, с':>10}{'плотность':>11}  собран из")
    for sid, rec in sorted(speakers.items(),
                           key=lambda kv: -kv[1]["speech_sec"])[:15]:
        merged = len(rec["merged_from"])
        print(f"{sid:<5}{rec['segments']:>8}{rec['speech_sec']:>10.0f}"
              f"{rec['spread']:>11.3f}  {merged + 1} кластеров")

    flags = Counter(f for s in segments for f in s.get("flags", []))
    print("\nФлаги для проверки человеком:",
          ", ".join(f"{k}={v}" for k, v in flags.items()) or "нет")
    conf = [s.get("speaker_confidence", 0) for s in segments]
    print(f"Уверенность привязки: медиана {sorted(conf)[len(conf) // 2]:.3f}, "
          f"минимум {min(conf):.3f}")
    print(f"\nСокращение: {len(before)} → {len(after)} "
          f"({100 * (1 - len(after) / max(1, len(before))):.0f}% лишних "
          "кластеров убрано)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
