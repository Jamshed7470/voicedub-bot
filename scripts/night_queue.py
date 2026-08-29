"""Ночная очередь: список ссылок → готовые дубляжи к утру.

Работает без присмотра, поэтому устроена по принципу «одна плохая ссылка
не должна стоить всей ночи»:

* каждая задача изолирована — падение одной не останавливает очередь;
* результат проверяется по факту (файл, длительность, наличие звука), а
  не по отсутствию исключения;
* место на диске проверяется перед каждой задачей, и очередь честно
  останавливается, не доведя диск до нуля;
* по каждой задаче пишется строка в отчёт, который можно прочитать утром
  за десять секунд.

    python -m scripts.night_queue ссылки.txt
    python -m scripts.night_queue url1 url2 --lang ru --hours 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import LOGS_DIR, load_config  # noqa: E402
from core.errors import UserError  # noqa: E402

log = logging.getLogger("night")

MIN_FREE_GB = 8.0          # ниже этого новую задачу не начинаем
REPORT = "ночной отчёт.md"


@dataclass
class QueueJob:
    """То же, что задача бота, но без чата: утверждение автоматическое."""
    kind: str
    payload: str
    target_lang: str
    settings: dict
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    user_id: int = 0
    chat_id: int = 0


def free_gb() -> float:
    return shutil.disk_usage(Path.cwd()).free / 1024 ** 3


def probe(path: Path) -> dict:
    """Что реально получилось: длительность, разрешение, есть ли звук."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration:stream=codec_type,width,height",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=120).stdout
        data = json.loads(out)
        streams = data.get("streams", [])
        video = next((s for s in streams if s.get("codec_type") == "video"), {})
        return {
            "duration": float(data.get("format", {}).get("duration", 0)),
            "width": video.get("width"), "height": video.get("height"),
            "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        }
    except Exception:  # noqa: BLE001
        return {}


def verify(result, source_duration: float | None) -> tuple[bool, str]:
    """Готов ли дубляж на самом деле.

    Проверяется файл, а не отсутствие исключения: задача может пройти до
    конца и оставить пустышку.
    """
    path = Path(result.saved_path) if result.saved_path else None
    if not path or not path.exists():
        return False, "файл не создан"
    size_mb = path.stat().st_size / 1024 ** 2
    if size_mb < 0.5:
        return False, f"файл пустой ({size_mb:.1f} МБ)"

    info = probe(path)
    if not info:
        return False, "файл не читается ffprobe"
    if not info.get("has_audio"):
        return False, "в файле нет звуковой дорожки"
    if source_duration and info.get("duration"):
        drift = abs(info["duration"] - source_duration)
        if drift > max(3.0, source_duration * 0.02):
            return False, (f"длительность {info['duration']:.0f} с против "
                           f"{source_duration:.0f} с у исходника")

    res = (f"{info.get('width')}x{info.get('height')}"
           if info.get("width") else "?")
    return True, f"{size_mb:.0f} МБ, {res}, {info.get('duration', 0):.0f} с"


def stability(job_id: str) -> str:
    """«Стабильность голосов» из отчёта задачи — главный показатель качества."""
    from project import store

    report = store.job_dir(job_id) / "report.md"
    if not report.exists():
        return "—"
    for line in report.read_text(encoding="utf-8").split("\n"):
        if "Стабильность голосов" in line:
            return line.split(":")[1].split("**")[0].strip()
    return "—"


async def run_one(url: str, lang: str, cfg) -> dict:
    """Один дубляж от ссылки до проверенного файла."""
    from core.pipeline import PipelineHooks, cleanup_job, run_job

    job = QueueJob(
        kind="url" if url.lower().startswith(("http://", "https://")) else "local",
        payload=url, target_lang=lang,
        settings={
            "keep_background": True,
            "keep_original_track": False,
            "translation_style": "normal",
            "voice_mode": "auto",
            "speakers": "auto",
            "auto_approve": True,      # ночью утверждать некому
        },
    )
    started = time.monotonic()
    stage = {"name": "старт"}

    async def report(st: int, label: str, pct: int) -> None:
        if pct in (0, 100):
            stage["name"] = label
            log.info("  этап %d/10: %s", st, label)

    hooks = PipelineHooks(report=report, confirm_same_lang=None,
                          cancel_event=None)
    try:
        result = await run_job(job, None, hooks, cfg)
    except UserError as e:
        return {"url": url, "ok": False, "why": e.message_ru,
                "stage": stage["name"], "minutes": (time.monotonic() - started) / 60}
    except Exception as e:  # noqa: BLE001
        log.exception("Задача упала")
        return {"url": url, "ok": False, "why": f"{type(e).__name__}: {e}",
                "stage": stage["name"], "minutes": (time.monotonic() - started) / 60}

    source_dur = None
    try:
        from core.media import probe_duration
        from project import store

        proj = store.load_or_none(job.id)
        if proj:
            source_dur = proj.source.duration_sec or None
    except Exception:  # noqa: BLE001
        pass

    ok, detail = verify(result, source_dur)
    entry = {
        "url": url, "ok": ok, "why": detail,
        "title": Path(result.saved_path).stem if result.saved_path else "",
        "path": str(result.saved_path or ""),
        "srt": str(result.saved_srt or ""),
        "stability": stability(job.id),
        "minutes": (time.monotonic() - started) / 60,
        "stage": "готово" if ok else stage["name"],
    }
    if ok:
        cleanup_job(job.id)          # место нужно следующей задаче
    return entry


def write_report(results: list[dict], out: Path) -> None:
    done = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]

    lines = [
        "# Ночной дубляж", "",
        f"Готово: **{len(done)}** из {len(results)}",
        f"Свободно на диске: {free_gb():.0f} ГБ", "",
    ]
    if done:
        lines += ["## Готово к просмотру", "",
                  "| Фильм | Стабильность | Время | Файл |",
                  "|---|---|---|---|"]
        for r in done:
            lines.append(f"| {r['title'][:60]} | {r['stability']} | "
                         f"{r['minutes']:.0f} мин | {r['why']} |")
        lines.append("")
    if failed:
        lines += ["## Не получилось", "",
                  "| Ссылка | Где остановилось | Причина |", "|---|---|---|"]
        for r in failed:
            lines.append(f"| {r['url'][:50]} | {r['stage']} | {r['why'][:80]} |")
    out.write_text("\n".join(lines), encoding="utf-8")


async def notify(cfg, results: list[dict], chat_id: int) -> None:
    """Утреннее сообщение в Telegram: что готово."""
    if not chat_id or not cfg.bot_token:
        return
    from aiogram import Bot

    done = [r for r in results if r["ok"]]
    text = [f"🌙 <b>Ночной дубляж: готово {len(done)} из {len(results)}</b>", ""]
    for r in done[:20]:
        text.append(f"✅ {r['title'][:55]}\n     стабильность {r['stability']}, "
                    f"{r['minutes']:.0f} мин")
    failed = [r for r in results if not r["ok"]]
    if failed:
        text.append(f"\n❌ Не получилось: {len(failed)} — подробности в файле "
                    f"«{REPORT}»")
    text.append("\nВсе файлы — в папке «готовые видео».")

    bot = Bot(cfg.bot_token)
    try:
        await bot.send_message(chat_id, "\n".join(text), parse_mode="HTML")
    except Exception:  # noqa: BLE001
        log.exception("Утреннее сообщение отправить не удалось")
    finally:
        await bot.session.close()


async def amain(args) -> int:
    cfg = load_config()
    urls: list[str] = []
    for item in args.sources:
        path = Path(item)
        if path.exists() and path.suffix.lower() in (".txt", ".list"):
            urls += [ln.strip() for ln in path.read_text(encoding="utf-8").split("\n")
                     if ln.strip() and not ln.strip().startswith("#")]
        else:
            urls.append(item)
    if not urls:
        log.error("Список пуст")
        return 1

    deadline = time.monotonic() + args.hours * 3600
    log.info("В очереди %d ссылок, времени %.1f ч, свободно %.0f ГБ",
             len(urls), args.hours, free_gb())

    results: list[dict] = []
    for i, url in enumerate(urls, 1):
        if time.monotonic() > deadline:
            log.warning("Время вышло — остановился на %d из %d", i - 1, len(urls))
            break
        if free_gb() < MIN_FREE_GB:
            log.warning("Осталось %.0f ГБ — дальше не берусь, чтобы не забить "
                        "диск полностью", free_gb())
            break

        log.info("[%d/%d] %s", i, len(urls), url)
        entry = await run_one(url, args.lang, cfg)
        results.append(entry)
        log.info("[%d/%d] %s — %s (%.0f мин)", i, len(urls),
                 "ГОТОВО" if entry["ok"] else "НЕ ВЫШЛО", entry["why"],
                 entry["minutes"])
        write_report(results, cfg.output_dir / REPORT)

    write_report(results, cfg.output_dir / REPORT)
    await notify(cfg, results, args.chat)
    done = sum(1 for r in results if r["ok"])
    log.info("Итог: готово %d из %d. Отчёт: %s", done, len(results),
             cfg.output_dir / REPORT)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Ночная очередь дубляжа")
    ap.add_argument("sources", nargs="+", help="ссылки или файл со списком")
    ap.add_argument("--lang", default="ru", help="язык озвучки")
    ap.add_argument("--hours", type=float, default=9.0,
                    help="через сколько часов остановиться")
    ap.add_argument("--chat", type=int, default=0,
                    help="chat_id для утреннего сообщения")
    args = ap.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(LOGS_DIR / "night_queue.log",
                                      encoding="utf-8")])
    for noisy in ("faster_whisper", "speechbrain", "pytorch_lightning", "TTS"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    raise SystemExit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()
