"""Сквозной прогон: видео → пайплайн → студия → утверждение → результат.

    python -m scripts.e2e_check [видео] [язык]

Поднимает студию, гонит задачу через настоящие модели и изображает
человека: дожидается паузы на проверку, читает проект через API, правит
реплику, утверждает — и проверяет, что правка дошла до озвучки.

Проверяется всё вместе, на настоящих моделях: распознавание, сведение
спикеров, фиксация профилей, кастинг, пауза на проверку, утверждение
через API студии, синтез с Identity QC, микс и сборка.

Исходник по умолчанию — фикстура с ИЗВЕСТНЫМ ответом (2 голоса), поэтому
в конце можно сказать не «сработало», а «сработало правильно».
Собрать фикстуры: python -m scripts.make_fixtures
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)

VIDEO = Path(sys.argv[1] if len(sys.argv) > 1
             else "tests/fixtures/dialog/audio.wav")
TARGET_LANG = sys.argv[2] if len(sys.argv) > 2 else "en"
JOB_ID = "e2echeck"
PORT = 8097
# секрет только для этой проверки: настоящий STUDIO_SECRET из .env не трогаем
SECRET = "e2e-check-secret-0123456789abcdefghijklmnop"

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for noisy in ("speechbrain", "urllib3", "numba", "matplotlib", "uvicorn.access"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("e2e")

from core.config import load_config  # noqa: E402

cfg = load_config()
cfg.studio_enabled = True
cfg.studio_secret = SECRET
cfg.studio_public_url = f"http://127.0.0.1:{PORT}"
cfg.yaml.setdefault("studio", {}).update(
    {"enabled": True, "host": "127.0.0.1", "port": PORT})

import core.config  # noqa: E402
core.config._config = cfg

from core.errors import UserError  # noqa: E402
from core.pipeline import PipelineHooks, run_job  # noqa: E402
from project import store  # noqa: E402
from project.schema import Stage  # noqa: E402
from studio import auth  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    log.info("%s %s%s", "OK  " if ok else "ПРОВАЛ", name,
             f" — {detail}" if detail else "")


class Job:
    kind = "local"
    target_lang = TARGET_LANG   # ru → en по умолчанию: перевод тоже участвует
    id = JOB_ID
    user_id = 777
    chat_id = 777

    def __init__(self, path: Path):
        self.payload = str(path.resolve())
        self.settings = {
            "keep_background": True,
            "keep_original_track": False,
            "translation_style": "normal",
            "voice_mode": "auto",
            "speakers": "auto",
            "auto_approve": False,        # ПАУЗА на проверку обязательна
        }


class FakeBot:
    """Вместо Telegram: печатает то, что бот отправил бы пользователю."""

    def __init__(self):
        self.messages: list[str] = []

    async def send_message(self, chat_id, text, **kw):
        self.messages.append(text)
        log.info("БОТ → пользователю:\n%s", text)

        class M:
            message_id = len(self.messages)
        return M()


# --------------------------------------------------------------- студия

def start_studio():
    import uvicorn

    from studio.server import create_app

    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1",
                                           port=PORT, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(60):
        time.sleep(0.25)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
            return server
        except Exception:
            pass
    raise RuntimeError("студия не поднялась")


def api(path, method="GET", body=None, version=None):
    token = auth.make_token(JOB_ID, 777, SECRET)
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/api/projects/{JOB_ID}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        method=method)
    req.add_header("Authorization", f"Bearer {token}")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    if version is not None:
        req.add_header("If-Match", str(version))
    try:
        r = urllib.request.urlopen(req, timeout=60)
        return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}


async def act_as_human():
    """Изображает человека в студии: ждёт паузы, смотрит, правит, утверждает."""
    log.info("--- ЧЕЛОВЕК: жду, когда проект встанет на проверку ---")
    proj = None
    for _ in range(1200):                       # до 20 минут на разбор
        await asyncio.sleep(1.0)
        proj = store.load_or_none(JOB_ID)
        if proj and proj.stage == Stage.REVIEW:
            break
    if proj is None or proj.stage != Stage.REVIEW:
        check("проект встал на проверку", False,
              f"стадия {proj.stage.value if proj else 'нет проекта'}")
        return

    check("проект встал на проверку (INV-5)", True,
          f"{len(proj.active_speakers())} спикеров, {len(proj.segments)} реплик")

    code, data = api("")
    check("студия отдаёт проект по подписанной ссылке", code == 200,
          f"HTTP {code}")
    if code != 200:
        return

    speakers = data["speakers"]
    check("найдено ровно 2 спикера (в фикстуре их 2)",
          len([s for s in speakers if not s["merged_into"]]) == 2,
          f"найдено {len([s for s in speakers if not s['merged_into']])}")

    locked = [s for s in speakers if s["voice"]["locked"]]
    check("у каждого спикера заблокированный профиль (INV-1)",
          len(locked) == len(speakers), f"{len(locked)} из {len(speakers)}")

    voices = {s["voice"].get("preset_id") or f"clone:{s['id']}" for s in speakers}
    check("голоса спикеров различаются", len(voices) == len(speakers),
          ", ".join(sorted(str(v) for v in voices)))

    # рендер до утверждения запрещён
    code_bad, _ = api("/approve", "POST", {}, version=data["version"] + 99)
    check("правка с чужой версией отклоняется (409)", code_bad == 409,
          f"HTTP {code_bad}")

    # человек правит перевод одной реплики
    seg_id = data["segments"][1]["id"]
    code_p, patched = api(f"/segments/{seg_id}", "PATCH",
                          {"text_tgt": "This line was edited by a human."},
                          version=data["version"])
    check("правка перевода через студию принята", code_p == 200, f"HTTP {code_p}")

    saved = store.load(JOB_ID)
    check("правка помечена как ручная (INV-3)",
          "text_tgt" in saved.segment(seg_id).edited_by_user.fields,
          str(saved.segment(seg_id).edited_by_user.fields))

    log.info("--- ЧЕЛОВЕК: нажимаю «Утвердить и рендерить» ---")
    code_a, res = api("/approve", "POST", {}, version=saved.version)
    check("утверждение принято", code_a == 200 and res.get("stage") == "approved",
          f"HTTP {code_a} {res}")

    from bot.review import enqueue_approved
    await enqueue_approved(JOB_ID)
    return seg_id


async def main() -> int:
    if not VIDEO.exists():
        log.error("нет файла %s", VIDEO)
        return 1

    store.job_dir(JOB_ID).mkdir(parents=True, exist_ok=True)
    (store.job_dir(JOB_ID) / "approved.flag").unlink(missing_ok=True)

    start_studio()
    check("студия запущена", True, f"http://127.0.0.1:{PORT}")

    job, bot = Job(VIDEO), FakeBot()
    t0 = time.monotonic()

    async def report(stage, label, pct):
        if pct in (0, 100):
            log.info("этап %d/10: %s… %d%%", stage, label, pct)

    hooks = PipelineHooks(report=report, confirm_same_lang=None,
                          cancel_event=None)

    human = asyncio.create_task(act_as_human())
    try:
        result = await run_job(job, bot, hooks, cfg)
    except UserError as e:
        check("пайплайн дошёл до конца", False, e.message_ru)
        human.cancel()
        return summarize(2)
    except Exception as e:  # noqa: BLE001
        log.exception("пайплайн упал")
        check("пайплайн дошёл до конца", False, repr(e))
        human.cancel()
        return summarize(3)

    edited_id = await human
    elapsed = time.monotonic() - t0
    check("пайплайн дошёл до конца", True, f"за {elapsed / 60:.1f} мин")

    # ---------- что получилось ----------
    out = Path(result.saved_path) if result.saved_path else None
    check("готовый файл создан", bool(out and out.exists()),
          f"{out.name if out else '—'} "
          f"({out.stat().st_size / 1024 / 1024:.1f} МБ)" if out and out.exists() else "")

    srt = Path(result.saved_srt) if result.saved_srt else None
    check("субтитры созданы", bool(srt and srt.exists()),
          srt.name if srt else "")

    report_path = store.job_dir(JOB_ID) / "report.md"
    check("карта голосов записана", report_path.exists())
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        log.info("\n%s", text)
        stability = None
        for line in text.split("\n"):
            if "Стабильность голосов" in line:
                try:
                    stability = float(line.split(":")[1].split("**")[0].strip())
                except (IndexError, ValueError):
                    pass
        # норма зависит от языка: замер на материале с заведомо одним
        # голосом даёт 0.78 при озвучке на языке оригинала и заметно
        # меньше при межъязыковой — выше этого не поднимется ничто
        # норма зависит от языка: замер на материале с заведомо одним
        # голосом даёт 0.78 по репликам от 3 с при озвучке на языке
        # оригинала и заметно меньше при межъязыковой
        target = 0.75 if TARGET_LANG == "ru" else 0.60
        check(f"стабильность голосов >= {target}",
              bool(stability and stability >= target),
              f"{stability}" if stability else "не найдена в отчёте")

    # правка человека дошла до озвучки
    final = store.load_or_none(JOB_ID)
    if final and edited_id:
        seg = final.segment(edited_id)
        check("правка человека применена при рендере (INV-3)",
              bool(seg and "human" in (seg.text_tts or "").lower()),
              (seg.text_tts or "")[:60] if seg else "")

    # длительность результата совпадает с исходником
    if out and out.exists():
        import subprocess
        try:
            dur = float(subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", str(out)],
                capture_output=True, text=True, timeout=60).stdout.strip())
            check("длительность совпадает с исходником (±2 с)",
                  abs(dur - 144.8) <= 2.0, f"{dur:.1f} с против 144.8 с")
        except Exception as e:  # noqa: BLE001
            check("длительность проверена", False, repr(e))

    return summarize(0)


def summarize(code: int) -> int:
    ok = sum(1 for _, good, _ in CHECKS if good)
    print("\n" + "=" * 72)
    print(f"СКВОЗНОЙ ПРОГОН: {ok} из {len(CHECKS)} проверок пройдено")
    print("=" * 72)
    for name, good, detail in CHECKS:
        print(f"  {'✓' if good else '✗'} {name}" + (f" — {detail}" if detail else ""))
    return code if code else (0 if ok == len(CHECKS) else 1)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
