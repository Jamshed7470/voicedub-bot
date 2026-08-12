# -*- coding: utf-8 -*-
"""Перевод бота на локальный Bot API сервер (и обратно).

Telegram требует выйти из текущего сервера перед переездом: пока бот
«залогинен» в облаке, локальный сервер его не примет.

    python migrate_bot.py to-local     — выйти из облака, перед переездом
    python migrate_bot.py to-cloud     — выйти из локального, вернуться в облако
    python migrate_bot.py check        — где бот сейчас и что отвечает
"""
from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLOUD = "https://api.telegram.org"


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    path = ROOT / ".env"
    if not path.exists():
        sys.exit(f"Не найден {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    return env


def call(base: str, token: str, method: str) -> tuple[bool, str]:
    url = f"{base.rstrip('/')}/bot{token}/{method}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return True, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return False, e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def main() -> None:
    action = sys.argv[1] if len(sys.argv) > 1 else "check"
    env = load_env()
    token = env.get("BOT_TOKEN", "")
    local = env.get("TELEGRAM_LOCAL_API_URL", "") or "http://127.0.0.1:8081"
    if not token:
        sys.exit("BOT_TOKEN не задан в .env")

    if action == "check":
        for name, base in (("облако", CLOUD), ("локальный", local)):
            ok, body = call(base, token, "getMe")
            print(f"{name:10} {base}\n           {'OK' if ok else 'недоступен'}: "
                  f"{body[:160]}")
        return

    if action == "to-local":
        print("Выхожу из облачного сервера…")
        ok, body = call(CLOUD, token, "logOut")
        print(("OK: " if ok else "Ответ: ") + body[:200])
        print("\nТеперь пропиши в .env:")
        print(f"  TELEGRAM_LOCAL_API_URL={local}")
        print(f"  TELEGRAM_LOCAL_FILES_DIR={ROOT / 'telegram-api' / 'data'}")
        print("и перезапусти бота.")
        return

    if action == "to-cloud":
        print("Выхожу из локального сервера…")
        ok, body = call(local, token, "logOut")
        print(("OK: " if ok else "Ответ: ") + body[:200])
        print("\nТеперь очисти в .env строки TELEGRAM_LOCAL_API_URL "
              "и TELEGRAM_LOCAL_FILES_DIR и перезапусти бота.")
        return

    sys.exit(f"Неизвестное действие: {action}. Ожидалось to-local | to-cloud | check")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
