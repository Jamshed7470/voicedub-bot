"""Настройки пользователей (sqlite): язык озвучки и переключатели."""
from __future__ import annotations

import sqlite3
import threading

from core.config import DB_PATH, load_config

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.execute(
            """CREATE TABLE IF NOT EXISTS users (
                   user_id INTEGER PRIMARY KEY,
                   lang TEXT,
                   keep_background INTEGER,
                   keep_original INTEGER
               )"""
        )
        _conn.commit()
    return _conn


def get_user(user_id: int) -> dict:
    cfg = load_config()
    defaults = {
        "lang": None,
        "keep_background": bool(cfg.y("mix", "keep_background_default", default=True)),
        "keep_original": bool(cfg.y("mix", "keep_original_track_default", default=False)),
    }
    with _lock:
        row = _get_conn().execute(
            "SELECT lang, keep_background, keep_original FROM users WHERE user_id=?",
            (user_id,),
        ).fetchone()
    if not row:
        return defaults
    lang, bg, orig = row
    return {
        "lang": lang,
        "keep_background": defaults["keep_background"] if bg is None else bool(bg),
        "keep_original": defaults["keep_original"] if orig is None else bool(orig),
    }


def _upsert(user_id: int, **fields) -> None:
    current = get_user(user_id)
    current.update(fields)
    with _lock:
        conn = _get_conn()
        conn.execute(
            """INSERT INTO users (user_id, lang, keep_background, keep_original)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                   lang=excluded.lang,
                   keep_background=excluded.keep_background,
                   keep_original=excluded.keep_original""",
            (user_id, current["lang"],
             int(current["keep_background"]), int(current["keep_original"])),
        )
        conn.commit()


def set_lang(user_id: int, lang: str) -> None:
    _upsert(user_id, lang=lang)


def toggle_background(user_id: int) -> bool:
    new = not get_user(user_id)["keep_background"]
    _upsert(user_id, keep_background=new)
    return new


def toggle_original(user_id: int) -> bool:
    new = not get_user(user_id)["keep_original"]
    _upsert(user_id, keep_original=new)
    return new
