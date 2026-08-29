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
                   keep_original INTEGER,
                   style TEXT
               )"""
        )
        try:  # миграция старой базы без колонки style
            _conn.execute("ALTER TABLE users ADD COLUMN style TEXT")
        except sqlite3.OperationalError:
            pass
        try:  # миграция: режим голосов (clone | bank)
            _conn.execute("ALTER TABLE users ADD COLUMN voice TEXT")
        except sqlite3.OperationalError:
            pass
        try:  # миграция: подсказка о числе спикеров (auto | 2..8 | 10)
            _conn.execute("ALTER TABLE users ADD COLUMN speakers TEXT")
        except sqlite3.OperationalError:
            pass
        _conn.commit()
    return _conn


def get_user(user_id: int) -> dict:
    cfg = load_config()
    defaults = {
        "lang": None,
        "keep_background": bool(cfg.y("mix", "keep_background_default", default=True)),
        "keep_original": bool(cfg.y("mix", "keep_original_track_default", default=False)),
        "style": "normal",
        "voice": "auto",
        "speakers": "auto",
    }
    with _lock:
        row = _get_conn().execute(
            "SELECT lang, keep_background, keep_original, style, voice, speakers "
            "FROM users WHERE user_id=?",
            (user_id,),
        ).fetchone()
    if not row:
        return defaults
    lang, bg, orig, style, voice, speakers = row
    return {
        "lang": lang,
        "keep_background": defaults["keep_background"] if bg is None else bool(bg),
        "keep_original": defaults["keep_original"] if orig is None else bool(orig),
        "style": style or "normal",
        "voice": {"bank": "preset"}.get(voice, voice) or "auto",
        "speakers": speakers or "auto",
    }


def _upsert(user_id: int, **fields) -> None:
    current = get_user(user_id)
    current.update(fields)
    with _lock:
        conn = _get_conn()
        conn.execute(
            """INSERT INTO users (user_id, lang, keep_background, keep_original,
                                  style, voice, speakers)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                   lang=excluded.lang,
                   keep_background=excluded.keep_background,
                   keep_original=excluded.keep_original,
                   style=excluded.style,
                   voice=excluded.voice,
                   speakers=excluded.speakers""",
            (user_id, current["lang"],
             int(current["keep_background"]), int(current["keep_original"]),
             current["style"], current["voice"], current["speakers"]),
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


def toggle_style(user_id: int) -> str:
    new = "street" if get_user(user_id)["style"] == "normal" else "normal"
    _upsert(user_id, style=new)
    return new


def set_voice_mode(user_id: int, mode: str) -> str:
    """Режим голосов: clone | preset | auto (bank — старое имя preset)."""
    mode = {"bank": "preset"}.get(mode, mode)
    if mode not in ("clone", "preset", "auto"):
        mode = "auto"
    _upsert(user_id, voice=mode)
    return mode


def toggle_voice(user_id: int) -> str:
    """Кнопка в настройках перебирает режимы по кругу."""
    order = ["auto", "clone", "preset"]
    current = get_user(user_id)["voice"]
    new = order[(order.index(current) + 1) % len(order)] if current in order else "auto"
    _upsert(user_id, voice=new)
    return new


def set_speakers(user_id: int, value: str) -> None:
    _upsert(user_id, speakers=value)
