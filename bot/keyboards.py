"""Инлайн-клавиатуры бота."""
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from core.config import TTS_LANGUAGES


def lang_keyboard() -> InlineKeyboardMarkup:
    """Выбор целевого языка (только языки, поддерживаемые XTTS)."""
    rows, row = [], []
    for code, name in TTS_LANGUAGES.items():
        row.append(InlineKeyboardButton(text=name, callback_data=f"lang:{code}"))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return InlineKeyboardMarkup(inline_keyboard=rows)


def settings_keyboard(keep_background: bool, keep_original: bool,
                      style: str = "normal") -> InlineKeyboardMarkup:
    from bot import texts
    bg = texts.SETTINGS_BG_ON if keep_background else texts.SETTINGS_BG_OFF
    orig = texts.SETTINGS_ORIG_ON if keep_original else texts.SETTINGS_ORIG_OFF
    st = (texts.SETTINGS_STYLE_STREET if style == "street"
          else texts.SETTINGS_STYLE_NORMAL)
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=bg, callback_data="set:bg")],
        [InlineKeyboardButton(text=orig, callback_data="set:orig")],
        [InlineKeyboardButton(text=st, callback_data="set:style")],
    ])


def same_lang_keyboard(job_id: str) -> InlineKeyboardMarkup:
    from bot import texts
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text=texts.SAME_LANG_CONTINUE,
                             callback_data=f"samelang:yes:{job_id}"),
        InlineKeyboardButton(text=texts.SAME_LANG_CANCEL,
                             callback_data=f"samelang:no:{job_id}"),
    ]])
