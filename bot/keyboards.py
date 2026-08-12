"""Клавиатуры бота: постоянное меню под полем ввода и инлайн-меню."""
from aiogram.types import (InlineKeyboardButton, InlineKeyboardMarkup,
                           KeyboardButton, ReplyKeyboardMarkup)

from core.config import TTS_LANGUAGES


def main_menu() -> ReplyKeyboardMarkup:
    """Постоянное меню: весь функционал под рукой, без набора команд."""
    from bot import texts
    rows = [
        [texts.BTN_LANG, texts.BTN_SETTINGS],
        [texts.BTN_STYLE, texts.BTN_VOICE],
        [texts.BTN_SPEAKERS, texts.BTN_STATUS],
        [texts.BTN_CANCEL, texts.BTN_HELP],
    ]
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=t) for t in row] for row in rows],
        resize_keyboard=True,
        is_persistent=True,
        input_field_placeholder=texts.INPUT_PLACEHOLDER,
    )


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


SPEAKER_CHOICES = ["auto", "2", "3", "4", "5", "6", "8", "10"]


def settings_keyboard(keep_background: bool, keep_original: bool,
                      style: str = "normal", voice: str = "clone",
                      speakers: str = "auto") -> InlineKeyboardMarkup:
    from bot import texts
    bg = texts.SETTINGS_BG_ON if keep_background else texts.SETTINGS_BG_OFF
    orig = texts.SETTINGS_ORIG_ON if keep_original else texts.SETTINGS_ORIG_OFF
    st = (texts.SETTINGS_STYLE_STREET if style == "street"
          else texts.SETTINGS_STYLE_NORMAL)
    vc = (texts.SETTINGS_VOICE_BANK if voice == "bank"
          else texts.SETTINGS_VOICE_CLONE)
    spk = texts.SETTINGS_SPEAKERS.format(
        value=_speaker_label(speakers))
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=bg, callback_data="set:bg")],
        [InlineKeyboardButton(text=orig, callback_data="set:orig")],
        [InlineKeyboardButton(text=st, callback_data="set:style")],
        [InlineKeyboardButton(text=vc, callback_data="set:voice")],
        [InlineKeyboardButton(text=spk, callback_data="set:speakers")],
    ])


def _speaker_label(value: str) -> str:
    from bot import texts
    if value == "auto":
        return texts.SPEAKERS_AUTO
    return "10+" if value == "10" else value


def speakers_keyboard(current: str = "auto") -> InlineKeyboardMarkup:
    """Выбор числа говорящих: подсказка для диаризации."""
    rows, row = [], []
    for value in SPEAKER_CHOICES:
        mark = "✅ " if value == current else ""
        row.append(InlineKeyboardButton(text=f"{mark}{_speaker_label(value)}",
                                        callback_data=f"spk:{value}"))
        if len(row) == 4:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="spk:back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def same_lang_keyboard(job_id: str) -> InlineKeyboardMarkup:
    from bot import texts
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text=texts.SAME_LANG_CONTINUE,
                             callback_data=f"samelang:yes:{job_id}"),
        InlineKeyboardButton(text=texts.SAME_LANG_CANCEL,
                             callback_data=f"samelang:no:{job_id}"),
    ]])
