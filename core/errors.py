"""Исключения пайплайна."""


class UserError(Exception):
    """Ошибка, текст которой показывается пользователю как есть (на русском)."""

    def __init__(self, message_ru: str):
        super().__init__(message_ru)
        self.message_ru = message_ru


class JobCancelled(Exception):
    """Задача отменена пользователем."""
