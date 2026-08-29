"""Подписанные ссылки на проект.

Токен несёт job_id и telegram-id владельца и подписан секретом. Ссылка
уходит в Telegram, поэтому у неё есть срок жизни: попавшая в чужие руки
переписка не должна открывать проект вечно.

Проверяются обе части: подпись (ссылку не подделали) и владелец (проект
свой). Одного job_id мало — он предсказуем.
"""
from __future__ import annotations

import logging
import secrets

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

log = logging.getLogger(__name__)

SALT = "voicedub-studio-link"


class AuthError(RuntimeError):
    """Токен не прошёл проверку."""


def generate_secret() -> str:
    return secrets.token_urlsafe(48)


def _serializer(secret: str) -> URLSafeTimedSerializer:
    if not secret or len(secret) < 32:
        raise AuthError(
            "STUDIO_SECRET не задан или короче 32 символов. Сгенерируйте:\n"
            '  python -c "import secrets; print(secrets.token_urlsafe(48))"\n'
            "и впишите в .env как STUDIO_SECRET=…")
    return URLSafeTimedSerializer(secret, salt=SALT)


def make_token(job_id: str, owner_telegram_id: int, secret: str) -> str:
    return _serializer(secret).dumps({"job": job_id, "uid": int(owner_telegram_id)})


def read_token(token: str, secret: str, max_age_hours: int = 72) -> dict:
    """Проверяет подпись и срок. Бросает AuthError с понятным текстом."""
    try:
        return _serializer(secret).loads(token, max_age=max_age_hours * 3600)
    except SignatureExpired as e:
        raise AuthError("Ссылка на студию устарела. Пришлите боту /review, "
                        "чтобы получить новую.") from e
    except BadSignature as e:
        raise AuthError("Ссылка на студию недействительна.") from e


def check(token: str, job_id: str, secret: str, max_age_hours: int = 72) -> dict:
    """Токен должен быть выдан именно на этот проект."""
    data = read_token(token, secret, max_age_hours)
    if data.get("job") != job_id:
        raise AuthError("Ссылка выдана на другой проект.")
    return data


def studio_link(base_url: str, job_id: str, owner_telegram_id: int,
                secret: str) -> str:
    token = make_token(job_id, owner_telegram_id, secret)
    return f"{base_url.rstrip('/')}/studio/{job_id}?t={token}"
