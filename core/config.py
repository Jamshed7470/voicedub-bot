"""Загрузка конфигурации: .env + configs/config.yaml."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "config.yaml"
DATA_DIR = ROOT / "data"
JOBS_DIR = DATA_DIR / "jobs"
LOGS_DIR = ROOT / "logs"
DB_PATH = DATA_DIR / "voicedub.sqlite3"

# Языки, поддерживаемые XTTS-v2 (код → название для клавиатуры)
TTS_LANGUAGES: dict[str, str] = {
    "ru": "🇷🇺 Русский",
    "en": "🇬🇧 English",
    "tr": "🇹🇷 Türkçe",
    "ar": "🇸🇦 العربية",
    "zh-cn": "🇨🇳 中文",
    "ja": "🇯🇵 日本語",
    "ko": "🇰🇷 한국어",
    "hi": "🇮🇳 हिन्दी",
    "es": "🇪🇸 Español",
    "fr": "🇫🇷 Français",
    "de": "🇩🇪 Deutsch",
    "it": "🇮🇹 Italiano",
    "pt": "🇵🇹 Português",
    "pl": "🇵🇱 Polski",
    "nl": "🇳🇱 Nederlands",
    "cs": "🇨🇿 Čeština",
    "hu": "🇭🇺 Magyar",
}


@dataclass
class Config:
    bot_token: str = ""
    hf_token: str = ""
    anthropic_api_key: str = ""
    xai_api_key: str = ""
    telegram_local_api_url: str = ""
    telegram_local_files_dir: str = ""  # где сервер складывает файлы (со стороны Windows)
    device: str = "cpu"
    profile: str = "full"
    yaml: dict[str, Any] = field(default_factory=dict)

    # ---- удобные доступы ----
    def y(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.yaml
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    @property
    def profile_cfg(self) -> dict[str, Any]:
        return self.y("profiles", self.profile, default={}) or {}

    @property
    def whisper_model(self) -> str:
        return self.profile_cfg.get("whisper_model", "large-v3")

    @property
    def nllb_model(self) -> str:
        return self.profile_cfg.get("nllb_model", "facebook/nllb-200-distilled-1.3B")

    @property
    def demucs_model(self) -> str:
        return self.profile_cfg.get("demucs_model", "htdemucs")

    @property
    def atempo_max(self) -> float:
        return float(self.y("timing", "atempo_max", default=1.35))

    @property
    def upload_limit_mb(self) -> float:
        """Предел размера отправки: 50 МБ через облако, ~2 ГБ через свой сервер."""
        if self.telegram_local_api_url:
            return float(self.y("limits", "telegram_upload_local_mb", default=1900))
        return float(self.y("limits", "telegram_upload_mb", default=50))

    @property
    def download_limit_mb(self) -> float:
        """Предел скачивания файла из Telegram: 20 МБ через облако."""
        if self.telegram_local_api_url:
            return float(self.y("limits", "telegram_download_local_mb", default=1900))
        return float(self.y("limits", "telegram_download_mb", default=20))

    @property
    def max_duration_s(self) -> float:
        """0 или отрицательное в конфиге — ограничения на длительность нет."""
        minutes = float(self.y("limits", "max_duration_minutes", default=0))
        return minutes * 60 if minutes > 0 else float("inf")

    @property
    def cache_max_source_s(self) -> float:
        """Дольше этого исходник не кладём в кэш медиа: там гигабайты wav."""
        minutes = float(self.y("cache", "max_source_minutes", default=30))
        return minutes * 60 if minutes > 0 else float("inf")

    @property
    def output_dir(self) -> Path:
        """Папка готовых видео внутри проекта (создаётся при первом обращении)."""
        name = str(self.y("output", "dir", default="готовые видео")).strip()
        path = ROOT / (name or "готовые видео")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def speech_rate(self, lang: str) -> float:
        rates = self.y("speech_rate_chars_per_s", default={}) or {}
        return float(rates.get(lang, 14))


def _resolve_device(requested: str) -> str:
    requested = (requested or "auto").strip().lower()
    if requested in ("cuda", "cpu"):
        return requested
    # auto
    try:
        import torch  # noqa: PLC0415 — тяжёлый импорт только при старте

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # torch может быть не установлен на этапе первичной настройки
        log.warning("PyTorch не найден или не смог определить GPU — работаю в режиме CPU.")
    return "cpu"


_config: Config | None = None


def load_config() -> Config:
    """Загружает .env и config.yaml. Кэшируется на процесс."""
    global _config
    if _config is not None:
        return _config

    load_dotenv(ROOT / ".env")

    with open(CONFIG_PATH, encoding="utf-8") as f:
        yaml_cfg = yaml.safe_load(f) or {}

    profile = os.getenv("MODEL_PROFILE", "full").strip().lower()
    if profile not in ("full", "light"):
        log.warning("Неизвестный MODEL_PROFILE=%r, использую full", profile)
        profile = "full"

    cfg = Config(
        bot_token=os.getenv("BOT_TOKEN", "").strip(),
        hf_token=os.getenv("HF_TOKEN", "").strip(),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
        xai_api_key=os.getenv("XAI_API_KEY", "").strip(),
        telegram_local_api_url=os.getenv("TELEGRAM_LOCAL_API_URL", "").strip(),
        telegram_local_files_dir=os.getenv("TELEGRAM_LOCAL_FILES_DIR", "").strip(),
        device=_resolve_device(os.getenv("DEVICE", "auto")),
        profile=profile,
        yaml=yaml_cfg,
    )

    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    _config = cfg
    return cfg
