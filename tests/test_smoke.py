"""Smoke-тест: 30-секундный сэмпл проходит весь пайплайн (MODEL_PROFILE=light).

Требует установленных ML-зависимостей, весов моделей и HF_TOKEN, поэтому
по умолчанию пропускается. Запуск:

    set VOICEDUB_SMOKE=1        (Windows)
    export VOICEDUB_SMOKE=1     (Linux)
    pytest tests/test_smoke.py -s
"""
import asyncio
import os
import shutil
import subprocess

import pytest

pytestmark = [
    pytest.mark.skipif(os.getenv("VOICEDUB_SMOKE") != "1",
                       reason="запустите с VOICEDUB_SMOKE=1 (нужны веса моделей)"),
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg"),
]


def make_sample(path) -> None:
    """30-секундный сэмпл: тон + синтетическая «речь» недоступна без записи,
    поэтому используем espeak-совместимый вариант — просто скачивание не нужно:
    генерируем видео с тестовым тоном и тишиной. Для реальной проверки речи
    подставьте свой файл через VOICEDUB_SMOKE_FILE."""
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", "testsrc=duration=30:size=320x240:rate=15",
         "-f", "lavfi", "-i", "sine=frequency=220:duration=30",
         "-c:v", "libx264", "-c:a", "aac", str(path)],
        check=True, capture_output=True,
    )


def test_full_pipeline_light(tmp_path):
    os.environ["MODEL_PROFILE"] = "light"

    from core.config import load_config
    from core.errors import UserError
    from core.pipeline import PipelineHooks, run_job

    cfg = load_config()

    sample = os.getenv("VOICEDUB_SMOKE_FILE")
    src = tmp_path / "sample.mp4"
    if sample and os.path.exists(sample):
        shutil.copy(sample, src)
    else:
        make_sample(src)

    class FakeJob:
        id = "smoke_test"
        user_id = 0
        chat_id = 0
        kind = "url"  # не используется: подменяем скачивание копией файла
        payload = str(src)
        target_lang = "ru"
        settings = {"keep_background": True, "keep_original_track": False}

    async def report(stage, label, pct):
        print(f"этап {stage}/10: {label} {pct}%")

    async def run():
        import core.downloader as dl
        dl.download_url = lambda url, dest_dir, cfg: shutil.copy(url, dest_dir / "input.mp4") or (dest_dir / "input.mp4")
        hooks = PipelineHooks(report=report, confirm_same_lang=None,
                              cancel_event=asyncio.Event())
        return await run_job(FakeJob(), None, hooks, cfg)

    try:
        result = asyncio.run(run())
        assert result.output_path.exists()
        print(result.summary)
    except UserError as e:
        # для синтетического сэмпла без речи корректный исход —
        # понятная ошибка «нет речи»
        assert "реч" in e.message_ru.lower()
