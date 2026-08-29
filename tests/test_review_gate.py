"""Тесты точки проверки: пауза перед рендером и её отключение.

Два сценария, которые обязаны работать одинаково надёжно:
* студия включена → рендер ждёт нажатия «Утвердить» (INV-5);
* студия выключена → бот работает как раньше (INV-6), но уже с
  зафиксированными профилями голоса.
"""
from __future__ import annotations

import asyncio

import pytest

from bot import review
from project import store
from project.schema import (Project, Reference, Segment, Speaker, Stage, Voice)

JOB = "reviewjob"


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "JOBS_DIR", tmp_path / "jobs")
    (tmp_path / "jobs").mkdir()

    proj = Project(job_id=JOB, owner_telegram_id=7, lang_src="tr", lang_tgt="ru",
                   stage=Stage.REVIEW)
    proj.speakers = [
        Speaker(id="S1", gender="male", label="Спикер 1",
                reference=Reference(clean_sec=20, clone_allowed=True),
                voice=Voice(mode="clone", locked=True)),
        Speaker(id="S2", gender="female", label="Спикер 2",
                reference=Reference(clean_sec=3, clone_allowed=False),
                voice=Voice(mode="preset", preset_id="v1",
                            preset_name="Женский мягкий", locked=True)),
    ]
    proj.segments = [
        Segment(id=1, start=0, end=2, speaker_id="S1", text_tgt="Раз"),
        Segment(id=2, start=3, end=5, speaker_id="S2", text_tgt="Два",
                flags=["low_speaker_conf"]),
    ]
    proj.recompute_stats()
    store.save(proj)
    return store.load(JOB)


class FakeCfg:
    studio_secret = "s" * 40
    studio_link_ttl_h = 72

    def __init__(self, on=True):
        self._on = on

    @property
    def studio_on(self):
        return self._on

    @property
    def studio_url(self):
        return "http://localhost:8080"


# ------------------------------------------------------------------ ссылка

def test_studio_link_contains_token_and_job():
    link = review.make_link(JOB, 7, FakeCfg())
    assert link.startswith("http://localhost:8080/studio/" + JOB)
    assert "?t=" in link


def test_no_secret_no_link(caplog):
    class NoSecret(FakeCfg):
        studio_secret = ""

    assert review.make_link(JOB, 7, NoSecret()) is None


def test_summary_mentions_speakers_and_warnings(project):
    text = review.summary_text(project)
    assert "Спикеров: <b>2</b>" in text
    assert "♂1 ♀1" in text
    assert "1</b> реплик" in text          # одно замечание
    assert "1 спикеров" in text            # один без клона


# ------------------------------------------------------------------ ожидание

@pytest.mark.asyncio
async def test_wait_for_approval_returns_on_event(project, monkeypatch):
    """Утверждение из того же процесса будит ожидание сразу."""
    monkeypatch.setattr(review, "POLL_SEC", 0.05)

    async def approve_soon():
        await asyncio.sleep(0.1)
        review.mark_approved(JOB)
        await review.enqueue_approved(JOB)

    task = asyncio.create_task(approve_soon())
    ok = await asyncio.wait_for(review.wait_for_approval(JOB), timeout=5)
    await task
    assert ok is True
    assert store.load(JOB).stage == Stage.APPROVED


@pytest.mark.asyncio
async def test_wait_for_approval_sees_marker_from_other_process(project, monkeypatch):
    """Студия работает отдельным процессом: связь идёт через отметку на диске."""
    monkeypatch.setattr(review, "POLL_SEC", 0.05)
    (store.job_dir(JOB) / "approved.flag").write_text("studio", encoding="utf-8")
    assert await asyncio.wait_for(review.wait_for_approval(JOB), timeout=5) is True


@pytest.mark.asyncio
async def test_wait_for_approval_times_out(project, monkeypatch):
    """Не утвердили — задача не висит вечно."""
    monkeypatch.setattr(review, "POLL_SEC", 0.02)
    ok = await asyncio.wait_for(
        review.wait_for_approval(JOB, timeout_hours=0.0002), timeout=5)
    assert ok is False


@pytest.mark.asyncio
async def test_cancelled_project_stops_waiting(project, monkeypatch):
    monkeypatch.setattr(review, "POLL_SEC", 0.05)
    store.update(JOB, lambda p: setattr(p, "stage", Stage.CANCELLED))
    assert await asyncio.wait_for(review.wait_for_approval(JOB), timeout=5) is False


# ------------------------------------------------------------------ карта голосов

def test_voice_map_lists_voices_and_stability(project):
    report = {"S1": {"passed": 10, "segments": 10},
              "S2": {"passed": 8, "segments": 10}}
    text = review.voice_map_text(project, report, overall=0.81)
    assert "клон оригинала" in text
    assert "Женский мягкий" in text
    assert "QC 10/10" in text
    assert "Стабильность голосов: 0.81" in text
    assert "голоса стабильны" in text


def test_voice_map_norm_depends_on_language(project):
    """Норма стабильности зависит от того, совпадает ли язык озвучки.

    Замер: материал с заведомо одним голосом даёт 0.71 при озвучке на
    языке оригинала и 0.47 при озвучке на другом. Один и тот же 0.60 —
    хороший результат для дубляжа с турецкого и плохой для переозвучки
    без смены языка.
    """
    cross = review.voice_map_text(project, {}, overall=0.60)   # tr → ru
    assert "голоса стабильны" in cross
    assert "норма от 0.60" in cross

    store.update(JOB, lambda p: setattr(p, "lang_src", p.lang_tgt))
    same = review.voice_map_text(store.load(JOB), {}, overall=0.60)
    assert "заметный разброс" in same
    assert "норма от 0.75" in same


def test_voice_map_warns_on_low_stability(project):
    text = review.voice_map_text(project, {}, overall=0.31)
    assert "заметный разброс" in text


def test_voice_map_truncates_long_cast(project):
    """Полсотни спикеров не должны превратить сообщение в простыню."""
    store.update(JOB, lambda p: p.speakers.extend(
        Speaker(id=f"S{i}", label=f"Спикер {i}") for i in range(3, 25)))
    text = review.voice_map_text(store.load(JOB), {}, overall=0.8)
    assert "и ещё" in text
    assert len(text.split("\n")) < 25


# ------------------------------------------------------ ссылка и её отправка

def test_localhost_link_cannot_be_a_button():
    """Telegram отвергает localhost в кнопке — это надо знать заранее.

    Иначе отказ приходит на отправку сообщения, то есть в самом конце
    разбора, и роняет задачу вместе со всей проделанной работой.
    """
    assert not review.link_is_button_safe("http://localhost:8080/studio/x?t=1")
    assert not review.link_is_button_safe("http://127.0.0.1:8080/studio/x")
    assert not review.link_is_button_safe(None)
    assert not review.link_is_button_safe("ftp://example.com/x")
    assert review.link_is_button_safe("http://192.168.0.10:8080/studio/x")
    assert review.link_is_button_safe("https://dub.example.com/studio/x")


def test_keyboard_omits_unusable_link(project):
    """Кнопка «Открыть студию» не создаётся для непригодного адреса."""
    local = review.review_keyboard(JOB, "http://localhost:8080/studio/x")
    texts = [b.text for row in local.inline_keyboard for b in row]
    assert not any("Открыть студию" in t for t in texts)
    assert any("Утвердить" in t for t in texts)

    public = review.review_keyboard(JOB, "https://dub.example.com/studio/x")
    texts = [b.text for row in public.inline_keyboard for b in row]
    assert any("Открыть студию" in t for t in texts)


@pytest.mark.asyncio
async def test_local_link_goes_into_message_text(project):
    """Раз кнопкой нельзя — ссылка должна быть в тексте, её скопируют."""
    sent = []

    class Bot:
        async def send_message(self, chat_id, text, **kw):
            sent.append(text)

    class Job:
        user_id, chat_id = 7, 7

    await review.request_review(Job(), Bot(), FakeCfg(), project)
    assert sent, "сообщение не отправлено"
    assert "localhost:8080/studio/" in sent[0]
    assert "Открыть студию" in sent[0]


@pytest.mark.asyncio
async def test_send_failure_does_not_lose_the_job(project, caplog):
    """Сорвавшаяся отправка не обесценивает разбор: пробуем без кнопок."""
    attempts = []

    class Bot:
        async def send_message(self, chat_id, text, **kw):
            attempts.append(kw.get("reply_markup"))
            if kw.get("reply_markup") is not None:
                raise RuntimeError("Bad Request: inline keyboard button URL is invalid")

    class Job:
        user_id, chat_id = 7, 7

    await review.request_review(Job(), Bot(), FakeCfg(), project)
    assert len(attempts) == 2, "повтор без клавиатуры не выполнен"
    assert attempts[1] is None


@pytest.mark.asyncio
async def test_send_failure_twice_is_survived(project):
    """Даже если Telegram недоступен вовсе — исключение наружу не уходит."""
    class Bot:
        async def send_message(self, *a, **kw):
            raise RuntimeError("сеть недоступна")

    class Job:
        user_id, chat_id = 7, 7

    await review.request_review(Job(), Bot(), FakeCfg(), project)   # без падения


def test_lan_url_replaces_localhost():
    """Для телефона нужен адрес машины: localhost там указывает на телефон."""
    alt = review.lan_url("http://localhost:8080")
    assert alt is None or (alt.startswith("http://") and "localhost" not in alt)
    assert review.lan_url("https://dub.example.com") is None


# ------------------------------------------- сводка не роняет готовую работу

def test_summary_survives_missing_fields():
    """Словарь профиля собирается в трёх местах — поля может не быть.

    Раньше отсутствие gender_confidence роняло задачу, у которой готовое
    видео уже лежало на диске: исключение возникало ПОСЛЕ сохранения
    файла, но ДО возврата результата, и очередь считала задачу упавшей.
    """
    from core.pipeline import _build_summary

    # минимальный словарь — ровно такой пришёл из студии
    text = _build_summary("en", "ru", {"S1": {"id": "S1", "gender": "male"}}, 125)
    assert "Готово" in text and "S1" in text and "мужчина" in text

    # совсем пустой словарь тоже не роняет
    assert "Готово" in _build_summary("en", "ru", {"S9": {}}, 5)


def test_summary_shows_voice_source():
    """В сводке видно, каким голосом озвучен спикер."""
    from core.pipeline import _build_summary

    bank = _build_summary("en", "ru", {
        "S1": {"id": "S1", "gender": "male", "gender_confidence": 0.91,
               "bank_voice": "Никита"}}, 60)
    assert "голос: Никита" in bank and "91%" in bank

    clone = _build_summary("en", "ru", {
        "S2": {"id": "S2", "gender": "female", "voice": {"mode": "clone"}}}, 60)
    assert "клон оригинала" in clone
