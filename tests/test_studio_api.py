"""Тесты API студии: авторизация, конфликты версий, сохранность правок.

Проверяется главное обещание Фазы 2: правка, сделанная человеком, не
теряется — ни при повторном рендере, ни при повторной диаризации, ни при
одновременном редактировании из двух окон.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from project import store
from project.schema import (Project, Reference, Segment, Speaker, Stage,
                            Voice, color_for)
from studio import auth

SECRET = "x" * 48
JOB = "testjob01"


# ------------------------------------------------------------------ фикстуры

@pytest.fixture
def project(tmp_path, monkeypatch):
    """Проект из двух спикеров и четырёх реплик в изолированной папке."""
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    monkeypatch.setattr(store, "JOBS_DIR", jobs)

    proj = Project(
        job_id=JOB, owner_telegram_id=42, lang_src="tr", lang_tgt="ru",
        stage=Stage.REVIEW,
        speakers=[
            # цвета разные, как их выдаёт пайплайн: на таймлайне спикеров
            # различают именно по цвету
            Speaker(id="S1", label="Спикер 1", gender="male",
                    color=color_for(0),
                    reference=Reference(clean_sec=20.0, clone_allowed=True,
                                        best_samples=[1, 2]),
                    voice=Voice(mode="clone", locked=True,
                                profile_path="p.pt")),
            Speaker(id="S2", label="Спикер 2", gender="female",
                    color=color_for(1),
                    reference=Reference(clean_sec=4.0, clone_allowed=False),
                    voice=Voice(mode="preset", preset_id="v1", locked=True)),
        ],
        segments=[
            Segment(id=1, start=0.0, end=3.0, speaker_id="S1",
                    text_tgt="Первая реплика", text_tts="Первая реплика",
                    speaker_confidence=0.9),
            Segment(id=2, start=4.0, end=6.0, speaker_id="S1",
                    text_tgt="Вторая реплика", text_tts="Вторая реплика",
                    flags=["low_speaker_conf"], speaker_confidence=0.4),
            Segment(id=3, start=7.0, end=10.0, speaker_id="S2",
                    text_tgt="Третья реплика", text_tts="Третья реплика"),
            Segment(id=4, start=11.0, end=13.0, speaker_id="S2",
                    text_tgt="Четвёртая", text_tts="Четвёртая"),
        ],
    )
    proj.recompute_stats()
    store.save(proj)
    return store.load(JOB)


@pytest.fixture
def client(project, monkeypatch):
    from core.config import Config

    cfg = Config(studio_enabled=True, studio_secret=SECRET,
                 studio_public_url="http://localhost:8080",
                 yaml={"studio": {"enabled": True, "link_ttl_hours": 72}})
    monkeypatch.setattr("core.config._config", cfg, raising=False)
    monkeypatch.setattr("core.config.load_config", lambda: cfg)
    monkeypatch.setattr("studio.api.load_config", lambda: cfg)
    monkeypatch.setattr("studio.server.load_config", lambda: cfg)

    from studio.server import create_app

    return TestClient(create_app())


@pytest.fixture
def token():
    return auth.make_token(JOB, 42, SECRET)


def headers(token: str, version: int | None = None) -> dict:
    h = {"Authorization": f"Bearer {token}"}
    if version is not None:
        h["If-Match"] = str(version)
    return h


# ------------------------------------------------------------------ доступ

def test_api_requires_token(client):
    assert client.get(f"/api/projects/{JOB}").status_code == 401


def test_api_auth_expired_token_401(client, monkeypatch):
    """Ссылка старше срока жизни не открывает проект.

    Токен выпускается «в прошлом» подменой часов itsdangerous — ждать
    трое суток тест не может, а max_age=0 просроченным свежий токен не
    делает: там сравнение строгое.
    """
    import itsdangerous.timed as timed

    four_days_ago = timed.time.time() - 4 * 24 * 3600
    monkeypatch.setattr(timed.time, "time", lambda: four_days_ago)
    old = auth.make_token(JOB, 42, SECRET)
    monkeypatch.undo()

    r = client.get(f"/api/projects/{JOB}", headers=headers(old))
    assert r.status_code == 401
    assert "устарела" in r.json()["error"]


def test_api_token_for_other_project_rejected(client):
    other = auth.make_token("someotherjob", 42, SECRET)
    assert client.get(f"/api/projects/{JOB}", headers=headers(other)).status_code == 401


def test_api_bad_signature_rejected(client):
    forged = auth.make_token(JOB, 42, "y" * 48)
    assert client.get(f"/api/projects/{JOB}", headers=headers(forged)).status_code == 401


def test_get_project(client, token):
    r = client.get(f"/api/projects/{JOB}", headers=headers(token))
    assert r.status_code == 200
    data = r.json()
    assert len(data["speakers"]) == 2
    assert len(data["segments"]) == 4
    assert data["warnings_count"] == 1
    assert data["stage"] == "review"


# ------------------------------------------------------------------ правки

def test_api_patch_segment_persists(client, token, project):
    r = client.patch(f"/api/projects/{JOB}/segments/2",
                     json={"speaker_id": "S2"},
                     headers=headers(token, project.version))
    assert r.status_code == 200, r.text
    assert r.json()["speaker_id"] == "S2"

    saved = store.load(JOB)
    seg = saved.segment(2)
    assert seg.speaker_id == "S2"
    assert "speaker_id" in seg.edited_by_user.fields
    # флаг неуверенности снят: человек посмотрел и решил
    assert "low_speaker_conf" not in seg.flags
    # статистика спикеров пересчитана, а не осталась старой
    assert saved.speaker("S1").stats.segments_count == 1
    assert saved.speaker("S2").stats.segments_count == 3


def test_api_patch_text_marks_synth_stale(client, token, project):
    client.patch(f"/api/projects/{JOB}/segments/1",
                 json={"text_tgt": "Совсем другой текст"},
                 headers=headers(token, project.version))
    seg = store.load(JOB).segment(1)
    assert seg.text_tgt == "Совсем другой текст"
    assert seg.text_tts == "Совсем другой текст"
    assert seg.synth.status == "pending", "старая озвучка не соответствует тексту"


def test_api_version_conflict_409(client, token, project):
    ok = client.patch(f"/api/projects/{JOB}/segments/1",
                      json={"text_tgt": "Правка из первого окна"},
                      headers=headers(token, project.version))
    assert ok.status_code == 200

    stale = client.patch(f"/api/projects/{JOB}/segments/2",
                         json={"text_tgt": "Правка из второго окна"},
                         headers=headers(token, project.version))
    assert stale.status_code == 409
    assert stale.json()["current_version"] > project.version
    # первая правка не затёрта
    assert store.load(JOB).segment(1).text_tgt == "Правка из первого окна"


def test_api_missing_if_match_rejected(client, token):
    r = client.patch(f"/api/projects/{JOB}/segments/1",
                     json={"text_tgt": "без версии"}, headers=headers(token))
    assert r.status_code == 428


def test_api_unknown_speaker_rejected(client, token, project):
    r = client.patch(f"/api/projects/{JOB}/segments/1",
                     json={"speaker_id": "S99"},
                     headers=headers(token, project.version))
    assert r.status_code == 400


def test_api_bulk_assign(client, token, project):
    r = client.post(f"/api/projects/{JOB}/segments/bulk",
                    json={"segment_ids": [3, 4], "speaker_id": "S1"},
                    headers=headers(token, project.version))
    assert r.status_code == 200
    saved = store.load(JOB)
    assert all(saved.segment(i).speaker_id == "S1" for i in (3, 4))
    assert saved.speaker("S1").stats.segments_count == 4


# ------------------------------------------------------------------ спикеры

def test_api_merge_speakers(client, token, project):
    r = client.post(f"/api/projects/{JOB}/speakers/merge",
                    json={"from_id": "S2", "into_id": "S1"},
                    headers=headers(token, project.version))
    assert r.status_code == 200, r.text
    assert r.json()["moved_segments"] == 2
    assert r.json()["rebuild_required"] is True

    saved = store.load(JOB)
    assert all(s.speaker_id == "S1" for s in saved.segments)
    assert saved.speaker("S2").merged_into == "S1"
    assert "S2" in saved.speaker("S1").merged_from
    # профиль принимающего собран по другому набору реплик — он разблокирован
    assert saved.speaker("S1").voice.locked is False
    assert len(saved.active_speakers()) == 1


def test_api_merge_speaker_into_itself_rejected(client, token, project):
    r = client.post(f"/api/projects/{JOB}/speakers/merge",
                    json={"from_id": "S1", "into_id": "S1"},
                    headers=headers(token, project.version))
    assert r.status_code == 400


def test_api_rename_and_set_gender(client, token, project):
    r = client.patch(f"/api/projects/{JOB}/speakers/S1",
                     json={"name": "Ведущий", "gender": "female"},
                     headers=headers(token, project.version))
    assert r.status_code == 200
    saved = store.load(JOB).speaker("S1")
    assert saved.name == "Ведущий"
    assert saved.gender == "female"
    assert saved.gender_edited_by_user is True


# ------------------------------------------------------------------ сегменты

def test_api_split_segment_text_boundary(client, token, project):
    r = client.post(f"/api/projects/{JOB}/segments/1/split",
                    json={"at_sec": 1.5},
                    headers=headers(token, project.version))
    assert r.status_code == 200, r.text
    left, right = r.json()
    assert left["end"] == 1.5 and right["start"] == 1.5
    # текст делится по границе слова, а не по символу
    assert left["text_tgt"] and right["text_tgt"]
    assert not left["text_tgt"].endswith(" ")
    assert (left["text_tgt"] + " " + right["text_tgt"]) == "Первая реплика"
    assert len(store.load(JOB).segments) == 5


def test_api_split_outside_segment_rejected(client, token, project):
    r = client.post(f"/api/projects/{JOB}/segments/1/split",
                    json={"at_sec": 99.0},
                    headers=headers(token, project.version))
    assert r.status_code == 400


def test_api_merge_segments(client, token, project):
    r = client.post(f"/api/projects/{JOB}/segments/merge",
                    json={"segment_ids": [1, 2]},
                    headers=headers(token, project.version))
    assert r.status_code == 200, r.text
    saved = store.load(JOB)
    assert len(saved.segments) == 3
    merged = saved.segment(1)
    assert merged.end == 6.0
    assert merged.text_tgt == "Первая реплика Вторая реплика"


def test_api_merge_segments_different_speakers_rejected(client, token, project):
    r = client.post(f"/api/projects/{JOB}/segments/merge",
                    json={"segment_ids": [2, 3]},
                    headers=headers(token, project.version))
    assert r.status_code == 400


# ------------------------------------------------------------------ отмена

def test_api_undo_restores_previous_state(client, token, project):
    client.patch(f"/api/projects/{JOB}/segments/2", json={"speaker_id": "S2"},
                 headers=headers(token, project.version))
    v = store.load(JOB).version
    r = client.post(f"/api/projects/{JOB}/undo", headers=headers(token, v))
    assert r.status_code == 200
    assert store.load(JOB).segment(2).speaker_id == "S1"


def test_api_undo_merge_segments(client, token, project):
    client.post(f"/api/projects/{JOB}/segments/merge",
                json={"segment_ids": [1, 2]},
                headers=headers(token, project.version))
    v = store.load(JOB).version
    client.post(f"/api/projects/{JOB}/undo", headers=headers(token, v))
    saved = store.load(JOB)
    assert len(saved.segments) == 4, "склеенная реплика не вернулась"
    assert saved.segment(2) is not None


def test_api_undo_nothing_to_undo(client, token, project):
    r = client.post(f"/api/projects/{JOB}/undo",
                    headers=headers(token, project.version))
    assert r.status_code == 400


# ------------------------------------------------------------------ стадии

def test_render_blocked_in_review(client, token, project):
    """Пока проект не утверждён, рендера нет (INV-5)."""
    assert store.load(JOB).stage == Stage.REVIEW
    r = client.post(f"/api/projects/{JOB}/approve",
                    headers=headers(token, project.version))
    assert r.status_code == 200
    assert store.load(JOB).stage == Stage.APPROVED
    # повторное утверждение уже неуместно
    v = store.load(JOB).version
    again = client.post(f"/api/projects/{JOB}/approve", headers=headers(token, v))
    assert again.status_code == 409


def test_approve_writes_marker(client, token, project):
    client.post(f"/api/projects/{JOB}/approve",
                headers=headers(token, project.version))
    assert (store.job_dir(JOB) / "approved.flag").exists()


# ------------------------------------------------------------------ экспорт

def test_export_srt(client, token):
    r = client.get(f"/api/projects/{JOB}/export/srt", headers=headers(token))
    assert r.status_code == 200
    assert "00:00:00,000 --> 00:00:03,000" in r.text
    assert "Первая реплика" in r.text


def test_media_missing_gives_clear_message(client, token):
    r = client.get(f"/api/projects/{JOB}/media/video", headers=headers(token))
    assert r.status_code == 404
    assert "7 дней" in r.json()["error"]


# ------------------------------------------------------------------ хранилище

def test_store_atomic_write_keeps_valid_json(project, tmp_path):
    """Файл всегда остаётся валидным JSON — запись идёт через переименование."""
    for i in range(5):
        store.update(JOB, lambda p, i=i: setattr(p.segment(1), "text_tgt", f"v{i}"))
    raw = json.loads(store.project_path(JOB).read_text(encoding="utf-8"))
    assert raw["segments"][0]["text_tgt"] == "v4"
    assert raw["version"] == project.version + 5


def test_store_version_conflict(project):
    with pytest.raises(store.VersionConflict):
        store.update(JOB, lambda p: None, expected_version=project.version + 99)


def test_from_pipeline_preserves_user_edits(project):
    """Повторный прогон пайплайна не затирает правки человека (INV-3)."""
    store.update(JOB, lambda p: (
        p.segment(2).edited_by_user.touch("speaker_id"),
        setattr(p.segment(2), "speaker_id", "S2"),
        setattr(p.speaker("S1"), "name", "Ведущий"),
    ))

    # пайплайн пересчитал и снова считает, что реплика 2 принадлежит S1
    segments = [{"id": s.id, "start": s.start, "end": s.end,
                 "speaker": "S1" if s.id <= 2 else "S2", "text": s.text_tgt}
                for s in store.load(JOB).segments]
    speakers = {"S1": {"first_sec": 0.0, "gender": "male",
                       "reference": {}, "voice": {"mode": "clone"}},
                "S2": {"first_sec": 7.0, "gender": "female",
                       "reference": {}, "voice": {"mode": "preset"}}}

    updated = store.from_pipeline(JOB, segments, speakers, "tr", "ru")
    assert updated.segment(2).speaker_id == "S2", "ручное назначение затёрто"
    assert updated.speaker("S1").name == "Ведущий", "имя спикера потеряно"


# ------------------------------------------------ добавление и удаление

def test_api_create_speaker(client, token, project):
    """Спикера можно завести вручную: система могла свести двоих в одного."""
    r = client.post(f"/api/projects/{JOB}/speakers", json={},
                    headers=headers(token, project.version))
    assert r.status_code == 201, r.text
    new = r.json()
    assert new["id"] == "S3"          # первый свободный номер
    assert new["stats"]["segments_count"] == 0

    saved = store.load(JOB)
    assert len(saved.active_speakers()) == 3
    # цвет не повторяет чужой — спикеров различают по цвету на таймлайне
    colors = [s.color for s in saved.speakers]
    assert len(set(colors)) == len(colors)


def test_api_create_speaker_reuses_free_id(client, token, project):
    """Номер берётся свободный, а не «следующий по счётчику»."""
    v = project.version
    client.delete(f"/api/projects/{JOB}/speakers/S2?move_to=S1",
                  headers=headers(token, v))
    v = store.load(JOB).version
    r = client.post(f"/api/projects/{JOB}/speakers", json={},
                    headers=headers(token, v))
    assert r.json()["id"] == "S2", "освободившийся номер не переиспользован"


def test_api_delete_speaker_moves_segments(client, token, project):
    r = client.delete(f"/api/projects/{JOB}/speakers/S2?move_to=S1",
                      headers=headers(token, project.version))
    assert r.status_code == 200, r.text
    assert r.json()["moved_segments"] == 2

    saved = store.load(JOB)
    assert saved.speaker("S2") is None
    assert all(s.speaker_id == "S1" for s in saved.segments)
    assert saved.speaker("S1").stats.segments_count == 4
    # набор реплик изменился — профиль принимающего надо пересобрать
    assert saved.speaker("S1").voice.locked is False


def test_api_delete_speaker_requires_target(client, token, project):
    """Реплики нельзя оставить без спикера: у них не будет голоса."""
    r = client.delete(f"/api/projects/{JOB}/speakers/S2",
                      headers=headers(token, project.version))
    assert r.status_code == 400
    assert "укажите" in r.json()["error"].lower()
    assert store.load(JOB).speaker("S2") is not None


def test_api_delete_empty_speaker_needs_no_target(client, token, project):
    """У пустого спикера переносить нечего."""
    v = project.version
    client.post(f"/api/projects/{JOB}/speakers", json={}, headers=headers(token, v))
    v = store.load(JOB).version
    r = client.delete(f"/api/projects/{JOB}/speakers/S3", headers=headers(token, v))
    assert r.status_code == 200
    assert r.json()["moved_segments"] == 0


def test_api_cannot_delete_last_speaker(client, token, project):
    v = project.version
    client.delete(f"/api/projects/{JOB}/speakers/S2?move_to=S1",
                  headers=headers(token, v))
    v = store.load(JOB).version
    r = client.delete(f"/api/projects/{JOB}/speakers/S1", headers=headers(token, v))
    assert r.status_code == 400
    assert store.load(JOB).speaker("S1") is not None


def test_api_undo_restores_deleted_speaker(client, token, project):
    """Отмена возвращает спикера вместе с его репликами."""
    client.delete(f"/api/projects/{JOB}/speakers/S2?move_to=S1",
                  headers=headers(token, project.version))
    v = store.load(JOB).version
    r = client.post(f"/api/projects/{JOB}/undo", headers=headers(token, v))
    assert r.status_code == 200, r.text

    saved = store.load(JOB)
    assert saved.speaker("S2") is not None, "спикер не вернулся"
    assert saved.speaker("S2").stats.segments_count == 2, "реплики не вернулись"


def test_api_undo_removes_created_speaker(client, token, project):
    client.post(f"/api/projects/{JOB}/speakers", json={},
                headers=headers(token, project.version))
    v = store.load(JOB).version
    client.post(f"/api/projects/{JOB}/undo", headers=headers(token, v))
    assert store.load(JOB).speaker("S3") is None
