"""Пути в кэше разбора: переезд папки speakers/ не должен ломать профили.

Кэш переносит папку speakers/ из задачи к себе, а speakers.json остаётся с
абсолютными путями на старое место. Забытое поле не даёт ошибки при
чтении — оно молча указывает в никуда, и задача падает много позже, уже на
синтезе, где причина не видна.
"""
from __future__ import annotations

import json
from pathlib import Path

from core import cache


def make_analysis(tmp_path: Path, job_dir: Path) -> Path:
    """Раскладка, какой её оставляет пайплайн после переезда в кэш."""
    analysis = tmp_path / "cache" / "key" / "analysis_auto"
    (analysis / "speakers" / "S1").mkdir(parents=True)
    for name in ("ref_main.wav", "voice_profile.pt",
                 "identity_embedding.npy", "centroid.npy"):
        (analysis / "speakers" / "S1" / name).write_bytes(b"x")

    profiles = {
        "S1": {
            "id": "S1",
            "ref_main": str(job_dir / "speakers" / "S1" / "ref_main.wav"),
            "centroid_path": str(job_dir / "speakers" / "S1" / "centroid.npy"),
            "reference": {
                "path": str(job_dir / "speakers" / "S1" / "ref_main.wav"),
                "clean_sec": 20.0,
            },
            "voice": {
                "mode": "clone",
                "profile_path": str(job_dir / "speakers" / "S1" / "voice_profile.pt"),
                "identity_path": str(job_dir / "speakers" / "S1" / "identity_embedding.npy"),
                "locked": True,
            },
        }
    }
    (analysis / "speakers.json").write_text(
        json.dumps(profiles, ensure_ascii=False), encoding="utf-8")
    return analysis


def test_load_profiles_rewrites_every_path(tmp_path):
    """Каждый путь внутрь speakers/ переписывается на папку кэша."""
    job_dir = tmp_path / "jobs" / "job1"
    analysis = make_analysis(tmp_path, job_dir)

    profiles = cache.load_profiles(analysis)
    voice = profiles["S1"]["voice"]

    assert Path(profiles["S1"]["ref_main"]).exists()
    assert Path(profiles["S1"]["reference"]["path"]).exists()
    assert Path(profiles["S1"]["centroid_path"]).exists()
    assert Path(voice["profile_path"]).exists(), "путь к профилю не переписан"
    assert Path(voice["identity_path"]).exists(), "путь к отпечатку не переписан"
    # ни один путь не должен указывать в исчезнувшую папку задачи
    assert "jobs" not in voice["profile_path"]


def test_verify_profiles_catches_missing_file(tmp_path):
    """Пропавший файл профиля обнаруживается сразу, а не на синтезе."""
    job_dir = tmp_path / "jobs" / "job1"
    analysis = make_analysis(tmp_path, job_dir)
    profiles = cache.load_profiles(analysis)
    assert cache.verify_profiles(profiles) is True

    Path(profiles["S1"]["voice"]["profile_path"]).unlink()
    assert cache.verify_profiles(profiles) is False


def test_load_profiles_survives_old_format(tmp_path):
    """Кэш версии 1 (без voice/reference) читается без падения."""
    analysis = tmp_path / "cache" / "key" / "analysis_auto"
    (analysis / "speakers" / "S1").mkdir(parents=True)
    (analysis / "speakers" / "S1" / "ref_main.wav").write_bytes(b"x")
    (analysis / "speakers.json").write_text(json.dumps({
        "S1": {"id": "S1", "ref_main": "C:/old/job/speakers/S1/ref_main.wav",
               "refs_emotion": {"happy": "C:/old/job/speakers/S1/ref_happy.wav"}}
    }), encoding="utf-8")

    profiles = cache.load_profiles(analysis)
    assert Path(profiles["S1"]["ref_main"]).exists()
    assert "old" not in profiles["S1"]["refs_emotion"]["happy"]


# --------------------------------------------- кэш не владеет профилями

def make_job(tmp_path: Path) -> Path:
    """Папка задачи с готовым разбором, как её оставляет пайплайн."""
    import json

    job = tmp_path / "jobs" / "job1"
    (job / "speakers" / "S1").mkdir(parents=True)
    for name in ("ref_main.wav", "voice_profile.pt", "identity_embedding.npy"):
        (job / "speakers" / "S1" / name).write_bytes(b"x" * 100)
    (job / "speakers.json").write_text(json.dumps({
        "S1": {"id": "S1",
               "ref_main": str(job / "speakers" / "S1" / "ref_main.wav"),
               "reference": {"path": str(job / "speakers" / "S1" / "ref_main.wav")},
               "voice": {"mode": "clone", "locked": True,
                         "profile_path": str(job / "speakers" / "S1" / "voice_profile.pt"),
                         "identity_path": str(job / "speakers" / "S1" / "identity_embedding.npy")}}
    }), encoding="utf-8")
    (job / "transcript.json").write_text('{"segments": []}', encoding="utf-8")
    return job


def test_store_analysis_leaves_profiles_in_job(tmp_path, monkeypatch):
    """Кэш забирает КОПИЮ: задача не должна от него зависеть.

    Раньше папка speakers/ переезжала в кэш. Очистка кэша по возрасту
    уносила профили голосов активной задачи, и та узнавала об этом только
    на синтезе — «у спикера нет профиля», хотя всё было посчитано.
    """
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    job = make_job(tmp_path)

    stored = cache.store_analysis(job, "key1", "auto")
    assert stored is not None

    assert (job / "speakers" / "S1" / "voice_profile.pt").exists(), \
        "профиль ушёл из задачи в кэш"
    assert (stored / "speakers" / "S1" / "voice_profile.pt").exists(), \
        "копия в кэш не попала"


def test_job_survives_cache_purge(tmp_path, monkeypatch):
    """Удаление кэша не ломает задачу, взявшую из него разбор."""
    import shutil

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    source_job = make_job(tmp_path)
    stored = cache.store_analysis(source_job, "key1", "auto")

    # новая задача берёт разбор из кэша
    new_job = tmp_path / "jobs" / "job2"
    new_job.mkdir(parents=True)
    profiles = cache.adopt_analysis(stored, new_job)

    # ...и кэш исчезает
    shutil.rmtree(tmp_path / "cache")

    assert cache.verify_profiles(profiles), "профили пропали вместе с кэшем"
    path = Path(profiles["S1"]["voice"]["profile_path"])
    assert path.exists()
    assert "cache" not in path.parts, "путь всё ещё смотрит в кэш"


# ------------------------------------ что переживает очистку после успеха

def test_cleanup_keeps_project_and_report(tmp_path, monkeypatch):
    """Медиа уходят, а проект и карта голосов остаются.

    Без project.json студия не откроет завершённую задачу и правки
    человека пропадут; без report.md негде посмотреть, каким голосом кто
    озвучен. Раньше очистка сносила и то, и другое.
    """
    from core import pipeline

    jobs = tmp_path / "jobs"
    job = jobs / "job1"
    (job / "speakers" / "S1").mkdir(parents=True)
    (job / "speakers" / "S1" / "voice_profile.pt").write_bytes(b"x")
    for name in ("project.json", "report.md", "speakers.json",
                 "transcript.json", "speaker_registry.json"):
        (job / name).write_text("{}", encoding="utf-8")
    # тяжёлое, что должно уйти
    for name in ("source.wav", "vocals.wav", "dubbed.wav", "input.mp4"):
        (job / name).write_bytes(b"0" * 1000)
    (job / "synth").mkdir()
    (job / "synth" / "seg_1.wav").write_bytes(b"0" * 1000)

    monkeypatch.setattr(pipeline, "JOBS_DIR", jobs)
    pipeline.cleanup_job("job1")

    assert (job / "project.json").exists(), "проект удалён — студия его не откроет"
    assert (job / "report.md").exists(), "карта голосов удалена"
    assert (job / "speakers" / "S1" / "voice_profile.pt").exists(), \
        "профили голосов удалены — переозвучка потребует полного пересчёта"

    assert not (job / "source.wav").exists(), "тяжёлые дорожки остались"
    assert not (job / "synth").exists(), "папка синтеза осталась"
