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
