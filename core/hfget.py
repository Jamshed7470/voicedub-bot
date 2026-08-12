"""Устойчивая загрузка моделей с HuggingFace.

Штатный загрузчик huggingface_hub на нестабильной сети зависает насмерть:
сокет умирает молча, ретраи не срабатывают, задача стоит часами. Здесь
файлы качаются через curl с докачкой (-C -) и сторожем скорости: если
скорость падает ниже порога — соединение рвётся и продолжается с места
обрыва. Целостность проверяется по sha256 из LFS-метаданных.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

API = "https://huggingface.co/api/models"
RESOLVE = "https://huggingface.co/{repo}/resolve/{rev}/{path}"

# файлы, которых достаточно для wav2vec2-модели выравнивания
MODEL_FILES = {"config.json", "preprocessor_config.json", "tokenizer_config.json",
               "vocab.json", "special_tokens_map.json", "added_tokens.json"}
WEIGHTS = ("model.safetensors", "pytorch_model.bin")

MAX_STALLS = 25         # обрывов связи на одну попытку скачивания
MAX_ATTEMPTS = 3        # попыток скачать файл целиком и сойтись по хэшу
SPEED_LIMIT = "20480"   # байт/с — ниже этого порога соединение считается мёртвым
SPEED_TIME = "30"       # секунд подряд ниже порога → разрыв и докачка


def _cache_root() -> Path:
    env = os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE")
    if env:
        return Path(env)
    home = os.getenv("HF_HOME")
    base = Path(home) if home else Path.home() / ".cache" / "huggingface"
    return base / "hub"


def _api_json(url: str, token: str | None) -> object:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _curl(url: str, dest: Path, token: str | None, expect_sha: str | None,
          size: int | None) -> bool:
    """Качает url в dest с докачкой и проверкой целостности.

    Докачка (-C -) опасна на подписанных ссылках CDN: если сервер
    проигнорирует диапазон и отдаст тело целиком, curl допишет его к остатку
    и получится склейка правильного размера, но битая. Поэтому размер
    сверяется точно, а при любом расхождении файл качается заново с нуля.
    """
    part = dest.with_suffix(dest.suffix + ".part")

    # huggingface_hub мог оборваться на середине — забираем его хвост себе,
    # иначе те же сотни мегабайт качаются заново
    if expect_sha and not part.exists():
        blob = dest.parent.parent.parent / "blobs" / f"{expect_sha}.incomplete"
        if blob.exists() and (not size or blob.stat().st_size < size):
            try:
                shutil.move(str(blob), str(part))
                log.info("HF: подхватил %.0f МБ незавершённой загрузки %s",
                         part.stat().st_size / 1e6, dest.name)
            except OSError:
                log.exception("HF: не удалось забрать незавершённую загрузку")

    def fetch(resume: bool) -> bool:
        cmd = ["curl", "-sL", "--fail", "--connect-timeout", "20",
               "--speed-limit", SPEED_LIMIT, "--speed-time", SPEED_TIME]
        if resume:
            cmd += ["-C", "-"]
        if token:
            cmd += ["-H", f"Authorization: Bearer {token}"]
        cmd += ["-o", str(part), url]
        return subprocess.run(cmd, capture_output=True, text=True).returncode == 0

    def part_size() -> int:
        return part.stat().st_size if part.exists() else 0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        # остаток прошлой попытки не меньше целевого размера — это склейка
        if size and part_size() >= size:
            part.unlink(missing_ok=True)

        ok = False
        for stall in range(1, MAX_STALLS + 1):
            # докачиваем только настоящий недокачанный хвост
            resume = part.exists() and (not size or part_size() < size)
            if fetch(resume):
                ok = True
                break
            log.warning("HF: %s — %.0f МБ, обрыв (%d/%d), докачиваю",
                        dest.name, part_size() / 1e6, stall, MAX_STALLS)
            if size and part_size() > size:
                log.warning("HF: %s — размер превысил ожидаемый, качаю заново",
                            dest.name)
                part.unlink(missing_ok=True)
        if not ok:
            log.error("HF: %s — связь не восстановилась", dest.name)
            return False

        if size and part_size() != size:
            log.warning("HF: %s — размер %d вместо %d, попытка %d/%d",
                        dest.name, part_size(), size, attempt, MAX_ATTEMPTS)
            part.unlink(missing_ok=True)
            continue
        if expect_sha and _sha256(part) != expect_sha:
            log.warning("HF: %s — хэш не сошёлся, качаю заново (%d/%d)",
                        dest.name, attempt, MAX_ATTEMPTS)
            part.unlink(missing_ok=True)
            continue

        part.replace(dest)
        return True

    log.error("HF: %s — не удалось получить целый файл за %d попыток",
              dest.name, MAX_ATTEMPTS)
    return False


def ensure_model(repo_id: str, revision: str = "main",
                 token: str | None = None) -> bool:
    """Гарантирует, что модель repo_id целиком лежит в кэше HuggingFace.

    Возвращает False, если скачать не вышло — вызывающий код может
    откатиться на штатный загрузчик.
    """
    if not shutil.which("curl"):
        return False
    try:
        info = _api_json(f"{API}/{repo_id}/revision/{revision}", token)
        sha = info["sha"]
        tree = {f["path"]: f for f in info.get("siblings", [])
                if f.get("rfilename") is None}
        if not tree:  # у /revision другой формат — берём дерево отдельно
            tree = {f["path"]: f for f in
                    _api_json(f"{API}/{repo_id}/tree/{revision}", token)
                    if f.get("type") == "file"}
    except Exception:  # noqa: BLE001 — нет сети/приватный репозиторий
        log.exception("HF: не удалось получить список файлов %s", repo_id)
        return False

    snap = _cache_root() / f"models--{repo_id.replace('/', '--')}" / "snapshots" / sha
    snap.mkdir(parents=True, exist_ok=True)

    wanted = [p for p in tree if p in MODEL_FILES]
    weights = next((w for w in WEIGHTS if w in tree), None)
    if weights:
        wanted.append(weights)

    for path in wanted:
        dest = snap / path
        meta = tree[path]
        size = meta.get("size")
        if dest.exists() and (not size or dest.stat().st_size == size):
            continue
        lfs_sha = (meta.get("lfs") or {}).get("oid")
        url = RESOLVE.format(repo=repo_id, rev=revision, path=path)
        mb = (size or 0) / 1e6
        log.info("HF: качаю %s / %s (%.0f МБ)", repo_id, path, mb)
        if not _curl(url, dest, token, lfs_sha, size):
            return False

    refs = _cache_root() / f"models--{repo_id.replace('/', '--')}" / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / revision).write_text(sha, encoding="utf-8")
    return True
