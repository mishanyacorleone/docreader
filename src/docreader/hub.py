"""
Автоматическая загрузка и кэширование моделей.

Модели скачиваются при первом вызове и сохраняются в:
    ~/.cache/docreader/models/
"""

import os
import hashlib
import logging
import tarfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

_BASE_URL_CLASSIFIER = "https://github.com/mishanyacorleone/docreader/releases/download/v0.2.0"
_BASE_URL = "https://github.com/mishanyacorleone/docreader/releases/download/v0.1.0"

MODEL_REGISTRY: dict[str, dict] = {
    # === Классификатор документов (YOLO OBB) ===
    "doc_classifier.pt": {
        "url": f"{_BASE_URL_CLASSIFIER}/doc_classifier.pt",
        "sha256": "b1af689fe58849474a6a5cf879458fcba6d017233ca1bd54b5d83098cd9387f5",
        "size_mb": 5.49,
    },

    # === Детекторы зон ===
    "passport.pt": {
        "url": f"{_BASE_URL}/passport.pt",
        "sha256": "bebe46bcd4270442c1e14e9b5a403c9f59212d92ed8181af1326f9f80bc0f0c0",
        "size_mb": 5.55,
    },
    "diplom.pt": {
        "url": f"{_BASE_URL}/diplom.pt",
        "sha256": "f1848733eefa4741ead199cf8226e2fc141b08b01d625912d19926bb7ebc6387",
        "size_mb": 5.71,
    },
    "attestat.pt": {
        "url": f"{_BASE_URL}/attestat.pt",
        "sha256": "9b6eaa5860b0cb0498995c0ab8015a9b85a9a910b429f2bef509e1202232199d",
        "size_mb": 5.72,
    },
    "snils.pt": {
        "url": f"{_BASE_URL}/snils.pt",
        "sha256": "84775a6ff1ababb3f8e31a8aa768717cf9d65d8b84df9c0cd48eb7bdaf680218",
        "size_mb": 5.82,
    },

    # === EasyOCR ===
    "easyocr_custom.tar.gz": {
        "url": f"{_BASE_URL}/easyocr_custom.tar.gz",
        "sha256": "832ce5a7f3a1086d81beb1c991347e3f545a425646bc87f3f576ae06fecd2420",
        "size_mb": 87.1,
        "extract_to": "easyocr"
    }
}

def get_cache_dir() -> Path:
    """Возвращает директорию кэша моделей."""
    cache = Path(os.environ.get("DOCREADER_CACHE", "~/.cache/docreader"))
    cache = cache.expanduser() / "models"
    cache.mkdir(parents=True, exist_ok=True)
    return cache



def _sha256_file(path: Path) -> str:
    """Считает SHA-256 хэш файла."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_file(url: str, dest: Path, expected_sha256: Optional[str] = None):
    """Скачивает файл с прогресс-баром и проверкой хэша."""
    logger.info(f"Downloading {url}")

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=dest.name,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    # Проверка целостности
    if expected_sha256:
        actual = _sha256_file(dest)
        if actual != expected_sha256:
            dest.unlink()
            raise ValueError(
                f"Hash mismatch for {dest.name}: "
                f"expected {expected_sha256[:16]}..., "
                f"got {actual[:16]}..."
            )

    logger.info(f"Saved to {dest}")


def _extract_archive(archive_path: Path, extract_to: Path):
    """Распаковывает tar.gz архив."""
    logger.info(f"Extracting {archive_path.name} → {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_to)

    archive_path.unlink()
    logger.info(f"Extracted and cleaned up: {archive_path.name}")


def ensure_model(filename: str, cache_dir: Optional[Path] = None) -> Path:
    """
    Гарантирует наличие файла модели. Скачивает, если отсутствует.

    Args:
        filename: имя файла из MODEL_REGISTRY.
        cache_dir: директория кэша (по умолчанию ~/.cache/docreader/models).

    Returns:
        Путь к файлу модели.

    Raises:
        KeyError: если файл не зарегистрирован.
        ConnectionError: если не удалось скачать.
    """
    if filename not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{filename}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    cache = cache_dir or get_cache_dir()
    meta = MODEL_REGISTRY[filename]

    if "extract_to" in meta:
        extract_dir = cache / meta["extract_to"]
        if extract_dir.exists() and any(extract_dir.iterdir()):
            logger.debug(f"Already extracted: {extract_dir}")
            return extract_dir
        
        # Скачиваем и распаковываем
        archive_path = cache / filename
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        _download_file(
            url=meta["url"],
            dest=archive_path,
            expected_sha256=meta.get("sha256")
        )
        _extract_archive(archive_path, extract_dir)
        
        return extract_dir
    
    filepath = cache / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.exists():
        logger.debug(f"Model cached: {filepath}")
        return filepath

    _download_file(
        url=meta["url"],
        dest=filepath,
        expected_sha256=meta.get("sha256"),
    )

    return filepath


def ensure_all_models(cache_dir: Optional[Path] = None) -> Path:
    """Скачивает все модели. Возвращает директорию кэша."""
    cache = cache_dir or get_cache_dir()
    for filename in MODEL_REGISTRY:
        ensure_model(filename, cache)
    return cache


def get_model_paths() -> dict[str, Path]:
    """
    Возвращает словарь {имя_модели: полный_путь}
    для всех зарегистрированных моделей.
    """
    cache = get_cache_dir()
    paths = {}

    for filename, meta in MODEL_REGISTRY.items():
        if "extract_to" in meta:
            paths[filename] = cache / meta["extract_to"]
        else:
            paths[filename] = cache / filename

    return paths


def get_model_status() -> dict[str, dict]:
    """
    Показывает статус всех моделей: путь, скачана ли, размер.
    """
    cache = get_cache_dir()
    status = {}

    for filename, meta in MODEL_REGISTRY.items():
        if "extract_to" in meta:
            path = cache / meta["extract_to"]
            exists = path.exists() and any(path.iterdir())
        else:
            path = cache / filename
            exists = path.exists()

        status[filename] = {
            "path": str(path),
            "downloaded": exists,
            "size_mb": meta.get("size_mb", "?"),
            "url": meta["url"],
        }

    return status