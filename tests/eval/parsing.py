"""
Парсер аннотаций Label Studio (JSON-формат).

Реальная структура экспорта Label Studio:
[
  {
    "id": 17537,
    "annotations": [
      {
        "id": 9162,
        "result": [
          {
            "original_width": 640,
            "original_height": 640,
            "value": {
              "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],  // % от размера
              "polygonlabels": ["zone_name"]
            },
            "meta": {"text": ["ground truth text"]},
            "type": "polygonlabels"
          }
        ]
      }
    ],
    "data": {"image": "/data/local-files/?d=data/passport/file.jpg"}
  }
]
"""

import json
import logging
from pathlib import Path

from eval.base import ImageAnnotation, ZoneAnnotation

logger = logging.getLogger(__name__)


def load_annotations(
    json_path: Path,
    doc_type: str,
    images_dir: Path,
) -> list[ImageAnnotation]:
    """
    Загружает аннотации из Label Studio JSON-экспорта.

    Args:
        json_path: путь к JSON-файлу экспорта.
        doc_type: тип документа (используется как метка).
        images_dir: папка с исходными изображениями.

    Returns:
        Список ImageAnnotation.
    """
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    annotations: list[ImageAnnotation] = []

    for item in raw:
        filename = _extract_filename(item)
        if filename is None:
            logger.warning(f"Skipping item without image path: {item.get('id')}")
            continue

        image_path = images_dir / filename
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            continue

        zones = _parse_zones(item, filename)
        annotations.append(ImageAnnotation(
            filename=str(image_path),
            doc_type=doc_type,
            zones=zones,
        ))

    logger.info(
        f"Loaded {len(annotations)} annotations "
        f"for doc_type='{doc_type}' from {json_path.name}"
    )
    return annotations


def _extract_filename(item: dict) -> str | None:
    """
    Извлекает имя файла из поля data.image.

    Поддерживает форматы:
      - /data/local-files/?d=data/passport/file.jpg
      - /data/upload/1/file.jpg
      - file:///path/to/file.jpg
      - просто имя файла
    """
    image_path = item.get("data", {}).get("image", "")
    if not image_path:
        return None

    # Формат /data/local-files/?d=data/passport/file.jpg
    if "?d=" in image_path:
        path_part = image_path.split("?d=")[-1]
        return Path(path_part).name

    # Любой другой формат — берём последний компонент пути
    return Path(image_path).name


def _parse_zones(item: dict, filename: str) -> list[ZoneAnnotation]:
    """
    Парсит зоны из аннотации.

    Label Studio хранит результаты в item["annotations"][0]["result"].
    Берём первую (последнюю сохранённую) аннотацию.
    """
    annotations_list = item.get("annotations", [])
    if not annotations_list:
        logger.warning(f"No annotations for '{filename}'")
        return []

    # Берём последнюю аннотацию (самую свежую правку)
    results = annotations_list[-1].get("result", [])

    zones: list[ZoneAnnotation] = []
    for result in results:
        # Пропускаем textarea-блоки — текст берём из meta внутри polygonlabels
        if result.get("type") != "polygonlabels":
            continue

        zone = _parse_single_zone(result, filename)
        if zone is not None:
            zones.append(zone)

    return zones


def _parse_single_zone(result: dict, filename: str) -> ZoneAnnotation | None:
    """Парсит одну зону из result-блока Label Studio."""
    value = result.get("value", {})
    labels = value.get("polygonlabels", [])

    if not labels:
        return None

    zone_name = labels[0]
    points = value.get("points", [])

    if len(points) != 4:
        logger.warning(
            f"Expected 4 points for zone '{zone_name}' "
            f"in '{filename}', got {len(points)} — skipping"
        )
        return None

    # Координаты уже в процентах (0–100) — flatten в плоский список
    obb_points = [coord for point in points for coord in point]

    # Текст хранится в meta.text внутри того же блока
    meta_text = result.get("meta", {}).get("text", [])
    text = meta_text[0].strip() if meta_text else None

    return ZoneAnnotation(
        zone_name=zone_name,
        obb_points=obb_points,
        text=text,
    )


def denormalize_points(
    obb_points: list[float],
    image_width: int,
    image_height: int,
) -> list[float]:
    """
    Конвертирует координаты из процентов (Label Studio) в пиксели.

    Args:
        obb_points: [x1,y1,x2,y2,x3,y3,x4,y4] в процентах (0–100).
        image_width: ширина изображения в пикселях.
        image_height: высота изображения в пикселях.

    Returns:
        [x1,y1,x2,y2,x3,y3,x4,y4] в пикселях.
    """
    result = []
    for i, coord in enumerate(obb_points):
        if i % 2 == 0:
            result.append(coord / 100.0 * image_width)
        else:
            result.append(coord / 100.0 * image_height)
    return result