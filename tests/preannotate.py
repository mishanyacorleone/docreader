"""
Предразметка данных для Label Studio.

Прогоняет детектор + OCR по всем изображениям и формирует JSON
в формате Label Studio pre-annotations.

Использование:
    # Запускать Label Studio нужно так:
    LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
    LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/mnt/mishutqa/PycharmProjects/sirius/docreader/tests \
    label-studio start

    # Затем запускать предразметку:
    python preannotate.py
    python preannotate.py --doc-types passport snils
    python preannotate.py --doc-types attestat --output annotations/pre

Результат:
    annotations/pre/{doc_type}.json  — импортируй в Label Studio

Как импортировать в Label Studio:
    1. Создай проект
    2. Settings → Labeling Interface → вставь XML конфиг
    3. Import → загрузи {doc_type}.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from docreader.factory import create_detector, create_ocr
from docreader.preprocessing.geometry import crop_obb_region
from docreader.utils import load_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

TESTS_DIR = Path(__file__).parent
DATA_DIR = TESTS_DIR / "data"
OUTPUT_DIR = TESTS_DIR / "annotations" / "pre"

# Должен совпадать с LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT
DOCUMENT_ROOT = TESTS_DIR

DOC_TYPES = ["attestat", "diplom", "passport", "snils"]

SKIP_OCR_ZONES = {"stamp", "gerb"}


# ---------------------------------------------------------------------------
# Конвертация координат
# ---------------------------------------------------------------------------

def obb_points_to_percent(
    obb_points: np.ndarray,
    image_width: int,
    image_height: int,
) -> list[list[float]]:
    """
    Конвертирует OBB-координаты из пикселей в проценты (формат Label Studio).

    Args:
        obb_points: массив координат shape (8,) или (4, 2).
        image_width: ширина изображения.
        image_height: высота изображения.

    Returns:
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] в процентах (0–100).
    """
    pts = np.array(obb_points, dtype=np.float32).reshape(4, 2)
    result = []
    for x, y in pts:
        result.append([
            round(float(x) / image_width * 100, 4),
            round(float(y) / image_height * 100, 4),
        ])
    return result


# ---------------------------------------------------------------------------
# Формирование Label Studio JSON
# ---------------------------------------------------------------------------

def build_label_studio_task(
    image_path: Path,
    doc_type: str,
    zones: list[dict],
    document_root: Path,
) -> dict:
    """
    Формирует одну задачу Label Studio с предразметкой.

    Текст зоны передаётся двумя способами одновременно:
      1. Через meta.text внутри polygonlabels-блока — подхватывается
         как perRegion textarea при импорте в большинстве версий LS.
      2. Через отдельный textarea-блок с parent_id — fallback для
         версий где meta не работает.

    Args:
        image_path: абсолютный путь к изображению.
        doc_type: тип документа.
        zones: список зон с полями zone_name, points_percent, text.
        document_root: корневая папка (совпадает с DOCUMENT_ROOT).

    Returns:
        Словарь задачи в формате Label Studio.
    """
    try:
        relative_path = image_path.resolve().relative_to(document_root.resolve())
    except ValueError:
        logger.warning(
            f"Image {image_path} is outside DOCUMENT_ROOT {document_root}, "
            f"using absolute path"
        )
        relative_path = image_path.resolve()

    image_url = f"/data/local-files/?d={str(relative_path).replace(chr(92), '/')}"

    results = []
    for i, zone in enumerate(zones):
        zone_id = f"zone_{i}"

        # Бокс зоны — текст передаём через meta (основной способ для perRegion)
        region: dict = {
            "id": zone_id,
            "type": "polygonlabels",
            "from_name": "label",
            "to_name": "image",
            "value": {
                "points": zone["points_percent"],
                "polygonlabels": [zone["zone_name"]],
                "closed": True,
            },
        }
        if zone.get("text") is not None:
            region["meta"] = {"text": [zone["text"]]}

        results.append(region)

        # Отдельный textarea-блок с parent_id — fallback для старых версий LS
        if zone.get("text") is not None:
            results.append({
                "id": f"{zone_id}_text",
                "type": "textarea",
                "from_name": "transcription",
                "to_name": "image",
                "value": {
                    "text": [zone["text"]],
                },
                "parent_id": zone_id,
            })

    return {
        "data": {
            "image": image_url,
            "doc_type": doc_type,
        },
        "predictions": [
            {
                "model_version": "docreader-preannotation-v1",
                "result": results,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Предразметка одного изображения
# ---------------------------------------------------------------------------

def preannotate_image(
    image_path: Path,
    doc_type: str,
    detector,
    ocr_engine,
    document_root: Path,
) -> dict:
    """
    Прогоняет детектор и OCR на одном изображении,
    формирует задачу Label Studio.
    """
    image = load_image(str(image_path))
    h, w = image.shape[:2]

    try:
        detections = detector.detect(image, doc_type)
    except Exception as exc:
        logger.warning(f"Detector failed on {image_path.name}: {exc}")
        detections = []

    zones = []
    for det in detections:
        points_percent = obb_points_to_percent(det.obb_points, w, h)

        text = None
        if det.zone_name not in SKIP_OCR_ZONES:
            crop = crop_obb_region(image, det.obb_points)
            if crop is not None and crop.size > 0:
                try:
                    ocr_result = ocr_engine.recognize(crop)
                    text = ocr_result.text.strip() or None
                except Exception as exc:
                    logger.warning(
                        f"OCR failed for zone '{det.zone_name}' "
                        f"in {image_path.name}: {exc}"
                    )

        zones.append({
            "zone_name": det.zone_name,
            "points_percent": points_percent,
            "text": text,
        })

    logger.debug(f"{image_path.name}: {len(zones)} zones detected")
    return build_label_studio_task(image_path, doc_type, zones, document_root)


# ---------------------------------------------------------------------------
# Предразметка одного типа документа
# ---------------------------------------------------------------------------

def preannotate_doc_type(
    doc_type: str,
    images_dir: Path,
    output_dir: Path,
    detector,
    ocr_engine,
    document_root: Path,
) -> None:
    """
    Прогоняет предразметку для всех изображений одного типа документа
    и сохраняет JSON для импорта в Label Studio.
    """
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    if not image_paths:
        logger.warning(f"No images found in {images_dir}")
        return

    logger.info(f"Processing '{doc_type}': {len(image_paths)} images")

    tasks = []
    for image_path in image_paths:
        task = preannotate_image(
            image_path, doc_type, detector, ocr_engine, document_root
        )
        tasks.append(task)
        logger.info(f"  ✓ {image_path.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{doc_type}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(tasks)} tasks → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Генерирует предразметку для Label Studio"
    )
    parser.add_argument(
        "--doc-types",
        nargs="+",
        choices=DOC_TYPES,
        default=DOC_TYPES,
        help="Типы документов для предразметки (default: все)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Папка с изображениями (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Папка для JSON-файлов (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--document-root",
        type=Path,
        default=DOCUMENT_ROOT,
        help=(
            f"Путь совпадающий с LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT "
            f"(default: {DOCUMENT_ROOT})"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Initializing detector and OCR...")
    detector = create_detector()
    ocr_engine = create_ocr()

    for doc_type in args.doc_types:
        images_dir = args.data_dir / doc_type

        if not images_dir.exists():
            logger.warning(f"Images dir not found, skipping: {images_dir}")
            continue

        preannotate_doc_type(
            doc_type=doc_type,
            images_dir=images_dir,
            output_dir=args.output,
            detector=detector,
            ocr_engine=ocr_engine,
            document_root=args.document_root,
        )

    logger.info(
        f"\nДля импорта в Label Studio:\n"
        f"  1. Запусти Label Studio с переменными:\n"
        f"       LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \\\n"
        f"       LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={args.document_root} \\\n"
        f"       label-studio start\n"
        f"  2. Открой проект → Import → выбери JSON из {args.output}\n"
        f"  3. Проверь и поправь предразметку руками\n"
        f"  4. Экспортируй в {TESTS_DIR}/annotations/{{doc_type}}.json\n"
    )


if __name__ == "__main__":
    main()