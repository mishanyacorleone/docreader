"""
Точка входа для evaluation пайплайна.

Использование:
    python run_eval.py                        # все компоненты
    python run_eval.py --components classifier ocr
    python run_eval.py --doc-types passport snils
    python run_eval.py --iou-threshold 0.5 --nms-iou-threshold 0.45
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from docreader import DocReader
from docreader.factory import create_classifier, create_detector, create_ocr

from eval import (
    load_annotations,
    evaluate_classifier,
    evaluate_detector,
    evaluate_ocr,
    compute_business_metrics,
    print_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Конфигурация путей
# ---------------------------------------------------------------------------

TESTS_DIR = Path(__file__).parent
DATA_DIR = TESTS_DIR / "data"
ANNOTATIONS_DIR = TESTS_DIR / "annotations"

DOC_TYPES = ["attestat", "diplom", "passport", "snils"]

# Ключевые поля для бизнес-метрик (None = все поля с текстом)
KEY_ZONES: dict[str, list[str]] = {
    "passport": ["surname", "firstname", "middlename", "passport_num", "dateOfBirth"],
    "snils": ["snils", "fio"],
    "attestat": ["fio", "number", "grad_year"],
    "diplom": ["fio", "series_numbers", "spec_name", "university_name"],
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate docreader components")
    parser.add_argument(
        "--components",
        nargs="+",
        choices=["classifier", "detector", "ocr", "business"],
        default=["classifier", "detector", "ocr", "business"],
        help="Компоненты для оценки (default: все)",
    )
    parser.add_argument(
        "--doc-types",
        nargs="+",
        choices=DOC_TYPES,
        default=DOC_TYPES,
        help="Типы документов для оценки (default: все)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.50,
        help="Порог IoU для матчинга GT vs предсказания (default: 0.50)",
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=0.45,
        help="Порог IoU для NMS — подавление дублей (default: 0.45)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Загружаем аннотации для выбранных типов документов
    all_annotations = []
    for doc_type in args.doc_types:
        json_path = ANNOTATIONS_DIR / f"{doc_type}.json"
        images_dir = DATA_DIR / doc_type

        if not json_path.exists():
            logger.warning(f"Annotations not found, skipping: {json_path}")
            continue
        if not images_dir.exists():
            logger.warning(f"Images dir not found, skipping: {images_dir}")
            continue

        annotations = load_annotations(json_path, doc_type, images_dir)
        all_annotations.extend(annotations)
        logger.info(f"{doc_type}: {len(annotations)} images loaded")

    if not all_annotations:
        logger.error("No annotations loaded. Exiting.")
        sys.exit(1)

    logger.info(
        f"Total: {len(all_annotations)} images "
        f"across {len(args.doc_types)} doc types"
    )

    # Инициализируем компоненты
    components = set(args.components)
    classifier_metrics = None
    detector_metrics = None
    ocr_metrics = None
    business_metrics = None

    if "classifier" in components:
        logger.info("Evaluating classifier...")
        classifier = create_classifier()
        classifier_metrics = evaluate_classifier(classifier, all_annotations)

    if "detector" in components:
        logger.info("Evaluating detector...")
        detector = create_detector()
        detector_metrics = evaluate_detector(
            detector,
            all_annotations,
            iou_threshold=args.iou_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
        )

    if "ocr" in components:
        logger.info("Evaluating OCR...")
        ocr = create_ocr()
        ocr_metrics = evaluate_ocr(ocr, all_annotations)

    if "business" in components:
        logger.info("Computing business metrics...")
        reader = DocReader()
        business_metrics = compute_business_metrics(
            reader,
            all_annotations,
            key_zones=KEY_ZONES,
        )

    print_report(
        classifier_metrics=classifier_metrics,
        detector_metrics=detector_metrics,
        ocr_metrics=ocr_metrics,
        business_metrics=business_metrics,
    )


if __name__ == "__main__":
    main()