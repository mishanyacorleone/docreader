"""
Сборка итогового отчёта и подсчёт бизнес-метрик.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from docreader.pipeline import DocReader
from docreader.utils import load_image
from eval.base import (
    BusinessMetrics,
    ClassifierMetrics,
    DetectorMetrics,
    ImageAnnotation,
    OcrMetrics,
)

logger = logging.getLogger(__name__)


def compute_business_metrics(
    reader: DocReader,
    annotations: list[ImageAnnotation],
    key_zones: Optional[dict[str, list[str]]] = None,
) -> BusinessMetrics:
    """
    Считает бизнес-метрики прогоняя полный пайплайн на GT-изображениях.

    Args:
        reader: готовый DocReader.
        annotations: список ImageAnnotation с ground truth.
        key_zones: {doc_type: [zone_name, ...]} — какие зоны считать ключевыми.
                   Если None — все зоны с текстом считаются ключевыми.

    Returns:
        BusinessMetrics.
    """
    per_doc_type: dict[str, dict[str, list]] = defaultdict(
        lambda: {"full_correct": [], "field_correct": [], "field_total": []}
    )

    for ann in annotations:
        image = load_image(ann.filename)
        page_result = reader.process(image)

        if not page_result.documents:
            per_doc_type[ann.doc_type]["full_correct"].append(0)
            per_doc_type[ann.doc_type]["field_correct"].append(0)
            per_doc_type[ann.doc_type]["field_total"].append(
                len([z for z in ann.zones if z.text is not None])
            )
            continue

        doc_result = page_result.documents[0]
        predicted_fields = doc_result.fields

        # Зоны для проверки
        gt_zones = [z for z in ann.zones if z.text is not None]
        if key_zones and ann.doc_type in key_zones:
            gt_zones = [z for z in gt_zones if z.zone_name in key_zones[ann.doc_type]]

        if not gt_zones:
            continue

        # Считаем совпадения
        field_correct = sum(
            1
            for z in gt_zones
            if predicted_fields.get(z.zone_name, "").strip() == z.text.strip()
        )
        full_correct = 1 if field_correct == len(gt_zones) else 0

        per_doc_type[ann.doc_type]["full_correct"].append(full_correct)
        per_doc_type[ann.doc_type]["field_correct"].append(field_correct)
        per_doc_type[ann.doc_type]["field_total"].append(len(gt_zones))

    # Агрегируем
    all_full_correct: list[int] = []
    all_field_correct: list[int] = []
    all_field_total: list[int] = []
    per_doc_summary: dict[str, dict] = {}

    for doc_type, values in per_doc_type.items():
        full_acc = float(np.mean(values["full_correct"])) if values["full_correct"] else 0.0
        total_fields = sum(values["field_total"])
        correct_fields = sum(values["field_correct"])
        field_acc = correct_fields / total_fields if total_fields > 0 else 0.0

        per_doc_summary[doc_type] = {
            "full_document_accuracy": round(full_acc, 4),
            "field_accuracy": round(field_acc, 4),
            "total_documents": len(values["full_correct"]),
            "total_fields": total_fields,
        }

        all_full_correct.extend(values["full_correct"])
        all_field_correct.extend(values["field_correct"])
        all_field_total.extend(values["field_total"])

    full_doc_acc = float(np.mean(all_full_correct)) if all_full_correct else 0.0
    total_f = sum(all_field_total)
    field_acc_global = sum(all_field_correct) / total_f if total_f > 0 else 0.0

    return BusinessMetrics(
        full_document_accuracy=round(full_doc_acc, 4),
        field_accuracy=round(field_acc_global, 4),
        document_error_rate=round(1.0 - full_doc_acc, 4),
        per_doc_type=per_doc_summary,
    )


def print_report(
    classifier_metrics: Optional[ClassifierMetrics] = None,
    detector_metrics: Optional[DetectorMetrics] = None,
    ocr_metrics: Optional[OcrMetrics] = None,
    business_metrics: Optional[BusinessMetrics] = None,
) -> None:
    """Печатает итоговый отчёт в stdout."""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    if classifier_metrics:
        print("\n── CLASSIFIER ──────────────────────────────────────────")
        print(f"  accuracy : {classifier_metrics.accuracy:.4f}")
        print(f"  total    : {classifier_metrics.total}")
        print()
        header = f"  {'class':<20} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}"
        print(header)
        print("  " + "-" * 54)
        for cls, m in sorted(classifier_metrics.per_class.items()):
            print(
                f"  {cls:<20} {m['precision']:>10.4f} "
                f"{m['recall']:>10.4f} {m['f1']:>10.4f} "
                f"{m['support']:>10}"
            )
        print()
        print("  Confusion matrix:")
        all_cls = sorted(classifier_metrics.confusion_matrix.keys())
        header_cm = "  " + " " * 20 + "".join(f"{c:>12}" for c in all_cls)
        print(header_cm)
        for true_cls in all_cls:
            row = classifier_metrics.confusion_matrix.get(true_cls, {})
            row_str = "".join(
                f"{row.get(pred_cls, 0):>12}" for pred_cls in all_cls
            )
            print(f"  {true_cls:<20}{row_str}")

    if detector_metrics:
        print("\n── DETECTOR ────────────────────────────────────────────")
        print(f"  mAP@50   : {detector_metrics.map50:.4f}")
        print()
        header = f"  {'zone':<25} {'precision':>10} {'recall':>10} {'f1':>8} {'iou':>8} {'support':>8}"
        print(header)
        print("  " + "-" * 67)
        for zone_name, m in sorted(detector_metrics.per_zone.items()):
            print(
                f"  {zone_name:<25} {m.precision:>10.4f} "
                f"{m.recall:>10.4f} {m.f1:>8.4f} "
                f"{m.iou_mean:>8.4f} {m.support:>8}"
            )

    if ocr_metrics:
        print("\n── OCR ─────────────────────────────────────────────────")
        print(f"  CER mean        : {ocr_metrics.cer_mean:.4f}")
        print(f"  WER mean        : {ocr_metrics.wer_mean:.4f}")
        print(f"  Exact match mean: {ocr_metrics.exact_match_mean:.4f}")
        print()
        header = f"  {'zone':<25} {'CER':>8} {'WER':>8} {'exact':>8} {'support':>8}"
        print(header)
        print("  " + "-" * 57)
        for zone_name, m in sorted(ocr_metrics.per_zone.items()):
            print(
                f"  {zone_name:<25} {m.cer:>8.4f} "
                f"{m.wer:>8.4f} {m.exact_match:>8.4f} "
                f"{m.support:>8}"
            )

    if business_metrics:
        print("\n── BUSINESS METRICS ────────────────────────────────────")
        print(f"  Full document accuracy : {business_metrics.full_document_accuracy:.4f}")
        print(f"  Field accuracy         : {business_metrics.field_accuracy:.4f}")
        print(f"  Document error rate    : {business_metrics.document_error_rate:.4f}")
        if business_metrics.per_doc_type:
            print()
            header = f"  {'doc_type':<20} {'full_acc':>10} {'field_acc':>10} {'docs':>8} {'fields':>8}"
            print(header)
            print("  " + "-" * 56)
            for doc_type, m in sorted(business_metrics.per_doc_type.items()):
                print(
                    f"  {doc_type:<20} "
                    f"{m['full_document_accuracy']:>10.4f} "
                    f"{m['field_accuracy']:>10.4f} "
                    f"{m['total_documents']:>8} "
                    f"{m['total_fields']:>8}"
                )

    print("\n" + "=" * 60 + "\n")