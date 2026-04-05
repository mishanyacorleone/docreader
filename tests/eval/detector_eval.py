"""
Evaluation детектора зон документа.
"""

import logging
from collections import defaultdict

import cv2
import numpy as np

from docreader.detector.base import BaseDetector, Detection
from docreader.utils import load_image
from eval.base import DetectorMetrics, ImageAnnotation, ZoneAnnotation, ZoneDetectorMetrics
from eval.parsing import denormalize_points

logger = logging.getLogger(__name__)

IOU_THRESHOLD = 0.50
NMS_IOU_THRESHOLD = 0.45  # порог IoU для подавления дублей


def evaluate_detector(
    detector: BaseDetector,
    annotations: list[ImageAnnotation],
    iou_threshold: float = IOU_THRESHOLD,
    nms_iou_threshold: float = NMS_IOU_THRESHOLD,
) -> DetectorMetrics:
    """
    Оценивает детектор зон на размеченном датасете.

    Для каждого изображения:
      1. Запускает детектор
      2. Применяет NMS per-class — убирает дублирующиеся боксы
      3. Матчит предсказания с GT по IoU (greedy, один-к-одному)

    Args:
        detector: готовый к работе детектор.
        annotations: список ImageAnnotation с ground truth.
        iou_threshold: порог IoU для True Positive (default 0.50).
        nms_iou_threshold: порог IoU для NMS (default 0.45).

    Returns:
        DetectorMetrics с per-zone precision/recall/F1 и mAP50.
    """
    stats: dict[str, dict] = defaultdict(lambda: {
        "tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0, "support": 0
    })

    for ann in annotations:
        image = load_image(ann.filename)
        h, w = image.shape[:2]

        gt_zones = _denormalize_annotations(ann.zones, w, h)
        predictions = detector.detect(image, ann.doc_type)

        # NMS per-class — убираем дублирующиеся боксы одного класса
        predictions = _apply_nms(predictions, nms_iou_threshold)

        _match_predictions(gt_zones, predictions, stats, iou_threshold)

    return _compute_detector_metrics(stats)


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def _apply_nms(
    detections: list[Detection],
    iou_threshold: float,
) -> list[Detection]:
    """
    Применяет Non-Maximum Suppression отдельно для каждого класса.

    Для каждой группы детекций одного zone_name оставляет только
    те боксы, которые не перекрываются сильно друг с другом.
    Боксы с большей уверенностью имеют приоритет.

    Args:
        detections: список Detection от детектора.
        iou_threshold: боксы с IoU выше порога подавляются.

    Returns:
        Отфильтрованный список Detection.
    """
    if not detections:
        return detections

    # Группируем по классу
    by_class: dict[str, list[Detection]] = defaultdict(list)
    for det in detections:
        by_class[det.zone_name].append(det)

    result: list[Detection] = []
    for zone_name, dets in by_class.items():
        if len(dets) == 1:
            result.extend(dets)
            continue

        # Сортируем по убыванию уверенности
        dets_sorted = sorted(dets, key=lambda d: d.confidence, reverse=True)
        kept: list[Detection] = []

        for candidate in dets_sorted:
            # Проверяем перекрытие с уже оставленными боксами
            suppressed = False
            for kept_det in kept:
                iou = _compute_iou(
                    np.array(candidate.obb_points).reshape(4, 2),
                    np.array(kept_det.obb_points).reshape(4, 2),
                )
                if iou > iou_threshold:
                    suppressed = True
                    break

            if not suppressed:
                kept.append(candidate)

        logger.debug(
            f"NMS '{zone_name}': {len(dets)} → {len(kept)} detections"
        )
        result.extend(kept)

    return result


# ---------------------------------------------------------------------------
# Matching GT vs Predictions
# ---------------------------------------------------------------------------

def _denormalize_annotations(
    zones: list[ZoneAnnotation],
    width: int,
    height: int,
) -> list[ZoneAnnotation]:
    """Конвертирует координаты зон из процентов в пиксели."""
    result = []
    for zone in zones:
        pixel_points = denormalize_points(zone.obb_points, width, height)
        result.append(ZoneAnnotation(
            zone_name=zone.zone_name,
            obb_points=pixel_points,
            text=zone.text,
        ))
    return result


def _match_predictions(
    gt_zones: list[ZoneAnnotation],
    predictions: list[Detection],
    stats: dict,
    iou_threshold: float,
) -> None:
    """
    Матчит предсказания с GT по принципу один-к-одному (greedy по IoU).
    Обновляет stats на месте.

    Для каждого предсказания ищется GT-бокс того же класса с
    максимальным IoU. Каждый GT-бокс может быть использован только раз.
    """
    # Группируем GT по zone_name
    gt_by_class: dict[str, list[ZoneAnnotation]] = defaultdict(list)
    for zone in gt_zones:
        gt_by_class[zone.zone_name].append(zone)
        stats[zone.zone_name]["support"] += 1

    # Группируем предсказания по zone_name
    pred_by_class: dict[str, list[Detection]] = defaultdict(list)
    for pred in predictions:
        pred_by_class[pred.zone_name].append(pred)

    all_classes = set(gt_by_class.keys()) | set(pred_by_class.keys())

    for zone_name in all_classes:
        gt_list = gt_by_class[zone_name]
        pred_list = pred_by_class[zone_name]

        matched_gt: set[int] = set()

        for pred in pred_list:
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_list):
                if gt_idx in matched_gt:
                    continue
                iou = _compute_iou(
                    np.array(pred.obb_points).reshape(4, 2),
                    np.array(gt.obb_points).reshape(4, 2),
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                stats[zone_name]["tp"] += 1
                stats[zone_name]["iou_sum"] += best_iou
                matched_gt.add(best_gt_idx)
            else:
                stats[zone_name]["fp"] += 1

        fn = len(gt_list) - len(matched_gt)
        stats[zone_name]["fn"] += fn


# ---------------------------------------------------------------------------
# IoU для OBB через маски
# ---------------------------------------------------------------------------

def _compute_iou(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """
    Считает IoU двух OBB через polygon masks.

    Использует фиксированный canvas 2000x2000 — достаточно для
    нормализованных координат в пикселях типичных документов.
    """
    h, w = 2000, 2000
    mask_a = np.zeros((h, w), dtype=np.uint8)
    mask_b = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(mask_a, [pts_a.astype(np.int32)], 1)
    cv2.fillPoly(mask_b, [pts_b.astype(np.int32)], 1)

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    return float(intersection / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Финальные метрики
# ---------------------------------------------------------------------------

def _compute_detector_metrics(stats: dict) -> DetectorMetrics:
    """Считает финальные метрики из накопленных stats."""
    per_zone: dict[str, ZoneDetectorMetrics] = {}
    ap_values: list[float] = []

    for zone_name, s in stats.items():
        tp = s["tp"]
        fp = s["fp"]
        fn = s["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        iou_mean = s["iou_sum"] / tp if tp > 0 else 0.0

        per_zone[zone_name] = ZoneDetectorMetrics(
            zone_name=zone_name,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            iou_mean=round(iou_mean, 4),
            support=s["support"],
        )
        ap_values.append(precision)

    map50 = float(np.mean(ap_values)) if ap_values else 0.0
    total_images = len(set())  # заполняется через support

    return DetectorMetrics(
        per_zone=per_zone,
        map50=round(map50, 4),
        total_images=sum(s["support"] for s in stats.values()),
    )