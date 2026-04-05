"""
Evaluation OCR-движка.

OCR оценивается независимо от детектора:
кропы вырезаются по GT-координатам, а не по предсказаниям детектора.
"""

import logging
from collections import defaultdict

import numpy as np

from docreader.ocr.base import BaseOcrEngine
from docreader.preprocessing.geometry import crop_obb_region
from docreader.utils import load_image
from eval.base import ImageAnnotation, OcrMetrics, ZoneOcrMetrics
from eval.parsing import denormalize_points

logger = logging.getLogger(__name__)


def evaluate_ocr(
    ocr_engine: BaseOcrEngine,
    annotations: list[ImageAnnotation],
) -> OcrMetrics:
    """
    Оценивает OCR на размеченном датасете.

    Для каждой зоны с ground truth текстом:
      1. Вырезает кроп по GT-координатам
      2. Прогоняет через OCR
      3. Сравнивает с GT-текстом через CER/WER/exact match

    Зоны без текстовой аннотации (text=None) пропускаются.

    Args:
        ocr_engine: готовый к работе OCR-движок.
        annotations: список ImageAnnotation с ground truth.

    Returns:
        OcrMetrics с per-zone CER/WER/exact_match.
    """
    # {zone_name: {"cer": [], "wer": [], "exact": []}}
    stats: dict[str, dict[str, list]] = defaultdict(
        lambda: {"cer": [], "wer": [], "exact": []}
    )

    for ann in annotations:
        image = load_image(ann.filename)
        h, w = image.shape[:2]

        for zone in ann.zones:
            if zone.text is None:
                continue

            pixel_points = denormalize_points(zone.obb_points, w, h)
            crop = crop_obb_region(image, np.array(pixel_points))

            if crop is None or crop.size == 0:
                logger.warning(
                    f"Empty crop for zone '{zone.zone_name}' "
                    f"in '{ann.filename}'"
                )
                continue

            ocr_result = ocr_engine.recognize(crop)
            pred_text = ocr_result.text.strip()
            gt_text = zone.text.strip()

            cer = _compute_cer(gt_text, pred_text)
            wer = _compute_wer(gt_text, pred_text)
            exact = 1.0 if pred_text == gt_text else 0.0

            stats[zone.zone_name]["cer"].append(cer)
            stats[zone.zone_name]["wer"].append(wer)
            stats[zone.zone_name]["exact"].append(exact)

            logger.debug(
                f"zone='{zone.zone_name}' | "
                f"gt='{gt_text}' | pred='{pred_text}' | "
                f"CER={cer:.3f} WER={wer:.3f}"
            )

    return _compute_ocr_metrics(stats, total_images=len(annotations))


def _compute_ocr_metrics(
    stats: dict[str, dict[str, list]],
    total_images: int,
) -> OcrMetrics:
    """Считает финальные метрики из накопленных stats."""
    per_zone: dict[str, ZoneOcrMetrics] = {}

    all_cer: list[float] = []
    all_wer: list[float] = []
    all_exact: list[float] = []

    for zone_name, values in stats.items():
        cer_mean = float(np.mean(values["cer"])) if values["cer"] else 0.0
        wer_mean = float(np.mean(values["wer"])) if values["wer"] else 0.0
        exact_mean = float(np.mean(values["exact"])) if values["exact"] else 0.0

        per_zone[zone_name] = ZoneOcrMetrics(
            zone_name=zone_name,
            cer=round(cer_mean, 4),
            wer=round(wer_mean, 4),
            exact_match=round(exact_mean, 4),
            support=len(values["cer"]),
        )

        all_cer.extend(values["cer"])
        all_wer.extend(values["wer"])
        all_exact.extend(values["exact"])

    return OcrMetrics(
        per_zone=per_zone,
        cer_mean=round(float(np.mean(all_cer)) if all_cer else 0.0, 4),
        wer_mean=round(float(np.mean(all_wer)) if all_wer else 0.0, 4),
        exact_match_mean=round(float(np.mean(all_exact)) if all_exact else 0.0, 4),
        total_images=total_images,
    )


# ---------------------------------------------------------------------------
# CER / WER
# ---------------------------------------------------------------------------

def _compute_cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate = edit_distance(ref, hyp) / len(ref).
    Возвращает 0.0 если reference пустой.
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    distance = _levenshtein(list(reference), list(hypothesis))
    return min(distance / len(reference), 1.0)


def _compute_wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate = edit_distance(ref_words, hyp_words) / len(ref_words).
    Возвращает 0.0 если reference пустой.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    distance = _levenshtein(ref_words, hyp_words)
    return min(distance / len(ref_words), 1.0)


def _levenshtein(seq_a: list, seq_b: list) -> int:
    """Вычисляет расстояние Левенштейна между двумя последовательностями."""
    len_a, len_b = len(seq_a), len(seq_b)
    dp = list(range(len_b + 1))

    for i in range(1, len_a + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, len_b + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)

    return dp[len_b]