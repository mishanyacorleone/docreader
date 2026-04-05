"""
Evaluation классификатора документов.
"""

import logging
from collections import defaultdict

from docreader.classifier.base import BaseClassifier
from docreader.utils import load_image
from eval.base import ClassifierMetrics, ImageAnnotation

logger = logging.getLogger(__name__)

# Классы которые классификатор объединяет в один —
# для eval считаем их корректными предсказаниями для обоих GT-классов.
# Например: классификатор выдал "attestat/diplom",
# GT = "attestat" → считаем как правильно на уровне классификатора
# (точное разделение — задача resolver'а, не классификатора)
AMBIGUOUS_CLASS_MAP: dict[str, set[str]] = {
    "attestat/diplom": {"attestat", "diplom"},
}


def evaluate_classifier(
    classifier: BaseClassifier,
    annotations: list[ImageAnnotation],
    ambiguous_map: dict[str, set[str]] | None = None,
) -> ClassifierMetrics:
    """
    Оценивает точность классификатора на размеченном датасете.

    Для неоднозначных классов (attestat/diplom) засчитывает предсказание
    как правильное если GT входит в множество допустимых классов.
    Это корректно — классификатор не обязан различать attestat и diplom,
    это задача resolver'а.

    Args:
        classifier: готовый к работе классификатор.
        annotations: список ImageAnnotation с ground truth.
        ambiguous_map: маппинг {pred_class: {допустимые gt_классы}}.
                       Если None — используется AMBIGUOUS_CLASS_MAP.

    Returns:
        ClassifierMetrics с accuracy, per-class F1 и confusion matrix.
    """
    mapping = ambiguous_map if ambiguous_map is not None else AMBIGUOUS_CLASS_MAP

    y_true: list[str] = []
    y_pred: list[str] = []

    for ann in annotations:
        image = load_image(ann.filename)
        classified = classifier.classify(image)

        if not classified:
            predicted = "__none__"
            logger.warning(f"No document detected: {ann.filename}")
        else:
            best = max(classified, key=lambda d: d.confidence)
            predicted = best.doc_type

        # Нормализуем предсказание для неоднозначных классов:
        # если классификатор выдал "attestat/diplom" а GT = "attestat"
        # → заменяем predicted на gt чтобы не штрафовать классификатор
        # за то что должен делать resolver
        normalized_pred = predicted
        if predicted in mapping and ann.doc_type in mapping[predicted]:
            normalized_pred = ann.doc_type
            logger.debug(
                f"Ambiguous class normalized: "
                f"'{predicted}' → '{normalized_pred}' "
                f"(gt='{ann.doc_type}')"
            )

        y_true.append(ann.doc_type)
        y_pred.append(normalized_pred)

    return _compute_classifier_metrics(y_true, y_pred)


def _compute_classifier_metrics(
    y_true: list[str],
    y_pred: list[str],
) -> ClassifierMetrics:
    """Считает метрики по спискам true/pred меток."""
    classes = sorted(set(y_true) | set(y_pred))
    total = len(y_true)

    # Confusion matrix
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for true, pred in zip(y_true, y_pred):
        confusion[true][pred] += 1

    # Accuracy
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / total if total > 0 else 0.0

    # Per-class precision / recall / F1
    per_class: dict[str, dict] = {}
    for cls in classes:
        tp = confusion[cls][cls]
        fp = sum(confusion[other][cls] for other in classes if other != cls)
        fn = sum(confusion[cls][other] for other in classes if other != cls)
        support = sum(confusion[cls].values())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    return ClassifierMetrics(
        accuracy=round(accuracy, 4),
        per_class=per_class,
        confusion_matrix={k: dict(v) for k, v in confusion.items()},
        total=total,
    )