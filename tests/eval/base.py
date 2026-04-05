"""
Базовые структуры данных для evaluation.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ZoneAnnotation:
    """Ground truth для одной зоны документа."""
    zone_name: str
    obb_points: list[float]   # [x1,y1,x2,y2,x3,y3,x4,y4]
    text: Optional[str] = None


@dataclass
class ImageAnnotation:
    """Ground truth для одного изображения."""
    filename: str
    doc_type: str
    zones: list[ZoneAnnotation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Метрики классификатора
# ---------------------------------------------------------------------------

@dataclass
class ClassifierMetrics:
    """Метрики классификатора по всему датасету."""
    accuracy: float
    per_class: dict[str, dict]   # {doc_type: {precision, recall, f1, support}}
    confusion_matrix: dict       # {true_label: {pred_label: count}}
    total: int


# ---------------------------------------------------------------------------
# Метрики детектора
# ---------------------------------------------------------------------------

@dataclass
class ZoneDetectorMetrics:
    """Метрики детектора для одной зоны."""
    zone_name: str
    precision: float
    recall: float
    f1: float
    iou_mean: float
    support: int             # сколько раз зона встречается в GT


@dataclass
class DetectorMetrics:
    """Метрики детектора по всему датасету."""
    per_zone: dict[str, ZoneDetectorMetrics]   # {zone_name: metrics}
    map50: float                               # mean AP@IoU=0.50
    total_images: int


# ---------------------------------------------------------------------------
# Метрики OCR
# ---------------------------------------------------------------------------

@dataclass
class ZoneOcrMetrics:
    """Метрики OCR для одной зоны."""
    zone_name: str
    cer: float          # Character Error Rate
    wer: float          # Word Error Rate
    exact_match: float  # доля точных совпадений (0.0–1.0)
    support: int        # количество примеров


@dataclass
class OcrMetrics:
    """Метрики OCR по всему датасету."""
    per_zone: dict[str, ZoneOcrMetrics]   # {zone_name: metrics}
    cer_mean: float
    wer_mean: float
    exact_match_mean: float
    total_images: int


# ---------------------------------------------------------------------------
# Бизнес-метрики (расширяемые)
# ---------------------------------------------------------------------------

@dataclass
class BusinessMetrics:
    """
    Бизнес-метрики качества пайплайна.

    Расширяй этот класс по мере появления новых требований.
    """
    # Доля документов где ВСЕ ключевые поля распознаны верно (exact match)
    full_document_accuracy: float

    # Доля отдельных полей, распознанных верно (exact match)
    field_accuracy: float

    # Доля документов где хотя бы одно ключевое поле распознано неверно
    document_error_rate: float

    # Детализация по типам документов
    per_doc_type: dict[str, dict] = field(default_factory=dict)

    # Зарезервировано для будущих метрик
    # Например: скорость обработки, процент unresolved документов и т.д.
    extra: dict = field(default_factory=dict)