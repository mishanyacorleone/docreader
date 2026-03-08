"""
docreader — библиотека для распознавания текста с документов.

Быстрый старт:
    from docreader import DocReader
    reader = DocReader()
    result = reader.process("photo.jpg")
    for doc in result.documents:
        print(doc.doc_type, doc.fields)

Standalone-компоненты (с автозагрузкой весов):
    from docreader import create_classifier, create_detector, create_ocr

    clf = create_classifier()
    det = create_detector()
    ocr = create_ocr()

Standalone-компоненты (со своими весами):
    from docreader import DocClassifier, ZoneDetector, TextRecognizer

    clf = DocClassifier(weights_path="/my/model.pt")
"""

from docreader.pipeline import DocReader
from docreader.schemas import DocumentResult, ZoneResult, PageResult
from docreader.config import PipelineConfig

# Классы компонентов (для кастомных весов)
from docreader.classifier import DocClassifier, BaseClassifier, ClassifiedDocument
from docreader.detector import ZoneDetector, BaseDetector, Detection
from docreader.ocr import TextRecognizer, BaseOcrEngine, OcrResult

# Фабрики (со стандартными весами)
from docreader.factory import create_classifier, create_detector, create_ocr

__all__ = [
    # Пайплайн
    "DocReader",
    "PipelineConfig",

    # Результаты
    "PageResult",
    "DocumentResult",
    "ZoneResult",

    # Фабрики (стандартные веса)
    "create_classifier",
    "create_detector",
    "create_ocr",

    # Классы компонентов (кастомные веса)
    "DocClassifier",
    "ZoneDetector",
    "TextRecognizer",

    # Базовые классы (для наследования)
    "BaseClassifier",
    "ClassifiedDocument",
    "BaseDetector",
    "Detection",
    "BaseOcrEngine",
    "OcrResult",
]

__version__ = "0.2.0"