"""
Фабричные функции для создания компонентов со стандартными весами.

Использование:
    from docreader import create_classifier, create_detector, create_ocr

    clf = create_classifier()
    clf = create_classifier(confidence_threshold=0.5) # Переопределение

    det = create_detector()
    det = create_detector(device="cuda")

    ocr = create_ocr()
    ocr = create_ocr(lang=["en", "ru"])
"""

from docreader.config import PipelineConfig
from docreader.hub import ensure_model
from docreader.classifier.yolo_classifier import DocClassifier
from docreader.detector.yolo_obb import ZoneDetector
from docreader.ocr.easyocr_engine import TextRecognizer
from docreader.resolver.lvl_resolver import LvlSubtypeResolver


def create_classifier(
    config: PipelineConfig | None = None,
    **kwargs,
) -> DocClassifier:
    """
    Создаёт классификатор документов со стандартными весами.

    Веса скачиваются автоматически при первом вызове.

    Args:
        config: конфигурация (если None — используется дефолтная).
        **kwargs: переопределение параметров DocClassifier
            (weights_path, device, confidence_threshold).

    Returns:
        Готовый к работе DocClassifier.

    Примеры:
        clf = create_classifier()
        clf = create_classifier(confidence_threshold=0.5)
        clf = create_classifier(device="cuda")
    """
    cfg = config or PipelineConfig()

    defaults = {
        "weights_path": str(ensure_model(cfg.classifier_weights)),
        "device": cfg.resolve_device(),
        "confidence_threshold": cfg.classifier_confidence
    }
    defaults.update(kwargs)
    return DocClassifier(**defaults)


def create_detector(
    config: PipelineConfig | None = None,
    **kwargs,
) -> ZoneDetector:
    """
    Создаёт детектор зон документов со стандартными весами.

    Args:
        config: конфигурация (если None — используется дефолтная).
        **kwargs: переопределение параметров ZoneDetector
            (weights_map, device, confidence_threshold).

    Returns:
        Готовый к работе ZoneDetector.

    Примеры:
        det = create_detector()
        det = create_detector(device="cuda")
        det = create_detector(confidence_threshold=0.1)
    """
    cfg = config or PipelineConfig()

    weights_map = {
        doc_type: str(ensure_model(filename))
        for doc_type, filename in cfg.detector_weights.items()
    }

    defaults = {
        "weights_map": weights_map,
        "device": cfg.resolve_device(),
        "confidence_threshold": cfg.detector_confidence,
    }
    defaults.update(kwargs)
    return ZoneDetector(**defaults)


def create_ocr(
    config: PipelineConfig | None = None,
    **kwargs,
) -> TextRecognizer:
    """
    Создаёт OCR-движок со стандартными моделями.

    Args:
        config: конфигурация (если None — используется дефолтная).
        **kwargs: переопределение параметров TextRecognizer
            (lang, gpu, model_storage_directory, и т.д.).

    Returns:
        Готовый к работе TextRecognizer.

    Примеры:
        ocr = create_ocr()
        ocr = create_ocr(lang=["en", "ru"])
        ocr = create_ocr(gpu=False)
    """
    cfg = config or PipelineConfig()
    easyocr_dir = ensure_model(cfg.ocr_model_archive)

    defaults = {
        "lang": cfg.ocr_lang,
        "gpu": cfg.resolve_device != "cpu",
        "model_storage_directory": str(
            easyocr_dir / cfg.ocr_model_subdir
        ),
        "user_network_directory": str(
            easyocr_dir / cfg.ocr_network_subdir
        ),
        "recog_network": cfg.ocr_recog_network,
        "download_enabled": cfg.ocr_download_enabled,
    }
    defaults.update(kwargs)
    return TextRecognizer(**defaults)


def create_resolver(
    config: PipelineConfig | None = None,
    ocr_engine: TextRecognizer | None = None,
    **kwargs
) -> LvlSubtypeResolver:
    """
    Создаёт resolver подтипа документа (attestat/diplom)

    Args:
        config: конфигурация (если None — используется дефолтная).
        ocr_engine: готовый OCR-движок (если None - создаётся новый).
        **kwargs: переопределение параметров TextRecognizer
            (weights_path, match_threshold, detector_confidence, device).

    Returns:
        Готовый к работе LvlSubtypeResolver.

    Примеры:
        resolver = create_resolver()
        resolver = create_resolver(match_threshold=70.0)
        resolver = create_resolver(ocr_engine=my_ocr)
    """
    cfg = config or PipelineConfig()
    ocr = ocr_engine or create_ocr(cfg)

    defaults = {
        "weights_path": cfg.resolver_weights,
        "ocr_engine": ocr_engine,
        "subtype_keywords": cfg.resolver_subtype_keywords,
        "fuzzy_threshold": cfg.resolver_fuzzy_threshold,
        "confidence_threshold": cfg.resolver_confidence,
        "fallback": cfg.resolver_fallback,
        "device": cfg.resolve_device(),
    }
    defaults.update(kwargs)
    return LvlSubtypeResolver(**defaults)
