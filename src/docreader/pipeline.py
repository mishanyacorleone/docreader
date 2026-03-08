"""
Главный пайплайн распознавания документов.
"""

import logging
from typing import Optional, Union

import numpy as np

from docreader.config import PipelineConfig
from docreader.schemas import DocumentResult, ZoneResult, PageResult
from docreader.utils import load_image
from docreader.hub import ensure_model
from docreader.preprocessing import deskew_image, crop_obb_region

from docreader.classifier.base import BaseClassifier

from docreader.detector.base import BaseDetector

from docreader.ocr.base import BaseOcrEngine

logger = logging.getLogger(__name__)

ImageSource = Union[str, np.ndarray]


class DocReader:
    """
    Полный пайплайн распознавания документов.

    Примеры:
        reader = DocReader()
        result = reader.process("photo.jpg")
        for doc in result.documents:
            print(doc.doc_type, doc.fields)

    Args:
        config: конфигурация пайплайна.
        classifier: кастомный классификатор (если None — из конфига).
        detector: кастомный детектор зон (если None — из конфига).
        ocr_engine: кастомный OCR-движок (если None — из конфига).
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        classifier: Optional[BaseClassifier] = None,
        detector: Optional[BaseDetector] = None,
        ocr_engine: Optional[BaseOcrEngine] = None,
    ):
        self._config = config or PipelineConfig()
        self._device = self._config.resolve_device()

        logger.info(f"DocReader init: device={self._device}")

        self._classifier = classifier or self._build_classifier()
        self._detector = detector or self._build_detector()
        self._ocr = ocr_engine or self._build_ocr()

    def _build_classifier(self) -> BaseClassifier:
        """Создаёт классификатор из конфига."""
        from docreader.classifier.yolo_classifier import DocClassifier

        weights_path = str(ensure_model(self._config.classifier_weights))

        return DocClassifier(
            weights_path=weights_path,
            device=self._device,
            confidence_threshold=self._config.classifier_confidence,
        )

    def _build_detector(self) -> BaseDetector:
        """Создаёт детектор зон из конфига."""
        from docreader.detector.yolo_obb import ZoneDetector

        weights_map = {}
        for doc_type, filename in self._config.detector_weights.items():
            weights_map[doc_type] = str(ensure_model(filename))

        return ZoneDetector(
            weights_map=weights_map,
            device=self._device,
            confidence_threshold=self._config.detector_confidence,
        )

    def _build_ocr(self) -> BaseOcrEngine:
        """Создаёт OCR-движок из конфига."""
        from docreader.ocr.easyocr_engine import TextRecognizer

        easyocr_dir = ensure_model(self._config.ocr_model_archive)

        return TextRecognizer(
            lang=self._config.ocr_lang,
            gpu=(self._device != "cpu"),
            model_storage_directory=str(
                easyocr_dir / self._config.ocr_model_subdir
            ),
            user_network_directory=str(
                easyocr_dir / self._config.ocr_network_subdir
            ),
            recog_network=self._config.ocr_recog_network,
            download_enabled=self._config.ocr_download_enabled,
        )

    # === Публичный API ===

    def process(
        self,
        source: ImageSource,
        return_crops: Optional[bool] = None,
    ) -> PageResult:
        """
        Полный пайплайн: находит все документы и распознаёт.

        Args:
            source: путь к файлу или numpy array (BGR).
            return_crops: сохранять ли кропы.

        Returns:
            PageResult со списком найденных документов.
        """
        save_crops = (
            return_crops
            if return_crops is not None
            else self._config.return_crops
        )

        image = load_image(source)

        # 1. Классификация
        classified_docs = self._classifier.classify(image)

        if not classified_docs:
            logger.info("No documents found")
            return PageResult(documents=[])

        # 2. Обработка каждого документа
        documents: list[DocumentResult] = []
        for doc in classified_docs:
            result = self._process_single_document(
                doc_image=doc.crop,
                doc_type=doc.doc_type,
                doc_confidence=doc.confidence,
                doc_bbox=doc.obb_points,
                save_crops=save_crops,
            )
            documents.append(result)

        page_result = PageResult(documents=documents)
        logger.info(f"Complete: {page_result}")
        return page_result

    def process_batch(
        self,
        sources: list[ImageSource],
        return_crops: Optional[bool] = None,
    ) -> list[PageResult]:
        """Обработка нескольких фотографий."""
        return [self.process(src, return_crops) for src in sources]

    # === Внутренняя логика ===

    def _process_single_document(
        self,
        doc_image: np.ndarray,
        doc_type: str,
        doc_confidence: float,
        doc_bbox: np.ndarray,
        save_crops: bool,
    ) -> DocumentResult:
        """Обрабатывает один документ."""

        if doc_type not in self._detector.supported_doc_types:
            logger.warning(f"No detector for '{doc_type}'")
            return DocumentResult(
                doc_type=doc_type,
                doc_confidence=doc_confidence,
                zones=[],
                doc_bbox=doc_bbox.tolist(),
                doc_crop=doc_image if save_crops else None,
            )

        if self._config.enable_deskew:
            doc_image = deskew_image(doc_image)

        detections = self._detector.detect(doc_image, doc_type)
        logger.info(f"'{doc_type}': {len(detections)} zones")

        zones: list[ZoneResult] = []
        for det in detections:
            zone = self._process_zone(doc_image, det, save_crops)
            if zone is not None:
                zones.append(zone)

        return DocumentResult(
            doc_type=doc_type,
            doc_confidence=doc_confidence,
            zones=zones,
            doc_bbox=doc_bbox.tolist(),
            doc_crop=doc_image if save_crops else None,
        )

    def _process_zone(self, image, detection, save_crops):
        """Обрабатывает одну зону."""

        zone_name = detection.zone_name

        if zone_name in self._config.skip_ocr_zones:
            return ZoneResult(
                name=zone_name,
                text="",
                confidence=detection.confidence,
                bbox=detection.obb_points.tolist(),
            )

        crop = crop_obb_region(image, detection.obb_points)
        if crop is None or crop.size == 0:
            logger.warning(f"Empty crop for '{zone_name}'")
            return None

        ocr_result = self._ocr.recognize(crop)

        return ZoneResult(
            name=zone_name,
            text=ocr_result.text,
            confidence=ocr_result.confidence,
            bbox=detection.obb_points.tolist(),
            crop_image=crop if save_crops else None,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Освобождает ресурсы."""
        self._classifier = None
        self._detector = None
        self._ocr = None
        try:
            import gc
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass