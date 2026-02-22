"""
Главный пайплайн распознавания документов.
"""

import os
import logging
from typing import Optional
from pathlib import Path

import numpy as np

from docreader.config import PipelineConfig
from docreader.schemas import DocumentResult, ZoneResult
from docreader.utils import load_image
from docreader.hub import ensure_model, get_cache_dir
from docreader.preprocessing import deskew_image, crop_obb_region

from docreader.classifier.base import BaseClassifier
from docreader.classifier.mobilenet import MobileNetClassifier

from docreader.detector.base import BaseDetector
from docreader.detector.yolo_obb import YoloObbDetector

from docreader.ocr.base import BaseOcrEngine
from docreader.ocr.easyocr_engine import EasyOcrEngine

logger = logging.getLogger(__name__)

class DocReader:
    """
    Распознавание текста с документов
    
    Пайплайн:
        1. Классификация типа документа
        2. Выравнивание (опционально)
        3. Детекция полей через YOLO OBB
        4. Кроп каждой зоны
        5. OCR каждого кропа
        6. Сборка результата

    Примеры:
        # Стандартное использование
        reader = DocReader(models_dir="./models")
        result = reader.process("passport.jpg")
        print(result.fields)

        # С кастомным OCR-движком
        my_ocr = MyTesseractEngine()
        reader = DocReader(models_dir="./models", ocr_engine=my_ocr)

        # С кастомным классификактором
        my_cls = MyResNetClassifier(weigths="resnet.pth")
        reader = DocReader(models_dir="./models", classifier=my_cls)
    """

    def __init__(
        self,
        models_dir: str | None,
        config: Optional[PipelineConfig] = None,
        classifier: Optional[BaseClassifier] = None,
        detector: Optional[BaseDetector] = None,
        ocr_engine: Optional[BaseOcrEngine] = None
    ):
        """
        Args:
            models_dir: директория с файлами моделей
            config: конфигурация пайплайна
            classifier: кастомный классификатор (если None - MobileNetV2)
            detector: кастомный детектор (если None - YOLO OBB)
            ocr_engine: кастомный OCR (если None - EasyOCR)
        """

        self._config = config or PipelineConfig()
        self._device = self._config.resolve_device()
        if models_dir is not None:
            self._models_dir = Path(models_dir)
            self._auto_download = False
        else:
            self._models_dir = get_cache_dir()
            self._auto_download = True

        logger.info(f"DocReader init: device={self._device}"
                    f"models_dir={self._models_dir}"
                    f"auto_download={self._auto_download}")

        self._classifier = classifier or self._init_classifier()
        self._detector = detector or self._init_detector()
        self._ocr = ocr_engine or self._init_ocr()

    def _resolve_weights(self, filename: str) -> str:
        """
        Возвращает путь к весам, скачивая при необходимости
        """
        if self._auto_download:
            return str(ensure_model(filename, self._models_dir))
        path = self._models_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        return str(path)
    
    def _init_classifier(self) -> BaseClassifier:
        from docreader.classifier.mobilenet import MobileNetClassifier

        weigths = self._resolve_weights(self._config.classification_weights)
        return MobileNetClassifier(
            weights_path=weigths,
            class_labels=self._config.class_labels,
            device=self._device
        )
    
    def _init_detector(self) -> BaseDetector:
        from docreader.detector.yolo_obb import YoloObbDetector

        resolved = {}
        for doc_type, filename in self._config.detector_weights.items():
            self._resolve_weights(filename)
            resolved[doc_type] = filename
        
        return YoloObbDetector(
            models_dir=str(self._models_dir),
            weights_map=resolved
        )

    def _init_ocr(self) -> BaseOcrEngine:
        from docreader.ocr.easyocr_engine import EasyOcrEngine

        easyocr_dir = ensure_model("easyocr_custom.tar.gz", self._models_dir) 

        return EasyOcrEngine(
            lang=["ru"],
            gpu=(self._device != "cpu"),
            model_storage_directory=str(easyocr_dir / "model"),
            user_network_directory=str(easyocr_dir / "user_network"),
            recog_network="custom_example",
            download_enabled=False
        )
    
    # Публичный API

    def process(
        self,
        source,
        return_crops: Optional[bool] = None,
    ) -> DocumentResult:
        """
        Полный пайплайн распознавания документа.

        Args:
            source: путь к файлу или numpy array (BGR)
            return_crops: сохранять ли кропы зон в результат
        
        Returns:
            DocumentResult
        """

        save_crops = (
            return_crops 
            if return_crops is not None 
            else self._config.return_crops
        )
        image = load_image(source)

        # 1. Классификация
        doc_type, doc_conf = self._classifier.predict(image)
        logger.info(f"Classificated as '{doc_type}' (conf={doc_conf:.3f})")
        
        supported = self._config.detector_weights.keys()
        if doc_type not in supported:
            logger.warning(
                f"Unknown doc_type '{doc_type}', "
                f"supported: {list(supported)}. "
                f"Returning empty result."
            )
            return DocumentResult(
                doc_type=doc_type,
                doc_confidence=doc_conf,
                zones=[]
            )

        if self._config.enable_deskew:
            image = deskew_image(image)

        detections = self._detector.detect(image, doc_type)
        logger.info(f"Detected {len(detections)} zones")

        zones: list[ZoneResult] = []
        for det in detections:
            zone = self._process_zone(image, det, save_crops)
            if zone is not None:
                zones.append(zone)
        
        return DocumentResult(
            doc_type=doc_type,
            doc_confidence=doc_conf,
            zones=zones
        )
    
    def process_batch(
        self,
        sources: list,
        return_crops: Optional[bool] = None 
    ) -> list[DocumentResult]:
        """
        Обработка нескольких документов
        """
        return [self.process(src, return_crops) for src in sources]
    
    # Внутренняя логика
    
    def _process_zone(self, image, detection, save_crops):
        """
        Обрабатывает одну зону: кроп -> OCR -> ZoneResult
        """
        from docreader.detector.base import Detection

        zone_name = detection.zone_name

        if zone_name in self._config.skip_ocr_zones:
            return ZoneResult(
                name=zone_name,
                text="",
                confidence=detection.confidence,
                bbox=detection.obb_points.tolist(),
                crop_image=None
            )
        
        crop = crop_obb_region(image, detection.obb_points)
        if crop is None or crop.size == 0:
            logger.warning(f"Empty crop for zone: '{zone_name}', skipping")
            return None

        
        ocr_result = self._ocr.recognize(crop)

        return ZoneResult(
            name=zone_name,
            text=ocr_result.text,
            confidence=ocr_result.confidence,
            bbox=detection.obb_points.tolist(),
            crop_image=crop if save_crops else None
        )
    