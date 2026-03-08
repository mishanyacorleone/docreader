"""Детектор зон документа на основе YOLO OBB."""

import logging

import numpy as np
from ultralytics import YOLO

from docreader.detector.base import BaseDetector, Detection

logger = logging.getLogger(__name__)


class ZoneDetector(BaseDetector):
    """
    Детектор полей документа через YOLO OBB с ленивой загрузкой.

    Примеры:
        det = ZoneDetector(weights_map={
            "passport": "/path/to/passport.pt",
            "diplom": "/path/to/diplom.pt",
        })
        zones = det.detect(image, doc_type="passport")

    Args:
        weights_map: словарь {doc_type: полный_путь_к_весам}.
        device: устройство ("cpu", "cuda").
        confidence_threshold: минимальная уверенность.
    """

    def __init__(
        self,
        weights_map: dict[str, str],
        device: str = "cpu",
        confidence_threshold: float = 0.25,
    ):
        self._weights_map = weights_map
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._loaded_models: dict[str, YOLO] = {}

        logger.info(
            f"ZoneDetector initialized: device={self._device}, "
            f"doc_types={list(self._weights_map.keys())}"
        )

    def _get_model(self, doc_type: str) -> YOLO:
        """Загружает модель при первом обращении."""
        if doc_type not in self._loaded_models:
            if doc_type not in self._weights_map:
                raise ValueError(
                    f"No weights for doc_type='{doc_type}'. "
                    f"Available: {list(self._weights_map.keys())}"
                )
            path = self._weights_map[doc_type]
            logger.info(f"Loading YOLO model for '{doc_type}': {path}")
            self._loaded_models[doc_type] = YOLO(path)
        return self._loaded_models[doc_type]

    @property
    def supported_doc_types(self) -> list[str]:
        """Список поддерживаемых типов документов."""
        return list(self._weights_map.keys())

    def detect(self, image: np.ndarray, doc_type: str) -> list[Detection]:
        """
        Обнаруживает зоны на изображении документа.

        Args:
            image: BGR изображение (кроп одного документа).
            doc_type: тип документа.

        Returns:
            Список Detection.
        """
        model = self._get_model(doc_type)
        results = model(image, device=self._device, verbose=False)

        detections = []
        if results[0].obb is None:
            return detections

        for det in results[0].obb:
            confidence = float(det.conf.cpu())
            if confidence < self._confidence_threshold:
                continue

            zone_id = int(det.cls.cpu())
            zone_name = model.names[zone_id]
            obb_points = det.xyxyxyxy.cpu().numpy().flatten()

            detections.append(Detection(
                zone_name=zone_name,
                obb_points=obb_points,
                confidence=confidence,
            ))

        return detections