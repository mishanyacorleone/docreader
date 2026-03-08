"""Классификатор документов на основе YOLO OBB."""

import logging
from typing import Optional

import numpy as np
from ultralytics import YOLO

from docreader.classifier.base import BaseClassifier, ClassifiedDocument
from docreader.preprocessing.geometry import crop_obb_region

logger = logging.getLogger(__name__)


class DocClassifier(BaseClassifier):
    """
    Классификатор документов через YOLO OBB.

    Находит один или несколько документов на фотографии,
    определяет тип каждого и вырезает кроп.

    Примеры:
        # Стандартные веса (путь передаётся из пайплайна или вручную)
        clf = DocClassifier(weights_path="/path/to/doc_classifier.pt")
        docs = clf.classify(image)

        # Без параметров — используется через DocReader,
        # который сам подставит путь из конфига
        reader = DocReader()

    Args:
        weights_path: путь к файлу весов YOLO.
        device: устройство ("cpu", "cuda").
        confidence_threshold: минимальная уверенность детекции.
    """

    def __init__(
        self,
        weights_path: str,
        device: str = "cpu",
        confidence_threshold: float = 0.3,
    ):
        self._confidence_threshold = confidence_threshold
        self._device = device
        self._model = YOLO(weights_path)

        logger.info(
            f"DocClassifier initialized: device={self._device}, "
            f"weights={weights_path}, "
            f"classes={list(self._model.names.values())}"
        )

    def classify(self, image: np.ndarray) -> list[ClassifiedDocument]:
        """
        Находит документы на изображении.

        Args:
            image: BGR изображение.

        Returns:
            Список ClassifiedDocument. Пустой, если документы не найдены.
        """
        results = self._model(image, device=self._device, verbose=False)

        documents = []

        if results[0].obb is None:
            logger.info("No documents detected")
            return documents

        for det in results[0].obb:
            confidence = float(det.conf.cpu())
            if confidence < self._confidence_threshold:
                continue

            class_id = int(det.cls.cpu())
            doc_type = self._model.names[class_id]
            obb_points = det.xyxyxyxy.cpu().numpy().reshape(4, 2)

            crop = crop_obb_region(image, obb_points)
            if crop is None or crop.size == 0:
                logger.warning(
                    f"Failed to crop document: type={doc_type}, "
                    f"conf={confidence:.3f}"
                )
                continue

            documents.append(ClassifiedDocument(
                doc_type=doc_type,
                confidence=confidence,
                crop=crop,
                obb_points=obb_points,
            ))

        logger.info(
            f"Found {len(documents)} document(s): "
            f"{[d.doc_type for d in documents]}"
        )
        return documents

    @property
    def class_names(self) -> list[str]:
        """Список поддерживаемых типов документов."""
        return list(self._model.names.values())
    