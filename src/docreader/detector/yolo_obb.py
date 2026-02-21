"""Детектор зон документа на основе YOLOv8 OBB."""

import os
import numpy as np
from ultralytics import YOLO

from docreader.detector.base import BaseDetector, Detection


class YoloObbDetector(BaseDetector):
    """
    Детектор полей документа через YOLOv8 с ориентированными боксами.

    Args:
        models_dir: директория с весами YOLO.
        weights_map: словарь {тип_документа: имя_файла_весов}.
    """

    def __init__(self, models_dir: str, weights_map: dict[str, str]):
        self._models: dict[str, YOLO] = {}

        for doc_type, filename in weights_map.items():
            path = os.path.join(models_dir, filename)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"YOLO weights not found: {path} (doc_type={doc_type})"
                )
            self._models[doc_type] = YOLO(path)

    @property
    def supported_doc_types(self) -> list[str]:
        return list(self._models.keys())

    def detect(self, image: np.ndarray, doc_type: str) -> list[Detection]:
        if doc_type not in self._models:
            raise ValueError(
                f"No YOLO model for doc_type='{doc_type}'. "
                f"Available: {self.supported_doc_types}"
            )

        model = self._models[doc_type]
        results = model(image)

        detections = []
        for det in results[0].obb:
            zone_id = int(det.cls)
            zone_name = model.names[zone_id]
            obb_points = det.xyxyxyxy.cpu().numpy().flatten()
            confidence = float(det.conf.cpu())

            detections.append(Detection(
                zone_name=zone_name,
                obb_points=obb_points,
                confidence=confidence,
            ))

        return detections