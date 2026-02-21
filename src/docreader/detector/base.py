"""Абстрактный интерфейс детектора зон документа."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """Одна обнаруженная зона."""
    zone_name: str
    obb_points: np.ndarray   # shape (4, 2) или (8,)
    confidence: float


class BaseDetector(ABC):
    """
    Интерфейс для детектора полей документа.

    Чтобы подключить свой детектор, наследуйтесь и реализуйте `detect`.
    """

    @abstractmethod
    def detect(self, image: np.ndarray, doc_type: str) -> list[Detection]:
        """
        Обнаруживает зоны (поля) на изображении документа.

        Args:
            image: BGR изображение.
            doc_type: тип документа (для выбора нужной модели).

        Returns:
            Список обнаруженных зон.
        """
        ...