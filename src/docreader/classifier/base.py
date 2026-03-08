"""Абстрактный интерфейс классификатора документов."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class ClassifiedDocument:
    """
    Один найденный документ на изображении
    """
    doc_type: str
    confidence: float
    crop: np.ndarray # вырезанный и выпрямленный документ (BGR)
    obb_points: np.ndarray # shape (4, 2) - координаты в исходном состоянии


class BaseClassifier(ABC):
    """
    Интерфейс для классификатора типа документа.

    Чтобы подключить свой классификатор, наследуйтесь от этого класса
    и реализуйте метод 'predict'.
    """

    @abstractmethod
    def classify(self, image: np.ndarray) -> list[str, float]:
        """
        Классифицирует изображение документа.

        Args:
            image: BGR изображение (numpy array).

        Returns:
            Кортеж (метка_класса, уверенность).
        """
        ...