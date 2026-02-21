"""Абстрактный интерфейс классификатора документов."""

from abc import ABC, abstractmethod
import numpy as np


class BaseClassifier(ABC):
    """
    Интерфейс для классификатора типа документа.

    Чтобы подключить свой классификатор, наследуйтесь от этого класса
    и реализуйте метод 'predict'.
    """

    @abstractmethod
    def predict(self, image: np.ndarray) -> tuple[str, float]:
        """
        Классифицирует изображение документа.

        Args:
            image: BGR изображение (numpy array).

        Returns:
            Кортеж (метка_класса, уверенность).
        """
        ...