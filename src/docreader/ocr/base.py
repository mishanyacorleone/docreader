"""Абстрактный интерфейс OCR-движка."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class OcrResult:
    """Результат OCR для одного кропа."""
    text: str
    confidence: float


class BaseOcrEngine(ABC):
    """
    Интерфейс OCR-движка.

    Чтобы подключить Tesseract, PaddleOCR или что-то ещё —
    наследуйтесь и реализуйте `recognize`.
    """

    @abstractmethod
    def recognize(self, image: np.ndarray) -> OcrResult:
        """
        Распознаёт текст на изображении (кропе зоны).

        Args:
            image: BGR изображение с текстом.

        Returns:
            OcrResult с текстом и уверенностью.
        """
        ...