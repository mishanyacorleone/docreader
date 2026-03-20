"""
Абстрактный интерфейс resolver'a подтипа документа.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ResolveResult:
    """
    Результат определения подтипа документа
    """
    subtype: Optional[str]
    ocr_text: str
    confidence: float
    fuzzy_score: float

    @property
    def resolve(self) -> None:
        return self.subtype is not None


class BaseSubtypeResolver(ABC):
    """
    Интерфейс для определения подтипа документа.

    Используется когда классификатор не может различать 2 похожих
    класса (attestat/diplom) и требуется дополнительный шаг
    """

    @abstractmethod
    def resolve(self, image: np.ndarray) -> ResolveResult:
        """
        Определяет подтип документа по его crop'y
        """
        ...