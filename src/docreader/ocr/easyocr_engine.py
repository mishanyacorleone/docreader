"""OCR-движок на основе EasyOCR."""

import logging
from typing import Optional

import numpy as np
import easyocr

from docreader.ocr.base import BaseOcrEngine, OcrResult

logger = logging.getLogger(__name__)


class TextRecognizer(BaseOcrEngine):
    """
    OCR через EasyOCR.

    Все пути к моделям передаются явно.
    Для автоматической загрузки используйте DocReader.

    Примеры:
        ocr = TextRecognizer(
            lang=["ru"],
            model_storage_directory="/path/to/models",
            user_network_directory="/path/to/networks",
            recog_network="custom_example",
        )
        result = ocr.recognize(image)

    Args:
        lang: список языков.
        gpu: использовать ли GPU.
        model_storage_directory: директория с моделями.
        user_network_directory: директория с кастомными сетями.
        recog_network: имя сети распознавания.
        download_enabled: разрешить скачивание моделей.
    """

    def __init__(
        self,
        lang: list[str],
        model_storage_directory: str,
        user_network_directory: str,
        recog_network: str,
        gpu: bool = False,
        download_enabled: bool = False,
    ):
        kwargs: dict = {
            "lang_list": lang,
            "gpu": gpu,
            "download_enabled": download_enabled,
            "model_storage_directory": model_storage_directory,
            "user_network_directory": user_network_directory,
            "recog_network": recog_network,
            "verbose": False,
        }

        self._reader = easyocr.Reader(**kwargs)
        logger.info(
            f"TextRecognizer initialized: lang={lang}, gpu={gpu}"
        )

    def recognize(self, image: np.ndarray) -> OcrResult:
        """
        Распознаёт текст на изображении.

        Args:
            image: BGR изображение (кроп зоны).

        Returns:
            OcrResult с текстом и средней уверенностью.
        """
        results = self._reader.readtext(image)

        if not results:
            return OcrResult(text="", confidence=0.0)

        texts = []
        confidences = []
        for _, text, conf in results:
            texts.append(text)
            confidences.append(conf)

        combined_text = " ".join(texts).strip()
        mean_confidence = (
            sum(confidences) / len(confidences)
            if confidences
            else 0.0
        )

        return OcrResult(text=combined_text, confidence=mean_confidence)