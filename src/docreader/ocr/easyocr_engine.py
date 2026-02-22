"""OCR-движок на основе EasyOCR."""

from typing import Optional
from pathlib import Path

import numpy as np
import easyocr

from docreader.ocr.base import BaseOcrEngine, OcrResult


class EasyOcrEngine(BaseOcrEngine):
    """
    OCR через EasyOCR (с поддержкой кастомных моделей).

    Args:
        lang: список языков, например ["ru"].
        gpu: использовать GPU.
        model_storage_directory: путь к директории с моделями.
        user_network_directory: путь к пользовательским сетям.
        recog_network: имя сети распознавания.
    """

    def __init__(
        self,
        lang: list[str] | None = None,
        gpu: bool = True,
        model_storage_directory: Optional[str] = None,
        user_network_directory: Optional[str] = None,
        recog_network: Optional[str] = None,
        download_enabled: bool = False
    ):
        # Путь для хранения моделей по умолчанию
        if model_storage_directory is None:
            model_storage_directory = str(Path.home() / ".cache" / "docreader" / "easyocr_models")
            Path(model_storage_directory).mkdir(parents=True, exist_ok=True)

        kwargs = {
            "lang_list": lang or ["ru"],
            "gpu": gpu,
            "download_enabled": download_enabled,
            "model_storage_directory": model_storage_directory,
            "verbose": False,
        }

        if user_network_directory:
            kwargs["user_network_directory"] = user_network_directory
        if recog_network:
            kwargs["recog_network"] = recog_network

        self._reader = easyocr.Reader(**kwargs)

    def recognize(self, image: np.ndarray) -> OcrResult:
        results = self._reader.readtext(image)

        if not results:
            return OcrResult(text="", confidence=0.0)

        texts = []
        confidences = []
        for _, text, conf in results:
            texts.append(text)
            confidences.append(conf)

        combined_text = " ".join(texts).strip()
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return OcrResult(text=combined_text, confidence=mean_confidence)