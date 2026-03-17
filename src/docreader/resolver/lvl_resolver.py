"""
Resolver подтипа документа через детекцию поля lvl + OCR + fuzzy matching.
"""

import logging
from typing import Optional

import numpy as np
from rapidfuzz import process, fuzz
from ultralytics import YOLO

from docreader.ocr.base import BaseOcrEngine
from docreader.preprocessing.geometry import crop_obb_region
from docreader.resolver.base import BaseSubtypeResolver, ResolveResult

logger = logging.getLogger(__name__)


class LvlSubtypeResolver(BaseSubtypeResolver):
    """
    Определяет подтип документа (attestat/diplom) через:
      1. YOLO OBB — детектирует поле lvl на crop'е документа
      2. OCR — распознаёт текст поля
      3. Fuzzy matching — сопоставляет текст с ключевыми словами подтипов

    Примеры:
        resolver = LvlSubtypeResolver(
            weights_path="/path/to/lvl_detector.pt",
            ocr_engine=ocr,
            subtype_keywords={
                "attestat": ["аттестат", "attestat"],
                "diplom": ["диплом", "diplom"],
            },
        )
        result = resolver.resolve(doc_crop)
        if result.resolved:
            print(result.subtype)  # "attestat" или "diplom"

    Args:
        weights_path: путь к YOLO-модели для детекции поля lvl.
        ocr_engine: движок OCR (BaseOcrEngine).
        subtype_keywords: словарь {подтип: [ключевые слова]}.
        fuzzy_threshold: минимальный score для признания совпадения (0–100).
        confidence_threshold: минимальная уверенность детектора.
        fallback: подтип по умолчанию если resolve не удался (None = unresolved).
        device: устройство ("cpu", "cuda").
    """

    def __init__(
        self,
        weights_path: str,
        ocr_engine: BaseOcrEngine,
        subtype_keywords: dict[str, list[str]],
        fuzzy_threshold: float = 60.0,
        confidence_threshold: float = 0.25,
        fallback: Optional[str] = None,
        device: str = "cpu",
    ):
        self._model = YOLO(weights_path)
        self._ocr = ocr_engine
        self._fuzzy_threshold = fuzzy_threshold
        self._confidence_threshold = confidence_threshold
        self._fallback = fallback
        self._device = device

        # Плоский список ключевых слов и маппинг слово → подтип
        self._keywords: list[str] = []
        self._keyword_to_subtype: dict[str, str] = {}
        for subtype, words in subtype_keywords.items():
            for word in words:
                normalized = word.lower()
                self._keywords.append(normalized)
                self._keyword_to_subtype[normalized] = subtype

        logger.info(
            f"LvlSubtypeResolver initialized: "
            f"subtypes={list(subtype_keywords.keys())}, "
            f"threshold={fuzzy_threshold}, fallback={fallback}"
        )

    def resolve(self, image: np.ndarray) -> ResolveResult:
        """
        Определяет подтип документа по crop'у.

        Args:
            image: BGR изображение документа.

        Returns:
            ResolveResult с подтипом и диагностической информацией.
        """
        lvl_crop = self._detect_lvl_field(image)

        if lvl_crop is None:
            logger.warning("lvl field not detected, using fallback")
            return ResolveResult(
                subtype=self._fallback,
                ocr_text="",
                confidence=0.0,
                fuzzy_score=0.0,
            )

        ocr_result = self._ocr.recognize(lvl_crop)
        logger.debug(
            f"lvl OCR: text='{ocr_result.text}', conf={ocr_result.confidence:.3f}"
        )

        if not ocr_result.text.strip():
            logger.warning("lvl OCR returned empty text, using fallback")
            return ResolveResult(
                subtype=self._fallback,
                ocr_text=ocr_result.text,
                confidence=ocr_result.confidence,
                fuzzy_score=0.0,
            )

        subtype, fuzzy_score = self._match_subtype(ocr_result.text)

        if subtype is None:
            logger.warning(
                f"Fuzzy match below threshold: "
                f"text='{ocr_result.text}', score={fuzzy_score:.1f}, "
                f"threshold={self._fuzzy_threshold}, fallback={self._fallback}"
            )

        return ResolveResult(
            subtype=subtype,
            ocr_text=ocr_result.text,
            confidence=ocr_result.confidence,
            fuzzy_score=fuzzy_score,
        )

    def _detect_lvl_field(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Детектирует поле lvl и возвращает его crop.
        """
        
        results = self._model(image, device=self._device, verbose=False)

        if results[0].obb is None:
            return None

        best_conf = -1.0
        best_crop = None

        for det in results[0].obb:
            confidence = float(det.conf.cpu())
            if confidence < self._confidence_threshold:
                continue

            zone_name = self._model.names[int(det.cls.cpu())]
            if zone_name != "lvl":
                continue

            if confidence <= best_conf:
                continue

            obb_points = det.xyxyxyxy.cpu().numpy().flatten()
            crop = crop_obb_region(image, obb_points)

            if crop is not None and crop.size > 0:
                best_conf = confidence
                best_crop = crop

        return best_crop

    def _match_subtype(self, text: str) -> tuple[Optional[str], float]:
        """
        Сопоставляет OCR-текст с ключевыми словами через fuzzy matching.

        Returns:
            Кортеж (подтип или None, fuzzy score).
        """
        normalized = text.lower().strip()

        match = process.extractOne(
            normalized,
            self._keywords,
            scorer=fuzz.WRatio,
        )

        if match is None:
            return None, 0.0

        best_keyword, score, _ = match

        if score < self._fuzzy_threshold:
            return None, float(score)

        subtype = self._keyword_to_subtype[best_keyword]
        logger.debug(
            f"Fuzzy matched: '{normalized}' -> '{best_keyword}' "
            f"(subtype={subtype}, score={score:.1f})"
        )
        return subtype, float(score)
    