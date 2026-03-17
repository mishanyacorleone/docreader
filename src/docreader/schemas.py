from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ZoneResult:
    """
    Результат распознавания одного поля документа
    """
    name: str
    text: str
    confidence: float
    bbox: list[float] = field(default_factory=list)
    crop_image: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "text": self.text,
            "confidence": round(self.confidence, 4)
        }
    

@dataclass
class DocumentResult:
    """
    Результат обработки одного документа
    """
    doc_type: str
    doc_confidence: float
    zones: list[ZoneResult] = field(default_factory=list)
    doc_bbox: Optional[list[float]] = None # координаты документа в исходном изображении
    doc_crop: Optional[np.ndarray] = None # кроп документа
    resolve_meta: dict = field(default_factory=dict) # диагностика resolver'a

    @property
    def fields(self) -> dict[str, str]:
        """
        Словарь: {имя_зоны: распознанный_текст}
        """ 
        return {zone.name: zone.text for zone in self.zones}
    
    def to_dict(self) -> dict:
        result = {
            "document": {
                "doc_type": self.doc_type,
                "doc_confidence": round(self.doc_confidence, 4),
                "zones": [zone.to_dict() for zone in self.zones],
                "fields": self.fields
            }
        }
        if self.resolve_meta:
            result["document"]["resolve_meta"] = self.resolve_meta
        return result
    
    def __repr__(self) -> str:
        return (
            f"DocumentResult(doc_type='{self.doc_type}', "
            f"confidence={self.doc_confidence:.3f}, "
            f"zones={len(self.zones)})"
        )
    

@dataclass
class PageResult:
    """
    Результат обработки одной страницы/фотографии.
    Может содержать несколько документов.
    """
    documents: list[DocumentResult] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.documents)
    
    def to_dict(self) -> dict:
        return {
            "documents": [doc.to_dict() for doc in self.documents],
            "total": self.count
        }
    
    def __repr__(self) -> str:
        types = [d.doc_type for d in self.documents]
        return f"PageResult(documents={self.count}, types={types})"
