"""
Конфигурация пайплайна
"""

from dataclasses import dataclass, field

DEFAULT_SKIP_OCR_ZONES = frozenset({"stamp", "gerb"})


@dataclass
class PipelineConfig:
    """
    Настройки пайплайна распознавания
    """
    device: str = "auto"

    # Типы документов и пути к YOLO-моделям (относительно models_dir)
    detector_weights: dict[str, str] = field(default_factory=lambda: {
        "attestat": "attestat.pt",
        "diplom": "diplom.pt",
        "passport": "passport.pt",
        "snils": "snils.pt",
    })

    # Путь к весам классификатора (относительно models_dir)
    classification_weights: str = "best_doc_classifier.pth"

    class_labels: list[str] = field(default_factory=lambda: [
        "attestat", "diplom", "passport", "snils", 'other'
    ])

    skip_ocr_zones: frozenset[str] = DEFAULT_SKIP_OCR_ZONES 

    enable_descew: bool = True  # Выравнивание по линиям Хафа

    return_crops: bool = True  # Сохранять кропы зон в результат

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
