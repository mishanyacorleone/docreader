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

    classifier_weights: str = "doc_classifier.pt"
    classifier_confidence: float = 0.3

    # Типы документов и пути к YOLO-моделям (относительно models_dir)
    detector_weights: dict[str, str] = field(default_factory=lambda: {
        "attestat": "attestat.pt",
        "diplom": "diplom.pt",
        "passport": "passport.pt",
        "snils": "snils.pt",
    })
    detector_confidence: float = 0.25

    # EasyOCR
    ocr_lang: list[str] = field(default_factory=lambda: ["ru"])
    ocr_model_archive: str = "easyocr_custom.tar.gz"
    ocr_model_subdir: str = "model"
    ocr_network_subdir: str = "user_network"
    ocr_recog_network: str = "custom_example"
    ocr_download_enabled: bool = False
    skip_ocr_zones: frozenset[str] = DEFAULT_SKIP_OCR_ZONES 

    enable_deskew: bool = True  # Выравнивание по линиям Хафа
    return_crops: bool = True  # Сохранять кропы зон в результат

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
        