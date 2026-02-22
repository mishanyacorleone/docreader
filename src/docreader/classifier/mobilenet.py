"""Классификатор документов на основе MobileNetV2."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from docreader.classifier.base import BaseClassifier


class MobileNetClassifier(BaseClassifier):
    """
    Классификатор типа документа на основе MobileNetV2.

    Args:
        weights_path: путь к файлу весов (.pth).
        class_labels: список меток классов.
        device: устройство ("cpu" / "cuda").
    """

    # Стандартные трансформации ImageNet
    _transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    def __init__(
        self,
        weights_path: str,
        class_labels: list[str],
        device: str = "cpu",
    ):
        self._class_labels = class_labels
        self._device = device

        num_classes = len(class_labels)
        self._model = self._build_model(num_classes)
        self._load_weights(weights_path)

    def _build_model(self, num_classes: int) -> nn.Module:
        net = models.mobilenet_v2(pretrained=False)
        net.classifier[1] = nn.Linear(net.last_channel, num_classes)
        return net

    def _load_weights(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self._device, weights_only=False)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("net."):
                new_key = key[4:]  # Убираем первые 4 символа "net."
            else:
                new_key = key
            new_state_dict[new_key] = value

        self._model.load_state_dict(new_state_dict)
        self._model.to(self._device)
        self._model.eval()

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> tuple[str, float]:
        """
        Args:
            image: BGR изображение.

        Returns:
            (метка_класса, уверенность)
        """
        tensor = self._transform(image).unsqueeze(0).to(self._device)
        logits = self._model(tensor)
        probs = F.softmax(logits, dim=1)

        confidence, idx = probs.max(dim=1)
        label = self._class_labels[idx.item()]
        return label, confidence.item()