"""
Утилиты для загрузки и базовой обработки изображений
"""

import cv2
import numpy as np


def load_image(source) -> np.ndarray:
    """
    Загружает изображение из разных источников.

    Args:
        source: путь к файлу (str), numpy array (BGR или RGB)
    Returns:
        Изображение в формате BGR (numpy array)
    Raises:
        ValueError: если формат не поддерживается или файл не найден
    """
    if isinstance(source, str):
        image = cv2.imread(source)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {source}")
        return image
    
    if isinstance(source, np.ndarray):
        if source.ndim == 3 and source.shape[2] == 3:
            return source.copy()
        raise ValueError(f"Неподдерживаемая форма массива: {source}")
    raise ValueError(f"Неподдерживаемый тип источника: {type(source)}")


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
