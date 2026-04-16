"""
Геометрические преобразования: выравнивание и кроп по OBB.
"""

import math
import cv2
import numpy as np


def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Выравнивает документ по горизонтали через преобразование Хафа.

    Args:
        image: BGR изображение.

    Returns:
        Повёрнутое изображение.

    Raises:
        ValueError: если линии не найдены.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5
    )

    if lines is None or len(lines) == 0:
        return image

    angles = [
        math.degrees(math.atan2(y2 - y1, x2 - x1))
        for [[x1, y1, x2, y2]] in lines
    ]
    median_angle = float(np.median(angles))

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (w, h))


def crop_obb_region(
    image: np.ndarray,
    obb_points: np.ndarray,
    zone_name: str = "",
) -> np.ndarray | None:
    
    # Зоны которые нужно повернуть после кропа
    ROTATE_CW_ZONES = {"passport_num"}

    pts = np.array(obb_points, dtype=np.float32).reshape(4, 2)
    rect = _order_points(pts)
    tl, tr, br, bl = rect

    width = max(
        int(np.linalg.norm(br - bl)),
        int(np.linalg.norm(tr - tl)),
    )
    height = max(
        int(np.linalg.norm(tr - br)),
        int(np.linalg.norm(tl - bl)),
    )

    if width <= 0 or height <= 0:
        return None

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))

    # Поворачиваем только конкретные зоны
    h, w = warped.shape[:2]
    if h > w and zone_name in ROTATE_CW_ZONES:
        warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return warped


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Упорядочивает 4 точки: TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect