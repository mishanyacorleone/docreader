# debug_ocr.py — положи в tests/
from pathlib import Path
from docreader.factory import create_detector, create_ocr
from docreader.preprocessing.geometry import crop_obb_region
from docreader.utils import load_image
import cv2

detector = create_detector()
ocr = create_ocr()

image_path = "data/passport/0_6e18a_ce36f57b_XXL_jpg.rf.03f5c7a705e8c206e4dccf17fabfdfbf.jpg"  # подставь реальный файл
image = load_image(image_path)

detections = detector.detect(image, "passport")
for det in detections:
    crop = crop_obb_region(image, det.obb_points)
    result = ocr.recognize(crop)
    print(f"zone={det.zone_name:20} | text='{result.text}' | ocr_conf={result.confidence:.2f}")

    # Сохраним кроп чтобы посмотреть глазами
    cv2.imwrite(f"debug_{det.zone_name}.jpg", crop)