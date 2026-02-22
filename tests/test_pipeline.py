# test_full_pipeline.py
import cv2
import numpy as np
from docreader import DocReader

# 1. Создай тестовое изображение (или возьми реальное)
# test_image = np.random.randint(0, 255, (1024, 768, 3), dtype=np.uint8)
test_image = cv2.imread("IMG_20211108_155730.jpg")

# 2. Инициализируй reader
reader = DocReader(models_dir=None)

# 3. Обработай изображение
result = reader.process(test_image)

# 4. Выведи результаты
print(f"📄 Тип документа: {result.doc_type}")
print(f"🔍 Поля: {result.fields}")
print(f"📝 Текст: {result.text}")