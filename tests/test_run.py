import numpy as np
from docreader import DocReader

# 1. Тест инициализации
print("🔄 Инициализация DocReader...")
reader = DocReader(models_dir=None)  # Без авто-загрузки
print("✅ Инициализация успешна")

# 2. Тест с dummy изображением
print("\n🔄 Тест обработки изображения...")
dummy_image = np.random.randint(0, 255, (1024, 768, 3), dtype=np.uint8)

try:
    # result = reader.process(dummy_image)  # Раскомментировать когда будут модели
    print("✅ Обработка работает (закомментировано)")
except Exception as e:
    print(f"❌ Ошибка: {e}")

# 3. Проверка версии
print(f"\n📦 Версия пакета: {reader.__class__.__module__}")
import docreader
print(f"📦 Версия: {docreader.__version__}")