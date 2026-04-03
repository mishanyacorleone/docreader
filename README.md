# docreader-ocr

[![PyPI version](https://img.shields.io/pypi/v/docreader-ocr.svg)](https://pypi.org/project/docreader-ocr/)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/mishanyacorleone/docreader.svg)](https://github.com/mishanyacorleone/docreader/stargazers)

Python-библиотека для автоматического распознавания российских документов. Сфотографировал — получил структурированные данные.

```python
from docreader import DocReader

result = DocReader().process("passport.jpg")
print(result.documents[0].fields)
# {"surname": "Иванов", "firstname": "Иван", "passport_num": "1234 567890", ...}
```

---

## Поддерживаемые документы

| Документ | Распознаваемые поля |
|----------|---------------------|
| Паспорт РФ | surname, firstname, middlename, dateOfBirth, birthCity, sex, passport_num, issued_by, issued_date, issued_code |
| СНИЛС | fio, snils, date_of_birth, sex, location, reg_date |
| Аттестат | fio, lvl, number, issue_date, grad_year, school_name, gerb |
| Диплом | fio, lvl, series_numbers, reg_num, issue_date, spec_name, university_name, gerb, stamp |

---

## Установка

```bash
pip install docreader-ocr
```

Модели скачиваются автоматически при первом запуске и кэшируются в `~/.cache/docreader/models/`.

### Требования

- Python 3.12+
- PyTorch (CPU или CUDA)
- ~200 МБ дискового пространства для моделей

---

## Быстрый старт

### Базовое использование

```python
from docreader import DocReader

reader = DocReader()
result = reader.process("photo.jpg")

for doc in result.documents:
    print(f"Тип: {doc.doc_type}")
    print(f"Поля: {doc.fields}")
```

### Пакетная обработка

```python
results = reader.process_batch(["doc1.jpg", "doc2.jpg", "doc3.jpg"])

for page in results:
    for doc in page.documents:
        print(doc.fields)
```

### Использование numpy array

```python
import cv2
from docreader import DocReader

image = cv2.imread("passport.jpg")
result = DocReader().process(image)
```

### Получение кропов зон

```python
result = DocReader().process("passport.jpg", return_crops=True)

for doc in result.documents:
    for zone in doc.zones:
        print(f"{zone.name}: {zone.text}")
        # zone.crop_image — numpy array с вырезанной зоной
```

### Использование GPU

```python
from docreader import DocReader
from docreader.config import PipelineConfig

config = PipelineConfig(device="cuda")
reader = DocReader(config=config)
```

---

## Архитектура

Библиотека реализует трёхэтапный конвейер:

```
Фото → [Классификатор] → [Детектор зон] → [OCR] → Словарь полей
         YOLO OBB          YOLO OBB          EasyOCR
         97.5% acc         mAP@50=90%        CER=0.15
```

**Классификатор** — определяет тип документа и вырезает его из произвольной фотографии. Работает при любом освещении и ракурсе.

**Детектор зон** — специализированная YOLO OBB модель для каждого типа документа. Находит поля с точностью mAP@50 = 90%.

**OCR-движок** — EasyOCR с кастомным дообучением под русскоязычные документы. Структурированные поля (числа, даты, коды) — Exact Match 85–92%.

### Resolver для аттестата/диплома

Аттестат и диплом визуально идентичны, поэтому их различение вынесено в отдельный компонент — `LvlSubtypeResolver`. Он детектирует поле `lvl`, читает его через OCR и с помощью fuzzy matching определяет подтип документа.

---

## Метрики качества

### Классификатор

| Класс | Precision | Recall | F1 |
|-------|-----------|--------|----|
| passport | 0.968 | 1.000 | 0.984 |
| snils | 1.000 | 0.903 | 0.949 |
| attestat | 1.000 | 1.000 | 1.000 |
| diplom | 1.000 | 1.000 | 1.000 |
| **Общая точность** | | | **97.5%** |

### Детектор зон

| Метрика | Значение |
|---------|----------|
| mAP@50 | 90.0% |
| Лучшая зона (gerb) | F1 = 99.1% |
| Слабейшая зона (location) | F1 = 82.5% |

### OCR

| Метрика | Значение |
|---------|----------|
| CER средний | 0.146 |
| WER средний | 0.276 |
| Exact Match средний | 58.8% |
| Exact Match (series_numbers) | 92.3% |
| Exact Match (fio) | 88.4% |

---

## Кастомизация

### Своя конфигурация

```python
from docreader import DocReader
from docreader.config import PipelineConfig

config = PipelineConfig(
    device="cuda",
    classifier_confidence=0.5,
    detector_confidence=0.3,
    enable_deskew=True,
    return_crops=False,
    skip_ocr_zones={"stamp", "gerb"},
)
reader = DocReader(config=config)
```

### Использование отдельных компонентов

```python
from docreader.factory import create_classifier, create_detector, create_ocr

# Только классификатор
clf = create_classifier()
docs = clf.classify("photo.jpg")

# Только детектор
det = create_detector()
zones = det.detect(image, doc_type="passport")

# Только OCR
ocr = create_ocr()
result = ocr.recognize(crop_image)
print(result.text, result.confidence)
```

### Подключение своего OCR-движка

```python
from docreader.ocr.base import BaseOcrEngine, OcrResult
import numpy as np

class MyOcrEngine(BaseOcrEngine):
    def recognize(self, image: np.ndarray) -> OcrResult:
        # ваша реализация
        return OcrResult(text="...", confidence=0.95)

reader = DocReader(ocr_engine=MyOcrEngine())
```

### Добавление нового типа документа

```python
from docreader.config import PipelineConfig

config = PipelineConfig(
    detector_weights={
        "passport":  "passport.pt",
        "snils":     "snils.pt",
        "attestat":  "attestat.pt",
        "diplom":    "diplom.pt",
        "inn":       "/path/to/your/inn.pt",  # ваша модель
    }
)
reader = DocReader(config=config)
```

---

## Структура результата

```python
PageResult
└── documents: list[DocumentResult]
    ├── doc_type: str          # "passport", "snils", "attestat", "diplom"
    ├── doc_confidence: float  # уверенность классификатора
    ├── doc_bbox: list[float]  # координаты документа в исходном изображении
    ├── doc_crop: np.ndarray   # вырезанный документ (если return_crops=True)
    ├── fields: dict           # {zone_name: text} — удобный доступ к полям
    ├── resolve_meta: dict     # диагностика resolver'а (для attestat/diplom)
    └── zones: list[ZoneResult]
        ├── name: str
        ├── text: str
        ├── confidence: float
        ├── bbox: list[float]
        └── crop_image: np.ndarray  # если return_crops=True
```

---

## Управление моделями

```python
from docreader.hub import get_model_status, ensure_all_models

# Статус всех моделей
status = get_model_status()
for name, info in status.items():
    print(f"{name}: {'✓' if info['downloaded'] else '✗'} ({info['size_mb']} MB)")

# Скачать все модели заранее
ensure_all_models()

# Кастомная директория кэша
import os
os.environ["DOCREADER_CACHE"] = "/path/to/custom/cache"
```

---

## Почему библиотека, а не сервис

**Данные остаются внутри.** Персональные данные не покидают инфраструктуру организации. Полное соответствие 152-ФЗ. Никаких внешних серверов.

**Интеграция без переписывания.** Встраивается в любую существующую систему — 1С, CRM, ERP, мобильное приложение — двумя строками кода.

**Полный контроль.** Новые типы документов подключаются через дообучение без участия вендора. IT-отдел контролирует всё: модели, данные, обновления.

**Нет операционных затрат.** В отличие от облачных API — никакой абонентской платы и лимитов на количество запросов.

---

## Лицензия

MIT License — см. [LICENSE](LICENSE).

---

## Ссылки

- [PyPI](https://pypi.org/project/docreader-ocr/)
- [GitHub](https://github.com/mishanyacorleone/docreader)