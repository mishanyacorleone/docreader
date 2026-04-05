# docreader-ocr

Python-библиотека для автоматического распознавания российских документов.

```python
from docreader import DocReader

result = DocReader().process("passport.jpg")
print(result.documents[0].fields)
# {"surname": "Иванов", "firstname": "Иван", "passport_num": "1234 567890", ...}
```

## Установка

```bash
pip install docreader-ocr
```

Модели скачиваются автоматически при первом запуске.

## Поддерживаемые документы

- Паспорт РФ
- СНИЛС
- Аттестат об образовании
- Диплом о высшем образовании

## Как работает

Трёхэтапный конвейер: **классификатор** (YOLO OBB, accuracy 97.5%) определяет тип документа → **детектор зон** (YOLO OBB, mAP@50 = 90%) находит поля → **OCR** (EasyOCR, word accuracy 87.3%) распознаёт текст.

Данные обрабатываются локально — никаких внешних серверов, полное соответствие 152-ФЗ.

## Документация

Полный README, примеры и API — на [GitHub](https://github.com/mishanyacorleone/docreader).