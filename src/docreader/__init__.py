"""
DocReader - библиотека для распознавания текста с документов

Использование:
    from docreader import DocReader

    reader = DocReader(models_dir="./models")
    result = reader.process("passport.jpg")
    print(result.doc_type)
    print(result.fields)
"""

from docreader.pipeline import DocReader
from docreader.schemas import DocumentResult, ZoneResult

__all__ = ["DocReader", "DocumentResult", "ZoneResult"]
__version__ = "0.1.0"
