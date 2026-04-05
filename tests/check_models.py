from docreader.factory import create_detector

det = create_detector()
print(det.supported_doc_types)

# Чтобы увидеть классы конкретной модели:
model_names = ["passport", "snils", "attestat", "diplom"]
for name in model_names:
    det._get_model(name)
    print(det._loaded_models[name].names)
