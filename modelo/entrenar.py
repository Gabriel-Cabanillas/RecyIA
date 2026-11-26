from mediapipe_model_maker import image_classifier

train_data = image_classifier.Dataset.from_folder("../dataset_split/train")
validation_data = image_classifier.Dataset.from_folder("../dataset_split/val")
test_data = image_classifier.Dataset.from_folder("../dataset_split/test")
print("Clases:", train_data.label_names)

options = image_classifier.ImageClassifierOptions(
    supported_model=image_classifier.SupportedModels.MOBILENET_V2,
    hparams=image_classifier.HParams(
        export_dir="exported_model", epochs=15, batch_size=32
    ),
)

model = image_classifier.ImageClassifier.create(train_data, validation_data, options)

loss, acc = model.evaluate(test_data)
print(f"Precisión en prueba: {acc:.3f}")

model.export_model()
print("Modelo exportado en modelo/exported_model/model.tflite")