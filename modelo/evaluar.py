from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

clases = ["carton", "metal", "organico", "papel", "plastico", "vidrio"]
test_dir = Path("../dataset_split/test")
docs = Path("../docs")
docs.mkdir(exist_ok=True)

base = python.BaseOptions(model_asset_path="exported_model/model.tflite")
opciones = vision.ImageClassifierOptions(base_options=base, max_results=1)
clf = vision.ImageClassifier.create_from_options(opciones)

cm = np.zeros((6, 6), dtype=int)
for i, c in enumerate(clases):
    for f in sorted((test_dir / c).iterdir()):
        img = mp.Image.create_from_file(str(f))
        res = clf.classify(img)
        pred = res.classifications[0].categories[0].category_name
        cm[i, clases.index(pred)] += 1

total = cm.sum()
exactitud = np.trace(cm) / total
precision = np.diag(cm) / np.maximum(cm.sum(axis=0), 1)
recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)

print(f"\nImágenes de prueba: {total}")
print(f"Exactitud: {exactitud:.3f}")
print(f"{'clase':<10}{'precisión':>10}{'recall':>10}{'F1':>8}")
for i, c in enumerate(clases):
    print(f"{c:<10}{precision[i]:>10.3f}{recall[i]:>10.3f}{f1[i]:>8.3f}")
print(f"{'macro':<10}{precision.mean():>10.3f}{recall.mean():>10.3f}{f1.mean():>8.3f}")
print("\nMatriz de confusión (filas = real, columnas = predicho):")
print(cm)

fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(cm, cmap="Greens")
ax.set_xticks(range(6)); ax.set_xticklabels(clases, rotation=45, ha="right")
ax.set_yticks(range(6)); ax.set_yticklabels(clases)
ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
for i in range(6):
    for j in range(6):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
fig.tight_layout()
fig.savefig(docs / "matriz_confusion.png", dpi=200)

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(clases, f1 * 100, color="#2f7d54")
ax.set_ylabel("F1 (%)"); ax.set_ylim(0, 100)
for i, v in enumerate(f1 * 100):
    ax.text(i, v + 1, f"{v:.0f}", ha="center")
fig.tight_layout()
fig.savefig(docs / "f1_por_clase.png", dpi=200)
print("\nImágenes guardadas en docs/")