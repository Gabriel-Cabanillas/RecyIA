import random
import shutil
from pathlib import Path

random.seed(42)
origen = Path("../dataset")
destino = Path("../dataset_split")

if destino.exists():
    shutil.rmtree(destino)

for carpeta in sorted(p for p in origen.iterdir() if p.is_dir()):
    imgs = sorted(f for f in carpeta.iterdir()
                  if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    random.shuffle(imgs)
    n = len(imgs)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    partes = {
        "train": imgs[:n_train],
        "val": imgs[n_train:n_train + n_val],
        "test": imgs[n_train + n_val:],
    }
    for nombre, lista in partes.items():
        d = destino / nombre / carpeta.name
        d.mkdir(parents=True)
        for f in lista:
            shutil.copy2(f, d / f.name)
    print(carpeta.name, {k: len(v) for k, v in partes.items()})