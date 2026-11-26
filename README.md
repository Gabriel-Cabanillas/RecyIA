\# RecyIA: Clasificador de residuos con visión artificial



Aplicación web que identifica en tiempo real, con la cámara, el tipo de residuo que se le muestra y recomienda cómo separarlo. Funciona de forma local y sin internet.



Proyecto final de la materia \*\*Fundamentos de IA\*\*, presentado en \*\*ExpoIA 2026\*\*.



\## Objetivo



Desarrollar un sistema de visión artificial capaz de clasificar residuos en seis categorías (plástico, papel, cartón, metal, vidrio y orgánico) y orientar al usuario sobre cómo separarlos correctamente.



\## Propósito y problemática



Separar mal los residuos dificulta su reciclaje y aumenta la contaminación. Muchas personas no saben en qué categoría cae cada objeto. RecyIA busca ayudar a:



\- Identificar rápidamente el tipo de residuo apuntando la cámara al objeto.

\- Recibir una recomendación de separación inmediata.

\- Fomentar la cultura del reciclaje con una herramienta simple y accesible.

\- Funcionar sin conexión, para usarse en lugares sin internet.



\## Cómo funciona



1\. La cámara se enciende y la web analiza lo que aparece en el recuadro central de enfoque.

2\. El modelo calcula la probabilidad de cada una de las 6 categorías.

3\. Si la confianza máxima es menor a 0.6, la web muestra \*\*"No identificado"\*\*.

4\. Si es suficiente, muestra la categoría, las probabilidades, la recomendación de separación y estadísticas por categoría.



\## Cómo se hizo



\*\*Dataset\*\*

\- \[TrashNet](https://github.com/garythung/trashnet): cartón, vidrio, metal, papel y plástico.

\- Clase orgánico (cáscaras y restos de comida) reunida por el equipo.

\- División fija train/val/test de 80/10/10 con semilla 42 (`modelo/separar\_datos.py`). La prueba tiene 308 imágenes.



\*\*Entrenamiento (local, en Python)\*\*

\- MediaPipe Model Maker con MobileNetV2, 15 épocas (`modelo/entrenar.py`).

\- Exporta un modelo `.tflite` de 6 clases.



\*\*Aplicación web\*\*

\- React + Vite.

\- `@mediapipe/tasks-vision` (ImageClassifier) ejecutándose en el navegador con `classifyForVideo` sobre la cámara.

\- El modelo y los archivos WASM se sirven localmente (sin CDN) para funcionar sin internet.



\## Resultados



Evaluación sobre la prueba fija de 308 imágenes (`modelo/evaluar.py`):



| Métrica | Valor |

|---|---|

| Exactitud | 0.854 |

| Precisión (macro) | 0.861 |

| Recall (macro) | 0.849 |

| F1 (macro) | 0.851 |



F1 por clase: orgánico 0.937, metal 0.894, cartón 0.880, papel 0.857, vidrio 0.800, plástico 0.739. La principal confusión del modelo es entre plástico y vidrio.



!\[Matriz de confusión](docs/matriz\_confusion.png)

!\[F1 por clase](docs/f1\_por\_clase.png)



\## Estructura del proyecto



```

clasificador-residuos/

├── dataset/      # imágenes por categoría (no incluidas en el repositorio)

├── modelo/       # scripts de preparación, entrenamiento y evaluación

├── web/          # aplicación React

└── docs/         # gráficas de evaluación

```



\## Cómo ejecutar la web



Requisitos: Node.js.



```powershell

cd web

npm install

npm run dev

```



Abre `http://localhost:3000` y permite el acceso a la cámara. El modelo ya está incluido en `web/public/modelo/model.tflite`.



\## Cómo reentrenar el modelo (opcional)



Requisitos: Python 3.11 y el dataset en `dataset/` (una carpeta por categoría: `carton`, `vidrio`, `metal`, `papel`, `plastico`, `organico`).



```powershell

cd modelo

python -m venv venv

.\\venv\\Scripts\\Activate.ps1

pip install mediapipe-model-maker

python separar\_datos.py

python entrenar.py

python evaluar.py

```



Copia el nuevo `model.tflite` generado a `web/public/modelo/`.



\## Limitaciones



\- El modelo se entrenó con imágenes de TrashNet, con fondos y condiciones de luz distintos a los de una cámara real.

\- Plástico y vidrio son las categorías más difíciles de distinguir.



\## Equipo



\- Fátima Lizárraga Velarde

\- Gabriel Cabanillas Cabanillas

\- Allison Hatziry Tirado Barraza

\- Ana Rubí Aguirre Sanchez



Materia: Fundamentos de IA, Dra. Alma Yadira Quiñonez Carrillo.

