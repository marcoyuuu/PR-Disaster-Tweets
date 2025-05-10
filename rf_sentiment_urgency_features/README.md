# Clasificador de Sentimiento y Desastres en Tweets


Este módulo (`sentiment_disaster_classifier.py`) implementa un pipeline profesional para la clasificación de tweets de desastres, integrando análisis de sentimiento (VADER), urgencia y otras características textuales y meta. Soporta modelos Random Forest, SVM y Regresión Logística con Bag of Words (BoW), así como validación cruzada, explicabilidad (SHAP) y manejo robusto de errores.

## Descripción del Pipeline


El pipeline sigue estos pasos principales:

1. **Parseo de Argumentos**
   - Utiliza `argparse` para leer argumentos de línea de comandos: ruta de datos, directorio de resultados, tipo de modelo, características TF-IDF, validación cruzada, explicabilidad y validación del mapeo de etiquetas.

2. **Carga de Datos**
   - Carga el archivo CSV de entrada usando pandas.
   - Verifica que los datos no estén vacíos y que existan las columnas requeridas.

3. **Mapeo y Validación de Etiquetas**
   - Mapea la columna `class_label` a un conjunto unificado de etiquetas (`informative`, `neutral`, `non_informative`).
   - Valida el mapeo: si la proporción de etiquetas no mapeadas (NaN) supera el umbral (`--nan_threshold`), el script lanza un error y se detiene.


4. **Ingeniería de Características y Enriquecimiento**
   - **Limpieza de Texto:** El texto se normaliza y limpia automáticamente.
   - **Análisis de Sentimiento (VADER):** Se calcula el puntaje de sentimiento VADER para cada tweet y se agrega como la columna `sentiment` (valor continuo entre 0 y 1). Esta característica se utiliza en todos los modelos.
   - **Detección de Urgencia:** Se marca si el tweet contiene palabras clave de urgencia.
   - **Características Textuales y Meta:** Se calculan longitud, signos de exclamación/interrogación, hashtags, menciones, likes y retweets.
   - **Vectorización:**
     - **TF-IDF:** Para Random Forest y SVM, se extraen n-gramas del texto limpio.
     - **Bag of Words (BoW):** Para Regresión Logística, se utiliza un vectorizador BoW sobre el texto limpio.
   - **Combinación de Características:** Se combinan las características textuales (TF-IDF o BoW) con las meta (incluyendo `sentiment`).
   - El vectorizador entrenado (TF-IDF o BoW) se guarda para uso posterior.


5. **Preparación de Datos**
   - Asegura que todas las características y etiquetas estén alineadas y sin NaNs.
   - Convierte las etiquetas de texto a códigos numéricos para la clasificación.

6. **Entrenamiento y Evaluación del Modelo**
   - **Modelos Disponibles:**
     - `rf`: Random Forest
     - `svm`: Support Vector Machine
     - `logreg_bow`: Regresión Logística con Bag of Words
   - **Validación Cruzada (si `--cv > 1`):**
     - Realiza validación cruzada k-fold usando el modelo seleccionado.
     - Calcula y guarda la media y desviación estándar de precisión, recall y F1-macro.
     - Guarda todas las métricas de validación cruzada en `metrics.json`.
   - **Partición Train/Test (si `--cv` es 0 o 1):**
     - Divide los datos en conjuntos de entrenamiento y prueba.
     - Entrena el modelo y evalúa en el conjunto de prueba.
     - Calcula F1, precisión, recall y soporte por clase.
     - Guarda el modelo entrenado, predicciones, matriz de confusión y métricas.

7. **Explicabilidad (Opcional, solo Random Forest)**
   - Si se activa `--explain` y se usa Random Forest, calcula valores SHAP para el conjunto de prueba.
   - Guarda los valores SHAP, el valor esperado y gráficos resumen para interpretabilidad del modelo.

8. **Trazabilidad del Experimento**
   - Registra la fecha/hora y el hash de git actual en `metrics.json` para reproducibilidad.

9. **Artefactos de Salida**
   - Todos los resultados (métricas, modelo, vectorizador, predicciones, gráficos, archivos SHAP) se guardan en el directorio de resultados especificado.

---



## Características Destacadas

- **Análisis de sentimiento VADER como feature para todos los modelos**
- **Soporte para Random Forest, SVM y Regresión Logística (BoW)**
- **Validación cruzada y partición train/test**
- **Explicabilidad SHAP (solo Random Forest)**
- **Métricas completas y trazabilidad (timestamp, git hash)**
- **Manejo robusto de errores y validación de etiquetas**
- **Pipeline profesional y extensible**


## Requisitos

- Python 3.8+
- Instalar dependencias:
  ```sh
  pip install -r requirements.txt
  ```


## Uso

Ejecuta el script desde la raíz del proyecto:

```sh
python code/sentiment_disaster_classifier.py --data <ruta_al_csv>
```

### Argumento Requerido

- `--data`  
  Ruta al dataset CSV (por ejemplo, humaid_dataset_preprocesado.csv).

### Argumentos Opcionales

- `--results_dir`  
  Directorio donde se guardarán los resultados (por defecto: `results`).

- `--max_feats`  
  Número máximo de características para el vectorizador TF-IDF (por defecto: `1000`).

- `--model_type`  
  Tipo de modelo: `rf` (Random Forest, por defecto), `svm` (Support Vector Machine), o `logreg_bow` (Regresión Logística con Bag of Words).

- `--explain`  
  Habilita explicabilidad SHAP (solo para Random Forest).

- `--cv`  
  Número de folds para validación cruzada (por ejemplo, `5`). Si es `0` o `1`, usa partición train/test.

- `--nan_threshold`  
  Umbral para la proporción de NaN en las etiquetas tras el mapeo; si se supera, se lanza error (por defecto: `0.5`).

### Comandos de Ejemplo


**Uso básico:**
```sh
python code/sentiment_disaster_classifier.py --data code/results/humaid_dataset_preprocesado.csv
```

**Especificar directorio de resultados y aumentar características TF-IDF:**
```sh
python code/sentiment_disaster_classifier.py --data code/results/humaid_dataset_preprocesado.csv --results_dir code/results/exp1 --max_feats 2000
```

**Usar modelo SVM:**
```sh
python code/sentiment_disaster_classifier.py --data code/results/humaid_dataset_preprocesado.csv --model_type svm
```

**Habilitar explicabilidad SHAP (solo Random Forest):**
```sh
python code/sentiment_disaster_classifier.py --data code/results/humaid_dataset_preprocesado.csv --explain
```

**Validación cruzada 5-fold:**
```sh
python code/sentiment_disaster_classifier.py --data code/results/humaid_dataset_preprocesado.csv --cv 5
```

**Umbral de NaN más estricto:**
```sh
python code/sentiment_disaster_classifier.py --data code/results/humaid_dataset_preprocesado.csv --nan_threshold 0.2
```

**Todos los argumentos para mejor resultado:**
```sh
python sentiment_disaster_classifier.py `
  --data results/humaid_dataset_preprocesado.csv `
  --results_dir results/exp_humaid_preprocesado `
  --max_feats 2000 `
  --model_type rf `
  --explain `
  --cv 5 `
  --nan_threshold 0.3
```



## Salidas

- `metrics.json` — Métricas incluyendo F1, precisión, recall, soporte, timestamp y git hash.
- `predictions.csv` — Etiquetas verdaderas y predichas para el set de prueba.
- `final_model.joblib` — Modelo entrenado (si no se usa validación cruzada).
- `tfidf_vectorizer.joblib` o `bow_vectorizer.joblib` — Vectorizador entrenado (según modelo).
- `confusion_matrix.png` — Gráfico de la matriz de confusión.
- `shap_summary*.png`, `shap_values.pkl`, `shap_expected_value.pkl` — Salidas de explicabilidad SHAP (si está habilitado y usando Random Forest).

## Notas

---

## Diagrama del Pipeline

```mermaid
flowchart TD
    A[Inicio/Parseo de Args] --> B[Cargar CSV]
    B --> C[Mapeo y Validación de Etiquetas]
    C --> D[Ingeniería de Características]
    D --> E[TF-IDF + Características Meta]
    E --> F{¿Validación Cruzada?}
    F -- Sí --> G[Validación Cruzada]
    F -- No --> H[Train/Test Split]
    G --> I[Métricas/Guardar]
    H --> J[Entrenar Modelo]
    J --> K[Evaluar/Test]
    K --> L[¿Explicabilidad (SHAP)?]
    L -- Sí --> M[Calcular y Guardar SHAP]
    L -- No --> N[Omitir SHAP]
    M --> O[Guardar Salidas]
    N --> O
    I --> O
    O[Guardar Todos los Artefactos]
    O --> P[Fin]
```

- Ejecuta todos los comandos desde la raíz del proyecto para asegurar la correcta resolución de rutas.
- El análisis de sentimiento VADER se calcula automáticamente y se utiliza como feature (`sentiment`) en todos los modelos.
- Para mejores resultados, asegúrate de que tu dataset tenga una columna `class_label` con categorías reconocidas.
- Para problemas o personalización, revisa el código fuente y los comentarios del script.
