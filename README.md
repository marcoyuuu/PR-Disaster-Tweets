
# Estudio de Reacciones Públicas en Redes Sociales durante Desastres Naturales en Puerto Rico

**Autores:**
- Sebastián H. Jansasoy Molina (Universidad de Puerto Rico, Mayagüez) — sebastian.jansasoy@upr.edu
- Marco Yu Cordero (Universidad de Puerto Rico, Mayagüez) — marco.yu@upr.edu
- Taix Pacheco García (Universidad de Puerto Rico, Mayagüez) — taix.pacheco@upr.edu
- Sebastian Hernandez Acevedo (Universidad de Puerto Rico, Mayagüez) — sebastian.hernandez10@upr.edu

## Resumen
Los desastres naturales provocan una notable actividad en redes sociales, generando tanto información crítica como contenido irrelevante o emocional. Este estudio presenta un análisis automatizado de tweets relacionados con el huracán María en Puerto Rico, dirigido a mejorar la gestión informativa durante emergencias. Inicialmente, se clasificaron los tweets en tres categorías—informativos, no informativos y neutrales—utilizando técnicas avanzadas de procesamiento de lenguaje natural (NLP), métodos clásicos de aprendizaje supervisado y redes neuronales profundas. Posteriormente, se realizó un análisis de sentimientos para determinar el tono emocional predominante en las publicaciones. Los resultados experimentales indican que los modelos supervisados, especialmente la regresión logística y redes neuronales simples, lograron altos niveles de precisión en la identificación de contenido relevante. Este enfoque evidencia su potencial para aplicaciones prácticas en tiempo real y establece una base sólida para futuras investigaciones que incorporen técnicas avanzadas de aprendizaje profundo y análisis multimodal.


## Descripción del proyecto
Puerto Rico, debido a su ubicación geográfica y condiciones climáticas, es particularmente vulnerable a desastres naturales como huracanes, terremotos y tsunamis. Estos eventos ocasionan daños físicos y sociales significativos, además de generar una notable actividad en plataformas digitales, especialmente en redes sociales como Twitter. Durante estas situaciones de crisis, los usuarios suelen compartir tanto información crítica y solicitudes de ayuda como mensajes irrelevantes o desinformativos. Esta gran cantidad de datos plantea una oportunidad y un desafío para los equipos encargados de gestionar emergencias.

Este proyecto presenta un enfoque basado en técnicas de machine learning para clasificar automáticamente publicaciones de Twitter relacionadas con el huracán María en Puerto Rico. La clasificación propuesta se realiza en tres categorías fundamentales: informativos, no informativos y neutrales. El objetivo principal es optimizar el procesamiento y análisis de grandes volúmenes de datos textuales en tiempo real, facilitando así la toma de decisiones oportunas, la coordinación eficiente de recursos y la intervención humanitaria efectiva.

Para lograr este objetivo, se emplearon técnicas avanzadas de procesamiento de lenguaje natural (NLP), modelos clásicos de aprendizaje supervisado (Naive Bayes, SVM, Regresión Logística) y redes neuronales profundas. Complementariamente, se realizó un análisis de sentimientos para profundizar en la comprensión emocional de los mensajes. El presente repositorio detalla la metodología utilizada, los experimentos desarrollados y los resultados obtenidos, proporcionando un análisis exhaustivo del desempeño, ventajas y limitaciones de cada modelo implementado.


## Estructura del repositorio
## Metodología

### Conjuntos de Datos
- **HumAID Hurricane María Dataset:** Más de 7,300 tweets anotados manualmente en categorías humanitarias (precaución, desplazados, infraestructura, solicitudes urgentes, etc.).
- **HumAID Additional Hurricanes Dataset:** Aproximadamente 34,000 tweets anotados de múltiples huracanes (Dorian, Florence, Harvey, Irma, Matthew), combinados para mejorar la generalización de los modelos.
- **ISCRAM18 Dataset, DTC2020, VT-TDB:** Utilizados para análisis exploratorio, validación cruzada y análisis de sentimiento.

### Pipeline metodológico
1. **Preprocesamiento:** Conversión a minúsculas, limpieza de URLs, menciones, hashtags, números y signos, tokenización y lematización.
2. **Análisis exploratorio:** Visualizaciones (histogramas, nubes de palabras, distribución de clases).
3. **Vectorización:** BoW (unigramas/bigramas) y TF-IDF.
4. **Modelado:** Naive Bayes, SVM, Regresión Logística, redes neuronales profundas (con regularización y ajuste de pesos por clase).
5. **Evaluación:** Matriz de confusión, precisión, recall, F1-score por clase y macro.
6. **Análisis de sentimiento:** VADER y RoBERTa para caracterización emocional.
7. **Complementos:** Modelos Random Forest, análisis de urgencia, explicabilidad (SHAP), análisis de noticias y terremotos.
## Resultados principales

- **Modelos clásicos:** Regresión Logística y SVM alcanzaron F1 macro de 0.61 y accuracy de 0.89.
- **Red neuronal mejorada:** F1 macro de 0.63 y accuracy de 0.90, mostrando mejor capacidad para manejar clases desbalanceadas.
- **Random Forest:** F1 macro de 0.36, útil por su interpretabilidad y análisis de características semánticas.
- **Análisis de sentimiento:** RoBERTa y VADER permitieron caracterizar el tono emocional, mostrando mayor negatividad en categorías asociadas a daños y pérdidas.
- **Impacto de datos adicionales:** El uso de múltiples huracanes mejoró la robustez y generalización de los modelos.
## Discusión y recomendaciones

Los resultados evidencian la efectividad de modelos neuronales mejorados y la importancia de técnicas de regularización y ajuste de pesos para abordar el desbalance de clases. Se recomienda:
- Optimizar hiperparámetros y realizar feature selection automática.
- Implementar técnicas de balanceo (SMOTE, oversampling).
- Comparar con arquitecturas avanzadas (Transformers, BERT, RoBERTa).
- Ampliar y anotar manualmente nuevos datasets.
- Profundizar en la explicabilidad (SHAP, LIME).
```
PR-Disaster-Tweets/
├── datasets/                          # Todos los conjuntos de datos utilizados
│   ├── HumAID_main/                   # Dataset HumAID completo y preprocesado
│   ├── HumAID_maria_tweets/           # Tweets anotados del huracán María + análisis y visualizaciones
│   │   ├── analysis/                  # Scripts y resultados de análisis exploratorio
│   │   ├── clean/                     # Versiones limpias de los datos
│   │   ├── figures/                   # Figuras y visualizaciones generadas
│   │   └── subcategories/             # Tweets separados por subcategoría humanitaria
│   ├── HumAID_additional_hurricanes/  # Tweets anotados de otros huracanes (Dorian, Florence, Harvey, Irma, Matthew)
│   ├── ISCRAM_maria_tweets/           # Tweets ISCRAM María + análisis, visualizaciones y limpieza
│   │   ├── analysis/                  # Scripts de análisis y visualizaciones
│   │   ├── clean/                     # Datos limpios
│   │   └── visualizaciones/           # Figuras y gráficos generados
│   ├── DTC2020/                       # Disaster Tweet Corpus 2020 (hurricane-maria-2017.ndjson)
│   └── VT_TDB_2020/                   # Dataset de terremotos 2020 y reporte original
├── HumAID_classification/             # Notebook de clasificación temática para HumAID María
├── complementary_analysis_trees_news_earthquakes/ # Análisis complementario de noticias y terremotos
├── rf_sentiment_urgency_features/     # Scripts y resultados de clasificación de sentimiento/urgencia
├── roberta_sentiment_maria/           # Análisis de sentimiento avanzado con RoBERTa sobre tweets del huracán María
├── report/                            # Reporte final del estudio
├── LICENSE.md                         # Información de la licencia
├── README.md                          # Documentación del proyecto
├── requirements.txt                   # Dependencias de Python
```


## Detalles de los conjuntos de datos y recursos

### `/datasets/HumAID_main/`
Contiene el dataset HumAID completo, preprocesado y reclasificado, útil para análisis globales y comparativos.

### `/datasets/HumAID_maria_tweets/`
Tweets anotados del huracán María (2017) con subcarpetas para análisis, limpieza y visualizaciones. Incluye:
- Datos originales y limpios (`clean/`)
- Subconjuntos por subcategoría humanitaria (`subcategories/`)
- Scripts y resultados de análisis exploratorio (`analysis/`)
- Figuras y gráficos (`figures/`)

### `/datasets/HumAID_additional_hurricanes/`
Tweets anotados de otros huracanes (Dorian 2019, Florence 2018, Harvey 2017, Irma 2017, Matthew 2016), con splits de entrenamiento, validación y prueba. Todas las subcarpetas comparten las mismas subcategorías que HumAID María.

### `/datasets/ISCRAM_maria_tweets/`
Tweets del huracán María del dataset ISCRAM 2018, con IDs hidratados y análisis avanzado. Incluye:
- Datos originales, preprocesados y limpios (`clean/`)
- Scripts y visualizaciones (`analysis/`, `visualizaciones/`)
- Notebook de análisis multidimensional (`ISCRAM_maria_tweets.ipynb`)

### `/datasets/DTC2020/`
Disaster Tweet Corpus 2020: tweets anotados de 48 desastres, incluyendo `hurricane-maria-2017.ndjson` (7,674 tweets). Útil para tareas de filtrado y clasificación binaria.

### `/datasets/VT_TDB_2020/`
Dataset de terremotos en Puerto Rico (2020) y huracán María, con archivos de IDs y reporte original. Incluye datos recolectados vía scraping y permite análisis comparativos entre desastres.


### Recursos de análisis y clasificación
- `code/`: Script de clasificación de sentimiento y urgencia (`sentiment_disaster_classifier.py`).
- `HumAID_classification/`: Notebook de clasificación temática para HumAID María.
- `rf_sentiment_urgency_features/`: Scripts y resultados de modelos de sentimiento/urgencia.
- `complementary_analysis_trees_news_earthquakes/`: Análisis de noticias y terremotos, scripts y visualizaciones.
- `roberta_sentiment_maria/`: Análisis de sentimiento avanzado con RoBERTa sobre tweets del huracán María, usando el modelo preentrenado `cardiffnlp/twitter-roberta-base-sentiment-latest` para texto informal en Twitter.
- `report/`: Reporte final del estudio en PDF.

### `/datasets/VT_TDB_2020/`
Este dataset fue desarrollado como parte del estudio **CS 4624: Multimedia, Hypertext, and Information Access** en Virginia Tech, Spring 2020. Permite realizar análisis comparativos entre desastres y explorar patrones de interacción en redes sociales. El dataset original fue proporcionado por la **Dra. Ziqian Song** e incluye los siguientes archivos con IDs de tweets:
- `PR_Earthquake.csv` – 21,755 IDs
- `PR_Earthquake_Location.csv` – 2,260,249 IDs
- `maria_tweets.json` – 801,939 IDs

Dado que estos archivos contienen únicamente IDs, se realizó un proceso de 'scraping' utilizando **Octoparse** para recolectar alrededor de 10,000 tweets relacionados con los sismos de 2020 en Puerto Rico, encontrado en el archivo `PR_Earthquake_2020.csv`. Este dataset permite realizar análisis comparativos entre desastres y explorar patrones de interacción en redes sociales.

---

## Ejecución del análisis

### Configuración e instalación

1. Clona el repositorio:
```bash
git clone https://github.com/marcoyuuu/PR-Disaster-Tweets.git
cd PR-Disaster-Tweets
```

2. **Opción 1: Usar el script `setup.bat` (Windows)**  
Ejecuta el script `setup.bat` para configurar automáticamente el entorno virtual, instalar dependencias y descargar recursos necesarios:
```cmd
setup.bat
```

3. **Opción 2: Configuración manual**  
   a. Crea un entorno virtual (opcional pero recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

   b. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```


### 📘 Notebooks y scripts de análisis

El repositorio incluye notebooks y scripts para análisis exploratorio, clasificación y visualización:

- `datasets/HumAID_maria_tweets/HumAID_maria_tweets.ipynb`: Análisis exploratorio y enriquecimiento semántico de tweets del huracán María, con limpieza, ingeniería de características, visualizaciones y análisis de sentimiento.
- `datasets/ISCRAM_maria_tweets/ISCRAM_maria_tweets.ipynb`: Análisis multidimensional de tweets ISCRAM María, con integración de metadatos, análisis temporal, sentimiento y visualizaciones avanzadas.
- `datasets/HumAID_maria_tweets/analysis/` y `datasets/ISCRAM_maria_tweets/analysis/`: Scripts Python para análisis y generación de figuras.
- `HumAID_classification/Classification_Model_Hurracane_Maria_tweets.ipynb`: Clasificación automática de temas humanitarios en HumAID María.
- `rf_sentiment_urgency_features/`: Modelos de clasificación de sentimiento y urgencia.
- `complementary_analysis_trees_news_earthquakes/`: Análisis complementario de noticias y terremotos.



## Conclusiones

Este proyecto demuestra la viabilidad de modelos automatizados para clasificar y analizar tweets durante desastres naturales en Puerto Rico, alcanzando altos niveles de precisión y generalización. La integración de múltiples datasets y técnicas avanzadas de NLP y aprendizaje profundo permite abordar tanto la relevancia informativa como la dimensión emocional de los mensajes. El código fuente, scripts de preprocesamiento, entrenamiento, análisis de sentimiento y visualización están disponibles en este repositorio, facilitando su reutilización y adaptación para investigaciones futuras sobre comunicación digital durante desastres naturales.

## Licencia
Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo LICENSE.md para más detalles.

## Agradecimientos
- [Dataset HumAID de CrisisNLP (ICWSM 2021)](https://crisisnlp.qcri.org/humaid_dataset).
- [Dataset ISCRAM 2018](https://arxiv.org/pdf/1805.05144)
- [Dataset Disaster Tweet Corpus 2020 (DTC2020)](https://zenodo.org/records/713920#:~:text=Disaster%20Tweet%20Corpus%202020%20,to%20this%20disaster%20or)
- Dataset VT_TDB_2020: Basado en el estudio **CS 4624: Multimedia, Hypertext, and Information Access**, Virginia Tech, Spring 2020.
  **Twitter Disaster Behavior: Final Report**  
  Kayley Bogemann, Shane Burchard, Jessie Butler, Austin Spencer, Taylor Thackaberry  
  Cliente: Ziqian (Alice) Song  
  Profesor: Edward Fox  
  Mayo 5, 2020