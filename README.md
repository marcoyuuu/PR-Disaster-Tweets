# PR-Disaster-Tweets: Análisis de la percepción pública y la cobertura mediática durante desastres naturales en Puerto Rico

## Descripción del proyecto
Este proyecto se enfoca en analizar la percepción pública y la cobertura mediática durante desastres naturales en Puerto Rico, con especial énfasis en el huracán María (2017), los terremotos de 2020 y la alerta de tsunami de 2025. El análisis combina múltiples conjuntos de datos, incluyendo HumAID, ISCRAM18 y datasets personalizados recolectados, con el fin de ofrecer información sobre patrones de respuesta ante desastres, sentimiento público y necesidades humanitarias.

## Estructura del repositorio
```
PR-Disaster-Tweets/
├── datasets/                          # Todos los conjuntos de datos utilizados
│   ├── HumAID_maria_tweets/           # Archivos del dataset HumAID para el huracán María
│   ├── ISCRAM_maria_tweets/           # Archivos del dataset ISCRAM para el huracán María
│   ├── VT_TDB_2020/                   # Dataset de tweets sobre terremotos en 2020
│   └── DTC2020/                       # Disaster Tweet Corpus 2020
├── LICENSE.md                         # Información de la licencia
├── README.md                          # Documentación del proyecto
├── requirements.txt                   # Dependencias de Python
├── setup.bat                          # Setup script para proyecto (Windows)
```

## Detalles de los conjuntos de datos

### `/datasets/HumAID_maria_tweets/`
Contiene datos anotados de desastres (huracanes y terremotos) del [Dataset HumAID de CrisisNLP (ICWSM 2021)](https://crisisnlp.qcri.org/humaid_dataset). Este dataset incluye aproximadamente 7,300 tweets clasificados por temas humanitarios y sirve como base para tareas de clasificación temática supervisada. Las subcategorías incluyen:
- Precaución y consejos
- Personas desplazadas y evacuaciones
- Daños a infraestructura y servicios
- Personas heridas o fallecidas
- No humanitario
- Otra información relevante
- Solicitudes o necesidades urgentes
- Voluntariado o donaciones de rescate
- Simpatía y apoyo

### `/datasets/HumAID_additional_hurricanes/`
Contiene datos anotados de desastres relacionados con huracanes adicionales del [Dataset HumAID de CrisisNLP (ICWSM 2021)](https://crisisnlp.qcri.org/humaid_dataset). Estos datasets incluyen tweets clasificados por temas humanitarios y son útiles para tareas de clasificación temática supervisada. Los huracanes incluidos son:

#### `/datasets/HumAID_additional_hurricanes/hurricane_dorian_2019/`
Este dataset contiene tweets relacionados con el huracán Dorian (2019) y comparte las mismas subcategorías que `/datasets/HumAID_maria_tweets/`.

#### `/datasets/HumAID_additional_hurricanes/hurricane_florence_2018/`
Este dataset contiene tweets relacionados con el huracán Florence (2018) y comparte las mismas subcategorías que `/datasets/HumAID_maria_tweets/`.

#### `/datasets/HumAID_additional_hurricanes/hurricane_harvey_2017/`
Este dataset contiene tweets relacionados con el huracán Harvey (2017) y comparte las mismas subcategorías que `/datasets/HumAID_maria_tweets/`.

#### `/datasets/HumAID_additional_hurricanes/hurricane_irma_2017/`
Este dataset contiene tweets relacionados con el huracán Irma (2017) y comparte las mismas subcategorías que `/datasets/HumAID_maria_tweets/`.

#### `/datasets/HumAID_additional_hurricanes/hurricane_matthew_2016/`
Este dataset contiene tweets relacionados con el huracán Matthew (2016) y comparte las mismas subcategorías que `/datasets/HumAID_maria_tweets/`.

### `/datasets/ISCRAM_maria_tweets/`
Incluye archivos del [dataset ISCRAM 2018](https://arxiv.org/pdf/1805.05144) sobre el huracán María. Contiene aproximadamente 2,500 tweets con ID de tweet. Como el dataset original solo incluía los IDs, el texto fue recuperado mediante "hydration" con [twikit](https://github.com/d60/twikit) para analizar la percepción pública durante el evento. Visualizaciones incluidas:
- Métricas de interacción
- Distribución de "likes"

### `/datasets/DTC2020/`
Incluye datos del [dataset Disaster Tweet Corpus 2020 (DTC2020)](https://doi.org/10.5281/zenodo.3713919), un conjunto de tweets recopilados durante 48 desastres en 10 tipos de desastres. Este dataset contiene tweets anotados manualmente para indicar si están relacionados con un desastre específico o no. El archivo `hurricane-maria-2017.ndjson` contiene 7,674 tweets relacionados con el huracán María (2017). Este dataset es ideal para tareas de filtrado y clasificación de tweets relacionados con desastres.
Referencias:
- Wiegmann, M., Kersten, J., Klan, F., Potthast, M., Stein, B. (2020). Analysis of Filtering Models for Disaster-Related Tweets. Proceedings of the 17th ISCRAM.
- Otros trabajos citados en la documentación del dataset.

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

### 📘 Notebook Jupyter

Este proyecto incluye dos notebooks principales para el análisis de datos:

#### 1. `datasets\HumAID_maria_tweets\HumAID_maria_tweets.ipynb`
**Análisis Exploratorio y Enriquecimiento Semántico de Tweets del Huracán María (`HumAID_maria_tweets.csv`)**

Este notebook realiza un análisis exhaustivo del dataset anotado de tweets del huracán María. El enfoque es multidimensional, combinando:
- **Limpieza de texto:** Eliminación de ruido textual, normalización y manejo de stopwords.
- **Ingeniería de características:** Creación de métricas como longitud del texto, frecuencia de hashtags y análisis de sentimiento.
- **Visualizaciones:** Histogramas, nubes de palabras y gráficos de barras para explorar patrones de datos.
- **Análisis de sentimiento:** Clasificación de tweets en categorías emocionales (positivos, negativos, neutrales).
- **Insights clave:** Identificación de temas humanitarios y patrones de comunicación durante el huracán.

#### 2. `datasets\ISCRAM_maria_tweets\ISCRAM_maria_tweets.ipynb`

Este notebook se centra en el análisis multidimensional de tweets relacionados con el huracán María, integrando datos crudos, metadatos y técnicas avanzadas de NLP. Las etapas clave incluyen:
- **Preprocesamiento y limpieza:** Renombrar columnas, eliminar ruido textual, normalización y detección de idioma.
- **Ingeniería de características:** Creación de métricas como longitud del texto, identificación de retweets y puntuación de engagement.
- **Análisis temporal:** Exploración de series de tiempo para correlacionar actividad en redes sociales con eventos climáticos.
- **Análisis de sentimiento y emociones:** Clasificación emocional y análisis de solidaridad, urgencia y críticas.
- **Visualizaciones:** Mapas de calor, nubes de palabras y gráficos de barras para explorar patrones de datos.
- **Integración con datos externos:** Comparación de menciones en redes sociales con estadísticas oficiales (ej. FEMA, NOAA).

Ambos notebooks están diseñados para proporcionar análisis detallados y visualizaciones interactivas que faciliten la comprensión de los datos sociales generados durante desastres naturales.

## Contexto del proyecto

Puerto Rico es altamente vulnerable a huracanes y terremotos. Aunque estos eventos son diferentes en su naturaleza, ambos generan un gran impacto mediático y provocan intensas reacciones en redes sociales. La percepción pública, la propagación de desinformación y el lenguaje emocional pueden variar significativamente. Este proyecto busca analizar y comparar cómo reaccionan los puertorriqueños ante huracanes (ej. María) y terremotos (ej. enero 2020) usando análisis de texto en redes sociales, especialmente en Twitter. Esta comparación permitirá identificar patrones de comunicación y diferencias en la percepción del riesgo ante distintas amenazas.

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