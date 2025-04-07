# PR-Disaster-Tweets: Análisis de la percepción pública y la cobertura mediática durante desastres naturales en Puerto Rico

## Descripción del proyecto
Este proyecto se enfoca en analizar la percepción pública y la cobertura mediática durante desastres naturales en Puerto Rico, con especial énfasis en el huracán María (2017), los terremotos de 2020 y la alerta de tsunami de 2025. El análisis combina múltiples conjuntos de datos, incluyendo HumAID, ISCRAM18 y datasets personalizados recolectados, con el fin de ofrecer información sobre patrones de respuesta ante desastres, sentimiento público y necesidades humanitarias.

## Estructura del repositorio
```
PR-Disaster-Tweets/
├── datasets/                          # Todos los conjuntos de datos utilizados
│   ├── HumAID_maria_tweets/           # Archivos del dataset HumAID para el huracán María
│   ├── ISCRAM_maria_tweets/           # Archivos del dataset ISCRAM para el huracán María
│   ├── PR_Earthquake_Tweets_Jan2020/  # Dataset personalizado para los terremotos de enero 2020
│   └── PR_Advisory_Tweets_Feb_2025/   # Dataset personalizado para la alerta de tsunami de febrero 2025
├── .venv/                             # Entorno virtual para dependencias
├── CITATION.md                        # Información de citación
├── LICENSE.md                         # Información de la licencia
├── README.md                          # Documentación del proyecto
├── requirements.txt                   # Dependencias de Python
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

### `/datasets/ISCRAM_maria_tweets/`
Incluye archivos del [dataset ISCRAM 2018](https://arxiv.org/pdf/1805.05144) sobre el huracán María. Contiene aproximadamente 1,000 tweets con ID de tweet. Como el dataset original solo incluía los IDs, el texto fue recuperado mediante "hydration" con [twikit](https://github.com/d60/twikit) para analizar la percepción pública durante el evento. Visualizaciones incluidas:
- Métricas de interacción
- Distribución de "likes"

### `/datasets/PR_Earthquake_Tweets_Jan2020/`
Colección personalizada de tweets relacionados con los **terremotos de enero 2020 en Puerto Rico**. Incluye aproximadamente 300 tweets y permite comparaciones entre desastres pasados y reacciones actuales en redes sociales. Los datos fueron recolectados usando [Octoparse](https://www.octoparse.com/) con filtros por palabras clave, fechas y geolocalización. Visualizaciones incluidas:
- Métricas de interacción
- Distribución de idiomas
- Distribución de likes
- Histogramas y boxplots de longitud de tweet
- Nubes de palabras

### `/datasets/PR_Advisory_Tweets_Feb_2025/`
Colección personalizada de tweets sobre la **alerta de tsunami en febrero de 2025**. Incluye aproximadamente 250 tweets. Permite comparar desastres anteriores con reacciones actuales. Los datos fueron recolectados usando [Octoparse](https://www.octoparse.com/) aplicando filtros por palabras clave, fechas y ubicación. Este dataset se utilizará para análisis de sentimiento y detección de desinformación. Visualizaciones incluidas:
- Métricas de interacción
- Distribución de idiomas
- Distribución de likes
- Histogramas y boxplots de longitud de tweet
- Nubes de palabras

## Ejecución del análisis

### Configuración e instalación

1. Clona el repositorio:
```bash
git clone https://github.com/marcoyuuu/PR-Disaster-Tweets.git
cd PR-Disaster-Tweets
```

2. Crea un entorno virtual (opcional pero recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

### 📘 Notebook Jupyter (Análisis unificado)

Se incluye un notebook consolidado, **`PR-Disaster-Tweets.ipynb`**, en la raíz del proyecto. Este archivo integra todos los scripts de análisis y documentación en un solo formato interactivo.

#### Para ejecutarlo:

1. **Navega al directorio del proyecto** (si no estás ya allí):
   ```bash
   cd PR-Disaster-Tweets
   ```

2. **Activa el entorno virtual** (opcional pero recomendado):
   ```bash
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

3. **Instala las dependencias necesarias** (si aún no lo hiciste):
   ```bash
   pip install -r requirements.txt
   ```

4. **Lanza Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

5. Abre `PR-Disaster-Tweets.ipynb` desde el navegador o desde un IDE y ejecuta las celdas para explorar todos los datasets y visualizaciones en un solo lugar.

> Este notebook es ideal para demostraciones, presentaciones académicas y análisis exploratorios completos.
> También está disponible como PDF: `PR-Disaster-Tweets.pdf`.

### Procesamiento y análisis de datos

#### Dataset HumAID
```bash
cd datasets/HumAID_maria_tweets/analysis
python analyze_humaid.py
```

#### Dataset ISCRAM
```bash
cd datasets/ISCRAM_maria_tweets/analysis/
python analyze_ISCRAM_tweets.py
```

#### Tweets de los Terremotos (Enero 2020)
```bash
cd datasets/PR_Earthquake_Tweets_Jan2020/analysis/
python analyze_Jan2020_tweets.py
```

#### Tweets de la Alerta (Febrero 2025)
```bash
cd datasets/PR_Advisory_Tweets_Feb_2025/analysis/
python analyze_Feb2025_tweets.py
```

## Ejemplos de visualizaciones

### Análisis de tweets del huracán María (HumAID)
- **Nube de Palabras**: ![Nube](datasets/HumAID_maria_tweets/analysis/humaid_tweet_word_cloud.png)
- **Distribución por Etiqueta**: ![Etiquetas](datasets/HumAID_maria_tweets/analysis/humaid_class_distribution.png)

### Análisis de tweets del huracán María (ISCRAM)
- **Nube de Palabras**: ![Nube](datasets/ISCRAM_maria_tweets/analysis/tweet_word_cloud.png)
- **Longitud vs Likes**: ![Longitud](datasets/ISCRAM_maria_tweets/analysis/length_vs_likes.png)

### Análisis de tweets de los terremotos (Enero 2020)
- **Nube de Palabras**: ![Nube](datasets/PR_Earthquake_Tweets_Jan2020/analysis/tweet_word_cloud.png)
- **Métricas de Interacción**: ![Interacción](datasets/PR_Earthquake_Tweets_Jan2020/analysis/interaction_metrics.png)

### Análisis de tweets de la alerta de tsunami (Febrero 2025)
- **Nube de Palabras**: ![Nube](datasets/PR_Advisory_Tweets_Feb_2025/analysis/advisory_tweet_word_cloud.png)
- **Longitud vs Likes**: ![Longitud](datasets/PR_Advisory_Tweets_Feb_2025/analysis/advisory_length_vs_likes.png)

## Contexto del proyecto

Puerto Rico es altamente vulnerable a huracanes y terremotos. Aunque estos eventos son diferentes en su naturaleza, ambos generan un gran impacto mediático y provocan intensas reacciones en redes sociales. La percepción pública, la propagación de desinformación y el lenguaje emocional pueden variar significativamente. Este proyecto busca analizar y comparar cómo reaccionan los puertorriqueños ante huracanes (ej. María) y terremotos (ej. enero 2020) usando análisis de texto en redes sociales, especialmente en Twitter. Esta comparación permitirá identificar patrones de comunicación y diferencias en la percepción del riesgo ante distintas amenazas.

## Licencia
Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo LICENSE.md para más detalles.

## Agradecimientos
- Dataset HumAID  
- Dataset ISCRAM18  
- A los colaboradores e investigadores que participaron en la recolección y análisis de datos

## Citación
Las citas académicas correspondientes se encuentran en el archivo CITATION.md.