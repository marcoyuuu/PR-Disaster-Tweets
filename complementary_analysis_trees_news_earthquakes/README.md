# Análisis Comparativo de Sentimiento en Noticias: Huracán María y Terremotos en Puerto Rico

Este módulo contiene scripts y recursos para el análisis de sentimiento y patrones informativos en noticias y titulares relacionados con el Huracán María (2017) y los terremotos de Puerto Rico (2019-2020). El objetivo es comparar la cobertura mediática y el tono emocional de ambos desastres, identificando diferencias en la narrativa y la percepción pública.

## Contenido

- **Procesamiento y limpieza de datos**: Scripts en Python para cargar, limpiar y preprocesar titulares de noticias.
- **Análisis de sentimiento**: Clasificación de titulares en categorías (positivo, negativo, neutral) usando modelos como VADER y reglas personalizadas.
- **Modelado y clasificación**: Implementación de árboles de decisión y random forest para clasificar tweets y titulares según su valor informativo.
- **Visualización**: Gráficos comparativos de distribución de sentimiento y palabras clave más frecuentes por clase emocional.
- **Comparación entre desastres**: Scripts para analizar diferencias en la cobertura mediática entre huracanes y terremotos.


## Estructura de carpetas

```
complementary_analysis_trees_news_earthquakes/
│
├── data/                # Datasets y archivos CSV originales
│   ├── marianews_labeled.csv
│   ├── earthquakenews_labeled.csv
│   └── ...
│
├── scripts/             # Scripts de análisis y procesamiento
│   ├── maria_tweets.py
│   ├── marianews_sent
│   ├── quakenews_sent
│   ├── news_comparison
│   ├── vader_sentlabels
│   └── ...
│
├── figures/             # Imágenes y visualizaciones generadas
│   ├── mnews_sent1.png
│   ├── mnews_sent2.png
│   ├── mnews_sent3.png
│   ├── eanews_sent1.png
│   ├── eanews_sent2.png
│   ├── eanews_sent3.png
│   ├── mtweet1.png
│   ├── mtweet2.png
│   ├── news_comparison1.png
│   └── ...
│
├── utils/               # Utilidades, scripts auxiliares, funciones comunes
│   ├── filterurls
│   ├── getheadline_cont
│   └── ...
│
└── README.md            # Documentación principal del módulo
```

### Descripción de carpetas y archivos

- **data/**: Datasets de titulares de noticias etiquetados con sentimiento.
- **scripts/**: Scripts para análisis de sentimiento, clasificación y comparación.
- **figures/**: Visualizaciones generadas de los análisis.
- **utils/**: Utilidades y funciones auxiliares para procesamiento de datos.

## Metodología

1. **Limpieza y preprocesamiento**: Eliminación de ruido textual, normalización y lematización.
2. **Etiquetado de sentimiento**: Uso de VADER y reglas para asignar clases emocionales a titulares.
3. **Modelado supervisado**: Entrenamiento de modelos de clasificación para distinguir entre información relevante, neutral y no informativa.
4. **Visualización**: Gráficos de barras y nubes de palabras para explorar la distribución de sentimiento y términos clave.
5. **Comparación cruzada**: Análisis de diferencias en la narrativa mediática entre huracanes y terremotos.

## Resultados esperados

- Distribución comparativa de sentimiento en noticias sobre huracanes y terremotos.
- Palabras clave más frecuentes por clase emocional y desastre.
- Métricas de desempeño de los modelos de clasificación (F1, accuracy).
- Visualizaciones que facilitan la interpretación de patrones mediáticos y emocionales.

## Requisitos

- Python 3.8+
- Bibliotecas: pandas, numpy, scikit-learn, matplotlib, seaborn, nltk, vaderSentiment

## Créditos

Desarrollado como parte del proyecto de análisis de desastres naturales y respuesta mediática en Puerto Rico.
