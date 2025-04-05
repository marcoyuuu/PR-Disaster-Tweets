#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Análisis de PR_Advisory_Tweets_Feb_2025.csv

Este script realiza la carga de datos, limpieza, ingeniería de características y 
análisis exploratorio (EDA) para el dataset de advertencia del 2025.
Se generan visualizaciones estáticas (PNG) e interactivas (HTML), las cuales se 
guardan en el directorio 'analysis/'.
"""

# -------------------------
# 1. Importar Librerías
# -------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Visualizaciones interactivas con Plotly
import plotly.express as px
import plotly.graph_objects as go

# PyWaffle para gráficos tipo waffle
from pywaffle import Waffle

# WordCloud para visualización de palabras frecuentes
from wordcloud import WordCloud

# -------------------------
# 2. Configuración del Directorio de Análisis
# -------------------------
directorio_analisis = "analysis"
os.makedirs(directorio_analisis, exist_ok=True)

# -------------------------
# 3. Carga de Datos
# -------------------------
# Cargar el archivo CSV del dataset de advertencia 2025
df = pd.read_csv("PR_Advisory_Tweets_Feb_2025.csv")
print("Datos cargados. Número de filas:", df.shape[0])

# -------------------------
# 4. Limpieza de Datos e Ingeniería de Características
# -------------------------
# Paso 1: Eliminar columnas con todos los valores nulos
df = df.dropna(axis=1, how='all')

# Paso 2: Eliminar columnas redundantes o no necesarias
columnas_redundantes = [
    'id', 'object', 'result_position', 'task_id', 'internal_unique_id',
    'tweet_url', 'original_tweet_url', 'name', 'user_id', 'username',
    'published_at', 'content', 'views_count', 'retweet_count', 'likes',
    'quote_count', 'reply_count', 'bookmarks_count',
    'media_0_thumbnail', 'media_0_type', 'media_0_url',
    'media_1_thumbnail', 'media_1_type', 'media_1_url',
    'media_2_thumbnail', 'media_2_type', 'media_2_url',
    'media_3_thumbnail', 'media_3_type', 'media_3_url',
    'binded_media_url', 'binded_media_domain', 'binded_media_thumbnail_url',
    'binded_media_title', 'binded_media_description',
    'is_retweeted', 'is_quoted', 'collected_at', 'input_url'
]
df = df.drop(columns=[col for col in columnas_redundantes if col in df.columns])

# Paso 3: Rellenar y convertir las columnas de engagement a enteros
columnas_engagement = ["Reply_Count", "Repost_Count", "Like_Count", "Bookmark_Count"]
for col in columnas_engagement:
    df[col] = df[col].fillna(0).astype(int)

# Paso 4: Crear la columna Total_Engagement (suma de interacciones)
df["Total_Engagement"] = df["Reply_Count"] + df["Repost_Count"] + df["Like_Count"] + df["Bookmark_Count"]

# Paso 5: Convertir la columna UTC_Time a datetime para análisis temporal
df["Timestamp_UTC"] = pd.to_datetime(df["UTC_Time"])

# Paso 6: Seleccionar las columnas relevantes para el análisis
df_selected = df[["Post_ID", "Tweet_Content", "Total_Engagement", "Language", "Timestamp_UTC"]].copy()

# Paso 7: Agregar columna de longitud del tweet
df_selected["Tweet_Length"] = df_selected["Tweet_Content"].str.len()

# Paso 8: Agregar características temporales adicionales
df_selected["Hour"] = df_selected["Timestamp_UTC"].dt.hour
df_selected["Weekday"] = df_selected["Timestamp_UTC"].dt.day_name()
df_selected["Date"] = df_selected["Timestamp_UTC"].dt.date

# Paso 9: Crear la categoría de longitud del tweet según intervalos definidos
bins_longitud = [0, 80, 140, 200, 280, df_selected["Tweet_Length"].max()]
etiquetas_longitud = ["Muy Corto", "Corto", "Medio", "Largo", "Muy Largo"]
df_selected["Tweet_Length_Category"] = pd.cut(
    df_selected["Tweet_Length"],
    bins=bins_longitud,
    labels=etiquetas_longitud,
    include_lowest=True
)

# -------------------------
# 5. Análisis de Datos y Visualizaciones
# -------------------------

# SECCIÓN A: Análisis de Distribución
# ----------------------------------
# Histograma: Distribución de Total Engagement (bruto)
plt.figure()
df_selected["Total_Engagement"].plot(kind="hist", bins=10, title="Distribución de Total Engagement")
plt.xlabel("Total Engagement")
plt.ylabel("Frecuencia")
plt.savefig(os.path.join(directorio_analisis, "total_engagement_hist.png"))
plt.close()

# Boxplot: Total Engagement (bruto)
plt.figure()
df_selected["Total_Engagement"].plot(kind="box", title="Boxplot de Total Engagement")
plt.ylabel("Total Engagement")
plt.savefig(os.path.join(directorio_analisis, "total_engagement_box.png"))
plt.close()

# Transformación logarítmica de Total Engagement
df_selected["Log_Total_Engagement"] = np.log1p(df_selected["Total_Engagement"])
plt.figure()
df_selected["Log_Total_Engagement"].plot(kind="hist", bins=10, title="Total Engagement Log-Transformado")
plt.xlabel("Log(1 + Total Engagement)")
plt.ylabel("Frecuencia")
plt.savefig(os.path.join(directorio_analisis, "log_total_engagement_hist.png"))
plt.close()

# Boxplot: Total Engagement filtrado (sin outliers extremos usando IQR)
Q1 = df_selected["Total_Engagement"].quantile(0.25)
Q3 = df_selected["Total_Engagement"].quantile(0.75)
IQR = Q3 - Q1
df_filtrado = df_selected[
    (df_selected["Total_Engagement"] >= Q1 - 1.5 * IQR) &
    (df_selected["Total_Engagement"] <= Q3 + 1.5 * IQR)
]
plt.figure()
df_filtrado["Total_Engagement"].plot(kind="box", title="Boxplot de Total Engagement Filtrado (Sin Outliers)")
plt.ylabel("Total Engagement")
plt.savefig(os.path.join(directorio_analisis, "filtered_total_engagement_box.png"))
plt.close()

# SECCIÓN A: Distribución y Categorías de Longitud del Tweet
# ------------------------------------------------------
# Histograma: Distribución de la Longitud del Tweet
plt.figure()
df_selected["Tweet_Length"].plot(kind="hist", bins=10, title="Distribución de la Longitud del Tweet")
plt.xlabel("Longitud del Tweet (caracteres)")
plt.ylabel("Frecuencia")
plt.savefig(os.path.join(directorio_analisis, "tweet_length_hist.png"))
plt.close()

# Boxplot: Longitud del Tweet
plt.figure()
df_selected["Tweet_Length"].plot(kind="box", title="Boxplot de Longitud del Tweet")
plt.ylabel("Longitud del Tweet (caracteres)")
plt.savefig(os.path.join(directorio_analisis, "tweet_length_box.png"))
plt.close()

# Gráfico de barras: Conteo de Tweets por Categoría de Longitud
plt.figure()
sns.countplot(
    x="Tweet_Length_Category",
    hue="Tweet_Length_Category",
    data=df_selected,
    order=["Muy Corto", "Corto", "Medio", "Largo", "Muy Largo"],
    palette="pastel",
    legend=False
)
plt.title("Conteo de Tweets por Categoría de Longitud")
plt.xlabel("Categoría de Longitud del Tweet")
plt.ylabel("Número de Tweets")
plt.savefig(os.path.join(directorio_analisis, "tweet_length_countplot.png"))
plt.close()

# Gráfico Waffle: Distribución de Categorías de Longitud del Tweet
counts_longitud = df_selected["Tweet_Length_Category"].value_counts().sort_index().to_dict()
fig = plt.figure(
    FigureClass=Waffle,
    rows=5,
    values=counts_longitud,
    figsize=(10, 4),
    title={"label": "Distribución de Longitud del Tweet", "loc": "center"},
    legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)},
    colors=["#b3e2cd", "#fdcdac", "#cbd5e8", "#f4cae4", "#e6f5c9"],
    block_arranging_style='snake'
)
plt.savefig(os.path.join(directorio_analisis, "tweet_length_waffle.png"))
plt.close()

# SECCIÓN C: Distribución por Idioma
# -------------------------------
# Gráfico de barras: Conteo de Tweets por Idioma
if "Language" in df_selected.columns:
    plt.figure()
    sns.countplot(
        x="Language",
        hue="Language",
        data=df_selected,
        palette="pastel",
        legend=False
    )
    plt.title("Conteo de Tweets por Idioma")
    plt.xlabel("Idioma")
    plt.ylabel("Cantidad de Tweets")
    plt.savefig(os.path.join(directorio_analisis, "language_countplot.png"))
    plt.close()

# Gráfico Waffle: Distribución de Idioma
counts_idioma = df_selected["Language"].value_counts().to_dict()
fig = plt.figure(
    FigureClass=Waffle,
    rows=5,
    values=counts_idioma,
    figsize=(8, 4),
    title={"label": "Distribución de Idioma (en vs es)", "loc": "center"},
    legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)},
    colors=["#66c2a5", "#fc8d62"],
    block_arranging_style='snake'
)
plt.savefig(os.path.join(directorio_analisis, "language_waffle.png"))
plt.close()

# SECCIÓN B: Tendencias Temporales
# ----------------------------
# Gráfico de línea: Total Engagement a lo largo del tiempo (bruto)
plt.figure()
df_selected.sort_values("Timestamp_UTC").plot(
    x="Timestamp_UTC", y="Total_Engagement", kind="line",
    title="Total Engagement a lo largo del Tiempo"
)
plt.xlabel("Tiempo")
plt.ylabel("Total Engagement")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(directorio_analisis, "time_raw_engagement_line.png"))
plt.close()

# Gráfico de línea: Total Engagement Log-Transformado a lo largo del tiempo
plt.figure()
df_selected.sort_values("Timestamp_UTC").plot(
    x="Timestamp_UTC", y="Log_Total_Engagement", kind="line",
    title="Total Engagement Log-Transformado a lo largo del Tiempo"
)
plt.xlabel("Tiempo")
plt.ylabel("Log(1 + Total Engagement)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(directorio_analisis, "time_log_engagement_line.png"))
plt.close()

# Gráfico de línea: Longitud del Tweet a lo largo del tiempo
plt.figure()
df_selected.sort_values("Timestamp_UTC").plot(
    x="Timestamp_UTC", y="Tweet_Length", kind="line",
    title="Longitud del Tweet a lo largo del Tiempo"
)
plt.xlabel("Tiempo")
plt.ylabel("Longitud del Tweet")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(directorio_analisis, "tweet_length_time_line.png"))
plt.close()

# SECCIÓN C: Comparaciones por Idioma
# --------------------------------------
# Gráfico de barras: Promedio de Total Engagement por Idioma
if "Language" in df_selected.columns:
    plt.figure()
    sns.barplot(x="Language", y="Total_Engagement", data=df_selected, hue="Language", palette="pastel", legend=False)
    plt.title("Promedio de Engagement por Idioma")
    plt.xlabel("Idioma")
    plt.ylabel("Engagement Promedio")
    plt.savefig(os.path.join(directorio_analisis, "language_bar_engagement.png"))
    plt.close()

# Gráfico tipo strip: Distribución de Engagement (bruto) por Idioma
if "Language" in df_selected.columns:
    plt.figure()
    sns.stripplot(x="Language", y="Total_Engagement", data=df_selected, hue="Language", palette="Set2", jitter=True, legend=False)
    plt.title("Distribución de Engagement por Idioma")
    plt.xlabel("Idioma")
    plt.ylabel("Total Engagement")
    plt.savefig(os.path.join(directorio_analisis, "language_strip_engagement.png"))
    plt.close()

# Gráfico de barras: Promedio de Engagement Log-Transformado por Idioma
df_selected["Log_Total_Engagement"] = np.log1p(df_selected["Total_Engagement"])
if "Language" in df_selected.columns:
    plt.figure()
    sns.barplot(x="Language", y="Log_Total_Engagement", data=df_selected, hue="Language", palette="muted", legend=False)
    plt.title("Promedio de Engagement Log-Transformado por Idioma")
    plt.xlabel("Idioma")
    plt.ylabel("Promedio Log(1 + Engagement)")
    plt.savefig(os.path.join(directorio_analisis, "language_bar_log_engagement.png"))
    plt.close()

# Gráfico tipo strip: Distribución de Engagement Log-Transformado por Idioma
if "Language" in df_selected.columns:
    plt.figure()
    sns.stripplot(x="Language", y="Log_Total_Engagement", data=df_selected, hue="Language", palette="coolwarm", jitter=True, legend=False)
    plt.title("Distribución de Engagement Log-Transformado por Idioma")
    plt.xlabel("Idioma")
    plt.ylabel("Log(1 + Total Engagement)")
    plt.savefig(os.path.join(directorio_analisis, "language_strip_log_engagement.png"))
    plt.close()

# SECCIÓN D: Relación entre Características
# -----------------------------------------
# Gráfico de dispersión: Longitud del Tweet vs. Total Engagement
plt.figure()
df_selected.plot(kind="scatter", x="Tweet_Length", y="Total_Engagement", title="Longitud del Tweet vs Total Engagement")
plt.xlabel("Longitud del Tweet (caracteres)")
plt.ylabel("Total Engagement")
plt.savefig(os.path.join(directorio_analisis, "tweet_length_vs_engagement_scatter.png"))
plt.close()

# Gráfico de barras: Promedio de Engagement por Categoría de Longitud del Tweet
plt.figure()
sns.barplot(x="Tweet_Length_Category", y="Total_Engagement", hue="Tweet_Length_Category", data=df_selected, palette="pastel", legend=False)
plt.title("Promedio de Engagement por Categoría de Longitud del Tweet")
plt.xlabel("Rango de Longitud del Tweet")
plt.ylabel("Engagement Promedio")
plt.savefig(os.path.join(directorio_analisis, "tweet_length_category_bar.png"))
plt.close()

# SECCIÓN E: Matriz de Correlación
# -----------------------------
# Convertir la columna Language a etiquetas numéricas para el análisis de correlación
df_selected["Language_Code"] = df_selected["Language"].map({"en": 0, "es": 1})
matriz_corr = df_selected[["Total_Engagement", "Tweet_Length", "Language_Code"]].corr()

plt.figure(figsize=(6, 4))
sns.heatmap(matriz_corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlación (Incluyendo Idioma)")
plt.savefig(os.path.join(directorio_analisis, "correlation_heatmap.png"))
plt.close()

# SECCIÓN F: Visualizaciones Interactivas con Plotly
# ---------------------------------------
# Gráfico de línea interactivo: Total Engagement a lo largo del tiempo
fig = px.line(
    df_selected.sort_values("Timestamp_UTC"),
    x="Timestamp_UTC",
    y="Total_Engagement",
    title="Total Engagement a lo largo del Tiempo (Interactivo)",
    labels={"Timestamp_UTC": "Tiempo", "Total_Engagement": "Total Engagement"}
)
fig.update_layout(xaxis_title="Tiempo", yaxis_title="Total Engagement")
fig.write_html(os.path.join(directorio_analisis, "interactive_total_engagement_line.html"))

# Gráfico de línea interactivo: Total Engagement Log-Transformado a lo largo del tiempo
fig = px.line(
    df_selected.sort_values("Timestamp_UTC"),
    x="Timestamp_UTC",
    y="Log_Total_Engagement",
    title="Total Engagement Log-Transformado a lo largo del Tiempo (Interactivo)",
    labels={"Timestamp_UTC": "Tiempo", "Log_Total_Engagement": "Log(1 + Engagement)"}
)
fig.update_layout(xaxis_title="Tiempo", yaxis_title="Log(1 + Total Engagement)")
fig.write_html(os.path.join(directorio_analisis, "interactive_log_engagement_line.html"))

# Histograma interactivo: Distribución de Total Engagement
fig = px.histogram(
    df_selected,
    x="Total_Engagement",
    nbins=30,
    title="Distribución de Total Engagement (Interactivo)",
    labels={"Total_Engagement": "Total Engagement"}
)
fig.update_layout(yaxis_title="Cantidad de Tweets")
fig.write_html(os.path.join(directorio_analisis, "interactive_total_engagement_hist.html"))

# Histograma interactivo: Distribución de Total Engagement Log-Transformado
fig = px.histogram(
    df_selected,
    x="Log_Total_Engagement",
    nbins=30,
    title="Distribución de Total Engagement Log-Transformado (Interactivo)",
    labels={"Log_Total_Engagement": "Log(1 + Engagement)"}
)
fig.update_layout(yaxis_title="Cantidad de Tweets")
fig.write_html(os.path.join(directorio_analisis, "interactive_log_engagement_hist.html"))

# SECCIÓN F: Word Cloud - Palabras Frecuentes en Tweets
custom_stopwords = {
    "https", "RT", "co", "amp",
    "de", "a", "t", "el", "que", "se", "la", "en", "por",
    "los", "las", "del", "al", "un", "una", "con", "para",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "y", "o", "u", "pero", "su", "sus", "porque", "son",
    "ser", "sido", "ha", "han", "hay", "qué", "etc", "PuertoRico",
    "the", "is", "to", "of", "and", "in", "for", "on", "at", "with", "as", "this", "that", "it", "are",
}

def generar_wordcloud_por_idioma(codigo_idioma, archivo_salida):
    texto = " ".join(df_selected[df_selected["Language"] == codigo_idioma]["Tweet_Content"].dropna().astype(str))
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=200,
        stopwords=custom_stopwords
    ).generate(texto)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Palabras Frecuentes en Tweets ({codigo_idioma})", fontsize=16)
    plt.savefig(os.path.join(directorio_analisis, archivo_salida))
    plt.close()

# Generar word clouds para inglés y español
generar_wordcloud_por_idioma("en", "wordcloud_en.png")
generar_wordcloud_por_idioma("es", "wordcloud_es.png")

# -------------------------
# Fin del Script de Análisis
# -------------------------
print("Análisis completado. Las figuras se han guardado en el directorio 'analysis/'.")
