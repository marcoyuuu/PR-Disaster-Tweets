#!/usr/bin/env python
"""
Script de Análisis y Visualización del CSV "PR_Advisory_Tweets_Feb_2025.csv"

Este script realiza un análisis exploratorio de datos (EDA) sobre un conjunto de tuits
relacionados con los eventos de advertencia de tsunami en febrero de 2025. Se consideran
las siguientes columnas:
    - UTC_Time: Fecha y hora en formato UTC.
    - Tweet_Content: Contenido textual del tuit.
    - Reply_Count: Número de respuestas.
    - Repost_Count: Número de reposts.
    - Like_Count: Número de likes.
    - Bookmark_Count: Número de bookmarks.
    - Language: Idioma del tuit.

El análisis incluye:
    - Carga y preprocesamiento del dataset.
    - Conversión de 'UTC_Time' a datetime.
    - Cálculo de la longitud de cada tuit a partir de 'Tweet_Content'.
    - Visualización de las métricas de interacción mediante diagramas de caja.
    - Visualización de la distribución de la longitud de los tuits (histograma y diagrama de caja).
    - Visualización de la distribución de likes.
    - Análisis de correlación entre la longitud del tuit y la cantidad de likes.
    - Visualización de la distribución de idiomas.
    - Generación de una nube de palabras a partir de 'Tweet_Content', aplicando limpieza
      para eliminar URLs, menciones y tokens irrelevantes.

Requerimientos:
    - pandas
    - matplotlib
    - seaborn
    - wordcloud

Para instalar las dependencias, ejecuta:
    pip install pandas matplotlib seaborn wordcloud

Autor: [Tu Nombre]
Fecha: [Fecha Actual]
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

def load_data(filepath):
    """
    Carga el archivo CSV en un DataFrame de pandas.
    
    Args:
        filepath (str): Ruta al archivo CSV.
        
    Returns:
        DataFrame: Datos cargados.
    """
    try:
        df = pd.read_csv(filepath)
        print("Datos cargados exitosamente.")
        return df
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None

def preprocess_data(df):
    """
    Preprocesa los datos:
      - Convierte la columna 'UTC_Time' a datetime.
      - Calcula la longitud de cada tuit a partir de 'Tweet_Content'.
    
    Args:
        df (DataFrame): DataFrame original.
        
    Returns:
        DataFrame: DataFrame preprocesado.
    """
    # Convertir 'UTC_Time' a datetime (se usa dateutil para mayor flexibilidad)
    if 'UTC_Time' in df.columns:
        try:
            df['UTC_Time'] = pd.to_datetime(df['UTC_Time'], errors='coerce')
            print("Columna 'UTC_Time' convertida a datetime.")
        except Exception as e:
            print(f"Error al convertir 'UTC_Time': {e}")
    else:
        print("La columna 'UTC_Time' no se encontró.")
    
    # Calcular la longitud de cada tuit basado en 'Tweet_Content'
    if 'Tweet_Content' in df.columns:
        df['tweet_length'] = df['Tweet_Content'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    else:
        print("La columna 'Tweet_Content' no se encontró para calcular la longitud de los tuits.")
    
    return df

def plot_engagement_metrics(df):
    """
    Genera un diagrama de caja para visualizar la distribución de métricas de interacción:
    'Reply_Count', 'Repost_Count', 'Like_Count' y 'Bookmark_Count'.
    """
    interaction_cols = ['Reply_Count', 'Repost_Count', 'Like_Count', 'Bookmark_Count']
    existing_cols = [col for col in interaction_cols if col in df.columns]
    
    if existing_cols:
        df_interactions = df[existing_cols].melt(var_name="Métrica", value_name="Conteo")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Métrica", y="Conteo", data=df_interactions)
        plt.title("Distribución de Métricas de Interacción")
        plt.tight_layout()
        plt.savefig("advisory_engagement_metrics.png")
        plt.show()
    else:
        print("No se encontraron columnas de interacción para visualizar.")

def plot_tweet_length_distribution(df):
    """
    Genera una visualización de la distribución de la longitud de los tuits.
    Se muestran un histograma y un diagrama de caja.
    """
    if 'tweet_length' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.hist(df['tweet_length'], bins=30, edgecolor='k', alpha=0.7)
        plt.title("Distribución de la Longitud de los Tuits")
        plt.xlabel("Longitud (número de caracteres)")
        plt.ylabel("Frecuencia")
        plt.tight_layout()
        plt.savefig("advisory_tweet_length_histogram.png")
        plt.show()
        
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df['tweet_length'])
        plt.title("Diagrama de Caja de la Longitud de los Tuits")
        plt.xlabel("Longitud (número de caracteres)")
        plt.tight_layout()
        plt.savefig("advisory_tweet_length_boxplot.png")
        plt.show()
    else:
        print("La columna 'tweet_length' no está disponible para el análisis de longitud.")

def plot_likes_distribution(df):
    """
    Genera una visualización de la distribución de likes.
    Se muestra un histograma.
    """
    if 'Like_Count' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.hist(df['Like_Count'], bins=30, edgecolor='k', alpha=0.7)
        plt.title("Distribución de Likes en los Tuits")
        plt.xlabel("Número de Likes")
        plt.ylabel("Frecuencia")
        plt.tight_layout()
        plt.savefig("advisory_likes_distribution.png")
        plt.show()
    else:
        print("La columna 'Like_Count' no está disponible para analizar la distribución de likes.")

def plot_length_vs_likes(df):
    """
    Genera un gráfico de dispersión para analizar la relación entre la longitud de los tuits
    y la cantidad de likes.
    """
    if 'tweet_length' in df.columns and 'Like_Count' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='tweet_length', y='Like_Count', data=df, alpha=0.7)
        plt.title("Relación entre Longitud de Tuits y Número de Likes")
        plt.xlabel("Longitud del Tuit (caracteres)")
        plt.ylabel("Número de Likes")
        plt.tight_layout()
        plt.savefig("advisory_length_vs_likes.png")
        plt.show()
    else:
        print("No se encontraron las columnas necesarias ('tweet_length', 'Like_Count') para este análisis.")

def plot_language_distribution(df):
    """
    Visualiza la distribución de idiomas de los tuits.
    """
    if 'Language' in df.columns:
        lang_counts = df['Language'].value_counts()
        plt.figure(figsize=(8, 5))
        sns.barplot(x=lang_counts.index, y=lang_counts.values)
        plt.title("Distribución de Idiomas de los Tuits")
        plt.xlabel("Idioma")
        plt.ylabel("Cantidad de Tuits")
        plt.tight_layout()
        plt.savefig("advisory_language_distribution.png")
        plt.show()
    else:
        print("La columna 'Language' no se encontró para visualizar la distribución de idiomas.")

def generate_word_cloud(df):
    """
    Genera una nube de palabras a partir del contenido de los tuits (columna 'Tweet_Content').
    Se aplica limpieza para eliminar URLs, menciones, el token 'RT' y caracteres irrelevantes,
    además de usar una lista de stopwords personalizada.
    """
    if 'Tweet_Content' not in df.columns:
        print("La columna 'Tweet_Content' no se encontró para generar la nube de palabras.")
        return
    
    # Combinar todos los tuits en un único texto
    all_text = " ".join(df['Tweet_Content'].dropna().astype(str))
    
    # Limpieza del texto:
    # 1. Eliminar URLs.
    cleaned_text = re.sub(r'https?://\S+', '', all_text)
    # 2. Eliminar menciones (tokens que comienzan con @).
    cleaned_text = re.sub(r'@\w+', '', cleaned_text)
    # 3. Eliminar el token 'RT' (retweets).
    cleaned_text = re.sub(r'\bRT\b', '', cleaned_text)
    # 4. Eliminar caracteres que no sean letras (permitiendo acentos y la ñ en español).
    cleaned_text = re.sub(r'[^A-Za-záéíóúñüÁÉÍÓÚÑÜ\s]', '', cleaned_text)
    # 5. Convertir a minúsculas.
    cleaned_text = cleaned_text.lower()
    
    # Stopwords en español adicionales
    spanish_stopwords = {
        "de", "el", "que", "se", "la", "en", "por", "los", "las", "del", "al", 
        "un", "una", "con", "para", "este", "esta", "estos", "estas", "ese", 
        "esa", "esos", "esas", "y", "o", "u", "pero", "su", "sus", "porque",
        "son", "un", "una", "ser", "sido", "ha", "han", "hay", "qué", "etc"
    }
    
    # Stopwords personalizadas (para URLs y tokens frecuentes)
    custom_stopwords = {"https", "http", "co", "amp"}
    
    # Combinar todo en un único set
    all_stopwords = STOPWORDS.union(spanish_stopwords).union(custom_stopwords)
    
    # Generar la nube de palabras
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=all_stopwords
    ).generate(cleaned_text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Nube de Palabras (Texto Limpio)")
    plt.tight_layout()
    plt.savefig("advisory_tweet_word_cloud.png")
    plt.show()

def main():
    """
    Función principal: carga, preprocesa y genera las visualizaciones del dataset.
    """
    filepath = "../PR_Advisory_Tweets_Feb_2025.csv"  # Actualiza la ruta si es necesario
    
    if not os.path.exists(filepath):
        print(f"El archivo '{filepath}' no existe. Verifica la ruta.")
        return

    # Cargar datos
    df = load_data(filepath)
    if df is None:
        return
    
    # Mostrar información básica del dataset
    print("Información del dataset:")
    print(df.info())
    print(df.head())

    # Preprocesar datos
    df = preprocess_data(df)

    # Visualizar métricas de interacción
    plot_engagement_metrics(df)
    
    # Visualizar la distribución de la longitud de los tuits
    plot_tweet_length_distribution(df)
    
    # Visualizar la distribución de likes
    plot_likes_distribution(df)
    
    # Análisis de correlación entre la longitud del tuit y los likes
    plot_length_vs_likes(df)
    
    # Visualizar la distribución de idiomas
    plot_language_distribution(df)
    
    # Generar la nube de palabras con el texto limpio
    generate_word_cloud(df)

if __name__ == "__main__":
    main()
