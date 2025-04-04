#!/usr/bin/env python
"""
Script de Análisis y Visualización del CSV "ISCRAM_maria_tweets.csv"

Este script realiza un análisis exploratorio de datos (EDA) sobre un conjunto de tuits "hidratados".
El análisis incluye:
    - Carga y preprocesamiento del dataset.
    - Conversión de la columna de fecha a formato datetime.
    - Análisis de la longitud de los tuits.
    - Visualización de la distribución de likes.
    - Visualización de métricas de interacción (retweets y likes) mediante diagramas de caja.
    - Análisis de correlación entre la longitud de los tuits y la cantidad de likes.
    - Generación de una nube de palabras a partir del contenido de los tuits, aplicando limpieza del texto 
      para eliminar URLs, menciones y tokens irrelevantes.

Requerimientos:
    - pandas
    - matplotlib
    - seaborn
    - wordcloud

Para instalar las dependencias:
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
      - Convierte la columna 'created_at' a datetime utilizando un formato adecuado.
      - Calcula la longitud de cada tuit.
    
    Args:
        df (DataFrame): DataFrame original.
        
    Returns:
        DataFrame: DataFrame preprocesado.
    """
    # Convertir 'created_at' a datetime con formato explícito
    if 'created_at' in df.columns:
        try:
            df['created_at'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce')
            print("Columna 'created_at' convertida a datetime.")
        except Exception as e:
            print(f"Error al convertir 'created_at': {e}")
    else:
        print("No se encontró la columna 'created_at'.")
    
    # Calcular la longitud de cada tuit (en número de caracteres) basado en la columna 'text'
    if 'text' in df.columns:
        df['tweet_length'] = df['text'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    else:
        print("No se encontró la columna 'text' para calcular la longitud de los tuits.")
    
    return df

def plot_engagement_metrics(df):
    """
    Genera un diagrama de caja para visualizar la distribución de métricas de interacción:
    'retweet_count' y 'like_count'.
    """
    engagement_cols = []
    for col in ['retweet_count', 'like_count']:
        if col in df.columns:
            engagement_cols.append(col)
    
    if engagement_cols:
        df_engagement = df[engagement_cols].melt(var_name="Métrica", value_name="Conteo")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Métrica", y="Conteo", data=df_engagement)
        plt.title("Distribución de Métricas de Interacción")
        plt.tight_layout()
        plt.savefig("engagement_metrics.png")
        plt.show()
    else:
        print("No se encontraron columnas de métricas de interacción (retweet_count, like_count).")

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
        plt.savefig("tweet_length_histogram.png")
        plt.show()
        
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df['tweet_length'])
        plt.title("Diagrama de Caja de la Longitud de los Tuits")
        plt.xlabel("Longitud (número de caracteres)")
        plt.tight_layout()
        plt.savefig("tweet_length_boxplot.png")
        plt.show()
    else:
        print("La columna 'tweet_length' no está disponible para el análisis de longitud.")

def plot_likes_distribution(df):
    """
    Genera una visualización de la distribución de likes.
    Se muestra un histograma.
    """
    if 'like_count' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.hist(df['like_count'], bins=30, edgecolor='k', alpha=0.7)
        plt.title("Distribución de Likes en los Tuits")
        plt.xlabel("Número de Likes")
        plt.ylabel("Frecuencia")
        plt.tight_layout()
        plt.savefig("likes_distribution.png")
        plt.show()
    else:
        print("La columna 'like_count' no está disponible para analizar la distribución de likes.")

def plot_length_vs_likes(df):
    """
    Genera un gráfico de dispersión para analizar la relación entre la longitud de los tuits
    y la cantidad de likes.
    """
    if 'tweet_length' in df.columns and 'like_count' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='tweet_length', y='like_count', data=df, alpha=0.7)
        plt.title("Relación entre Longitud de Tuits y Número de Likes")
        plt.xlabel("Longitud del Tuit (caracteres)")
        plt.ylabel("Número de Likes")
        plt.tight_layout()
        plt.savefig("length_vs_likes.png")
        plt.show()
    else:
        print("No se encontraron las columnas necesarias ('tweet_length', 'like_count') para este análisis.")

def generate_word_cloud(df):
    """
    Genera una nube de palabras a partir del contenido de los tuits (columna 'text').
    Se aplica limpieza para eliminar URLs, menciones, el token 'RT' y caracteres irrelevantes,
    además de usar una lista de stopwords personalizada.
    """
    if 'text' not in df.columns:
        print("La columna 'text' no se encontró para generar la nube de palabras.")
        return
    
    # Combinar todos los tuits en un único texto
    all_text = " ".join(df['text'].dropna().astype(str))
    
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
    
    # Definir stopwords personalizadas y combinarlas con las del paquete wordcloud.
    custom_stopwords = set(["https", "http", "co", "amp"])
    stopwords = STOPWORDS.union(custom_stopwords)
    
    # Generar la nube de palabras.
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          stopwords=stopwords).generate(cleaned_text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Nube de Palabras (Texto Limpio)")
    plt.tight_layout()
    plt.savefig("tweet_word_cloud.png")
    plt.show()

def main():
    """
    Función principal: carga, preprocesa y genera las visualizaciones del dataset.
    """
    filepath = "../ISCRAM_maria_tweets.csv"  # Actualiza la ruta si es necesario
    
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

    # Generar visualizaciones de métricas de interacción
    plot_engagement_metrics(df)
    
    # Visualizar la distribución de la longitud de los tuits
    plot_tweet_length_distribution(df)
    
    # Visualizar la distribución de likes
    plot_likes_distribution(df)
    
    # Análisis de correlación entre longitud de tuit y likes
    plot_length_vs_likes(df)
    
    # Generar la nube de palabras con el texto limpio
    generate_word_cloud(df)

if __name__ == "__main__":
    main()
