#!/usr/bin/env python
# main.py — Análisis de sentimiento desde un CSV con GPU (RTX 4070)

import os
import argparse
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Suprimir warnings de transformers y otros
warnings.filterwarnings("ignore")


def load_model(model_name: str):
    """Carga el tokenizer y el modelo, y lo envía al dispositivo (GPU si está disponible)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def analyze_sentiment(text: str, tokenizer, model, device, labels: list):
    """Analiza el texto y devuelve la etiqueta de sentimiento y la confianza."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    pred_tensor = torch.argmax(probs, dim=1)
    pred = int(pred_tensor[0].item())
    confidence = float(probs[0, pred].item())
    return labels[pred], confidence


def plot_sentiment_class_heatmap(df):
    """Genera y retorna un heatmap de porcentaje de sentimiento por clase."""
    pivot = df.pivot_table(
        index="class_label",
        columns="sentiment",
        values="tweet_id",
        aggfunc="count",
        fill_value=0,
    )
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_pct, annot=True, fmt=".1f", cmap="YlGnBu")
    plt.title("Heatmap de Porcentaje: Sentimiento vs Clase")
    plt.xlabel("Sentiment")
    plt.ylabel("class_label")
    plt.tight_layout()
    return plt.gcf()


def plot_confidence_by_sentiment(df):
    """Genera y retorna un boxplot de confianza por sentimiento."""
    plt.figure(figsize=(7, 5))
    sns.boxplot(
        x="sentiment",
        y="confidence",
        hue="sentiment",
        data=df,
        palette="Set2",
        legend=False,
    )
    plt.title("Confianza del Modelo por Sentimiento")
    plt.xlabel("Sentiment")
    plt.ylabel("Confidence")
    plt.tight_layout()
    return plt.gcf()


def main():
    """Ejecuta el análisis de sentimiento y guarda resultados y visualizaciones."""
    # 1) Parámetros del modelo
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer, model, device = load_model(model_name)
    labels = ["Negative", "Neutral", "Positive"]

    # 2) Argumentos de línea de comandos para flexibilidad
    parser = argparse.ArgumentParser(
        description="Análisis de sentimiento de tweets CSV"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=os.path.join("data", "HumAID_maria_tweets_clean.csv"),
        help="Ruta al archivo CSV de entrada",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=os.path.join("results", "sentiment_results.csv"),
        help="Ruta al archivo CSV de salida",
    )
    args = parser.parse_args()
    input_csv = args.input_csv
    output_csv = args.output_csv

    # 3) Carga del CSV
    df = pd.read_csv(input_csv)

    # 4) Detectar columna de texto
    text_col = None
    for col in ("clean_text", "tweet", "text", "Tweet", "Text"):
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        raise ValueError(
            "No se encontró ninguna columna de texto válida en el CSV (ej: 'full_text', 'tweet', 'text')."
        )

    # 5) Análisis de sentimiento
    sentiments = []
    confidences = []
    for txt in df[text_col].astype(str):
        s, c = analyze_sentiment(txt, tokenizer, model, device, labels)
        sentiments.append(s)
        confidences.append(c)

    df["sentiment"] = sentiments
    df["confidence"] = confidences

    # 6) Guardar resultados
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Análisis completado. Resultados guardados en:\n   {output_csv}")

    # Visualizaciones principales
    base_name = os.path.splitext(os.path.basename(input_csv))[0]
    results_dir = os.path.dirname(output_csv) or "."

    # 1. Heatmap de porcentaje de sentimiento por clase temática
    fig1 = plot_sentiment_class_heatmap(df)
    heatmap_path = os.path.join(results_dir, f"heatmap_{base_name}.png")
    fig1.savefig(heatmap_path, bbox_inches="tight")
    plt.close(fig1)

    # 2. Boxplot de confianza por sentimiento
    fig2 = plot_confidence_by_sentiment(df)
    boxplot_path = os.path.join(results_dir, f"boxplot_confidence_{base_name}.png")
    fig2.savefig(boxplot_path, bbox_inches="tight")
    plt.close(fig2)

    print(f"Visualizaciones guardadas en: {heatmap_path} y {boxplot_path}")


if __name__ == "__main__":
    main()
