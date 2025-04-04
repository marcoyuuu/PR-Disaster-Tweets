#!/usr/bin/env python
"""
Script for Analysis and Visualization of the CSV "HumAID_maria_tweets.csv"

This script performs an exploratory data analysis (EDA) on a dataset that contains the following columns:
    - tweet_id: Unique identifier for each tweet.
    - tweet_text: Content of the tweet.
    - class_label: A label describing the tweet (e.g., "other_relevant_information").
    - split: Dataset split (e.g., "train", "test", etc.)

The analysis includes:
    - Loading and preprocessing the dataset.
    - Calculating tweet length based on tweet_text.
    - Visualizing the distribution of class labels and dataset splits.
    - Visualizing the distribution of tweet lengths (histogram and boxplot).
    - Generating a word cloud from tweet_text after cleaning the text by removing URLs,
      mentions, non-alphabetic characters, and common English stopwords.

Requirements:
    - pandas
    - matplotlib
    - seaborn
    - wordcloud

To install the dependencies, run:
    pip install pandas matplotlib seaborn wordcloud

Author: [Your Name]
Date: [Current Date]
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

def load_data(filepath):
    """
    Loads the CSV file into a pandas DataFrame.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        DataFrame: Loaded data.
    """
    try:
        df = pd.read_csv(filepath)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading the file: {e}")
        return None

def preprocess_data(df):
    """
    Preprocesses the data:
      - Calculates the length of each tweet based on the 'tweet_text' column.
    
    Args:
        df (DataFrame): Original DataFrame.
        
    Returns:
        DataFrame: Preprocessed DataFrame.
    """
    if 'tweet_text' in df.columns:
        df['tweet_length'] = df['tweet_text'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    else:
        print("Column 'tweet_text' not found for calculating tweet length.")
    
    return df

def plot_class_distribution(df):
    """
    Plots the distribution of class labels.
    """
    if 'class_label' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x='class_label', order=df['class_label'].value_counts().index)
        plt.title("Distribution of Class Labels")
        plt.xlabel("Class Label")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("humaid_class_distribution.png")
        plt.show()
    else:
        print("Column 'class_label' not found for class distribution.")

def plot_split_distribution(df):
    """
    Plots the distribution of dataset splits.
    """
    if 'split' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x='split', order=df['split'].value_counts().index)
        plt.title("Distribution of Dataset Splits")
        plt.xlabel("Split")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("humaid_split_distribution.png")
        plt.show()
    else:
        print("Column 'split' not found for split distribution.")

def plot_tweet_length_distribution(df):
    """
    Generates visualizations for the distribution of tweet lengths:
      - Histogram.
      - Boxplot.
    """
    if 'tweet_length' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.hist(df['tweet_length'], bins=30, edgecolor='k', alpha=0.7)
        plt.title("Tweet Length Distribution")
        plt.xlabel("Length (number of characters)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig("humaid_tweet_length_histogram.png")
        plt.show()

        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df['tweet_length'])
        plt.title("Tweet Length Boxplot")
        plt.xlabel("Length (number of characters)")
        plt.tight_layout()
        plt.savefig("humaid_tweet_length_boxplot.png")
        plt.show()
    else:
        print("Column 'tweet_length' is not available for length analysis.")

def generate_word_cloud(df):
    """
    Generates a word cloud from tweet text (column 'tweet_text').
    The text is cleaned by removing URLs, mentions, non-alphabetic characters,
    and common English stopwords.
    """
    if 'tweet_text' not in df.columns:
        print("Column 'tweet_text' not found for generating word cloud.")
        return

    # Combine all tweet texts into a single string.
    all_text = " ".join(df['tweet_text'].dropna().astype(str))

    # Clean the text:
    cleaned_text = re.sub(r'https?://\S+', '', all_text)  # Remove URLs
    cleaned_text = re.sub(r'@\w+', '', cleaned_text)         # Remove mentions
    cleaned_text = re.sub(r'\bRT\b', '', cleaned_text)        # Remove 'RT' token
    cleaned_text = re.sub(r'[^A-Za-z\s]', '', cleaned_text)    # Keep only letters and spaces
    cleaned_text = cleaned_text.lower()                       # Convert to lowercase

    # Define custom stopwords (English) if needed.
    custom_stopwords = {"https", "http", "co", "amp", "rt"}
    all_stopwords = STOPWORDS.union(custom_stopwords)

    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          stopwords=all_stopwords).generate(cleaned_text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud (Cleaned Tweet Text)")
    plt.tight_layout()
    plt.savefig("humaid_tweet_word_cloud.png")
    plt.show()

def main():
    """
    Main function: loads, preprocesses, and generates visualizations for the dataset.
    """
    # Update filepath: file is one directory up
    filepath = "../HumAID_maria_tweets.csv"

    if not os.path.exists(filepath):
        print(f"The file '{filepath}' does not exist. Check the path.")
        return

    # Load data
    df = load_data(filepath)
    if df is None:
        return

    # Display basic dataset information
    print("Dataset Information:")
    print(df.info())
    print(df.head())

    # Preprocess data
    df = preprocess_data(df)

    # Plot distributions for class labels and splits
    plot_class_distribution(df)
    plot_split_distribution(df)

    # Plot tweet length distribution
    plot_tweet_length_distribution(df)

    # Generate word cloud from tweet text
    generate_word_cloud(df)

if __name__ == "__main__":
    main()
