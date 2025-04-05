#!/usr/bin/env python
"""
Module: preprocess_text.py

This module provides functionality to preprocess tweet text data from CSV files.
It applies standard NLP preprocessing steps to prepare the text for semantic analysis.
The preprocessing steps include:
    - Removal of URLs
    - Removal of Twitter mentions and the 'RT' token
    - Removal of non-alphabetic characters (keeping accented letters)
    - Conversion to lowercase
    - Language detection (English or Spanish)
    - Tokenization using the appropriate language
    - Stopwords removal using language-specific stopword lists
    - (Optional) Stemming (using NLTK WordNetLemmatizer for English and SnowballStemmer for Spanish)

The module scans through subdirectories under a given root (e.g., "datasets") and processes
each CSV file found. The cleaned CSV is saved in a subdirectory "clean" within the same dataset directory.

Dependencies:
    - pandas
    - nltk
    - langdetect

Usage:
    Run the module directly to process all CSV files under the specified datasets root.
    Example:
        python preprocess_text.py

Author: [Your Name]
Date: [Current Date]
"""

import os
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from langdetect import detect, DetectorFactory

# Set seed for langdetect for reproducibility
DetectorFactory.seed = 0

# Ensure required NLTK resources are available
def ensure_nltk_resources():
    required = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4"
    }
    for resource, path in required.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource)

ensure_nltk_resources()

# Define possible column names that may contain tweet text
POSSIBLE_TEXT_COLUMNS = ['tweet_text', 'text', 'Tweet_Content']

# Initialize the WordNet lemmatizer (for English)
LEMMATIZER = WordNetLemmatizer()
# Initialize SnowballStemmer for Spanish
STEMMER_ES = SnowballStemmer("spanish")
# For English, if desired you may also use a stemmer, but here we use lemmatization.
# Alternatively, you could initialize a SnowballStemmer for English as well:
# STEMMER_EN = SnowballStemmer("english")

def clean_text(text, apply_lemmatization=False):
    """
    Cleans a given text string by applying the following steps:
      1. Remove URLs.
      2. Remove Twitter mentions.
      3. Remove the 'RT' token.
      4. Remove non-alphabetic characters (keeping accented letters).
      5. Convert text to lowercase.
      6. Detect language (English or Spanish).
      7. Tokenize the text with the appropriate language.
      8. Remove stopwords using language-specific stopword lists.
      9. (Optional) Apply stemming/lemmatization:
            - For English: uses NLTK WordNetLemmatizer.
            - For Spanish: uses SnowballStemmer.
      
    Args:
        text (str): Original text.
        apply_lemmatization (bool): Whether to apply stemming/lemmatization.
        
    Returns:
        str: Cleaned text as a single string of tokens separated by space.
    """
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove Twitter mentions
    text = re.sub(r'@\w+', '', text)
    # Remove 'RT' token
    text = re.sub(r'\bRT\b', '', text)
    # Remove non-alphabetic characters (keep spaces, allow accented letters)
    text = re.sub(r'[^A-Za-záéíóúñüÁÉÍÓÚÑÜ\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    
    # Detect language; default to English if detection fails
    try:
        detected_lang = detect(text)
    except Exception:
        detected_lang = 'en'
    lang = 'spanish' if detected_lang == 'es' else 'english'
    
    # Tokenize the text using the detected language
    try:
        tokens = word_tokenize(text, language=lang)
    except Exception:
        tokens = text.split()  # fallback
    
    # Load stopwords for the detected language
    if lang == 'spanish':
        stop_words = set(stopwords.words('spanish'))
    else:
        stop_words = set(stopwords.words('english'))
    
    # Add custom stopwords
    stop_words.update({"rt", "http", "https", "co", "amp"})
    
    # Filter out stopwords
    tokens = [token for token in tokens if token not in stop_words]
    
    # Optional: Apply stemming/lemmatization if enabled
    if apply_lemmatization:
        if lang == 'english':
            tokens = [LEMMATIZER.lemmatize(token) for token in tokens]
        elif lang == 'spanish':
            tokens = [STEMMER_ES.stem(token) for token in tokens]
    
    return " ".join(tokens)

def preprocess_csv_file(file_path, text_column=None, apply_lemmatization=False):
    """
    Processes a CSV file by cleaning the tweet text and saving the result in a 'clean' subdirectory.
    The function tries to detect the appropriate text column if none is provided.
    
    Args:
        file_path (str): Full path to the CSV file.
        text_column (str): Column name containing the tweet text. If None, the function
                           will search in POSSIBLE_TEXT_COLUMNS.
        apply_lemmatization (bool): Whether to apply stemming/lemmatization during cleaning.
        
    Returns:
        None
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Identify the text column if not explicitly provided
    if text_column is None:
        for col in POSSIBLE_TEXT_COLUMNS:
            if col in df.columns:
                text_column = col
                break
    if text_column is None:
        print(f"No text column found in {file_path}. Skipping file.")
        return
    
    # Create a new column 'clean_text' by cleaning the original text
    df['clean_text'] = df[text_column].apply(lambda x: clean_text(x, apply_lemmatization=apply_lemmatization))
    
    # Prepare output directory (same folder as the file, within a subdirectory 'clean')
    file_dir = os.path.dirname(file_path)
    output_dir = os.path.join(file_dir, 'clean')
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the output file name with a suffix _clean.csv
    base_name = os.path.basename(file_path)
    output_file = os.path.join(output_dir, base_name.replace('.csv', '_clean.csv'))
    
    try:
        df.to_csv(output_file, index=False)
        print(f"Processed file saved to {output_file}")
    except Exception as e:
        print(f"Error writing {output_file}: {e}")

def process_all_datasets(root_directory, apply_lemmatization=False):
    """
    Processes all CSV files within subdirectories of the given root directory.
    For each CSV found in each dataset folder (non-recursively), the function applies
    text preprocessing and saves the cleaned CSV in a 'clean' subfolder within that dataset directory.
    
    Args:
        root_directory (str): Root directory containing dataset subdirectories.
        apply_lemmatization (bool): Whether to apply stemming/lemmatization.
        
    Returns:
        None
    """
    for entry in os.scandir(root_directory):
        if entry.is_dir():
            dataset_dir = entry.path
            print(f"Processing dataset directory: {dataset_dir}")
            # Process CSV files in the dataset directory (ignoring subfolders like analysis/ or clean/)
            for file_entry in os.scandir(dataset_dir):
                if file_entry.is_file() and file_entry.name.lower().endswith('.csv'):
                    file_path = file_entry.path
                    print(f"  Processing file: {file_path}")
                    preprocess_csv_file(file_path, apply_lemmatization=apply_lemmatization)
    print("All datasets processed.")

if __name__ == "__main__":
    # Define the root directory containing your datasets (adjust if needed)
    datasets_root = "datasets"
    # Process all CSV files in the datasets directory, applying stemming/lemmatization if desired
    process_all_datasets(datasets_root, apply_lemmatization=True)
