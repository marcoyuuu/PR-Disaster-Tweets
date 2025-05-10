#!/usr/bin/env python
# main.py — Análisis de sentimiento desde un CSV con GPU (RTX 4070) y pipeline de clasificación

import os
import argparse
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Imports for the new classification pipeline
import logging
import time
import json
import datetime
import subprocess
import numpy as np
from tqdm import tqdm # type: ignore
from sklearn.model_selection import train_test_split, cross_validate # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.svm import SVC  # type: ignore
from sklearn.metrics import classification_report, f1_score, confusion_matrix, make_scorer, precision_score, recall_score  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from joblib import dump, load # type: ignore
from scipy.sparse import hstack, csr_matrix, issparse # For combining sparse and dense features and checking

# Attempt to import shap, but make it optional
try:
    import shap # type: ignore
except ImportError:
    shap = None


# Suprimir warnings de transformers y otros
warnings.filterwarnings("ignore")

# Setup logging
log = logging.getLogger(__name__)
# Add a basic configuration if no handlers are set
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')


# --- Helper Classes and Functions ---

class NpEncoder(json.JSONEncoder):
    """ Custom JSON encoder to handle numpy types that may appear in metrics. """
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super(NpEncoder, self).default(o)

def get_git_hash():
    """Gets the current git hash."""
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], universal_newlines=True, stderr=subprocess.DEVNULL).strip()
        return git_hash
    except Exception:
        log.warning("Could not retrieve git hash. Is this a git repository? Git CLI installed?")
        return None

# --- RoBERTa Sentiment Analysis Functions (from original main.py) ---
def load_model(model_name: str):
    """Carga el tokenizer y el modelo, y lo envía al dispositivo (GPU si está disponible)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

def analyze_sentiment(text: str, tokenizer, model, device, labels: list):
    """Analiza el texto y devuelve la etiqueta de sentimiento y la confianza."""
    if not text or pd.isna(text): # Handle empty or NaN text
        return "Neutral", 0.0  # Default for empty/NaN text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128, # Consider making this configurable
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
    # Ensure 'tweet_id' or a unique identifier exists for accurate counting
    if 'tweet_id' not in df.columns:
        log.warning("Column 'tweet_id' not found for heatmap, using index for counting if unique.")
        # Create a temporary unique ID if 'tweet_id' is missing and index is unique
        if df.index.is_unique:
            df_copy = df.copy() # Work on a copy
            df_copy['temp_id_for_pivot'] = df_copy.index
            pivot_val_col = 'temp_id_for_pivot'
        else: # Fallback if index is not unique, count occurrences (less ideal)
            log.warning("Index is not unique, heatmap might not be accurate. Counting rows.")
            # This requires a different approach or simply counting rows per group
            # For simplicity, let's assume 'class_label' and 'sentiment' are what we pivot on
            # and we count occurrences.
            # A more robust solution would be to ensure a unique ID or handle this case explicitly.
            # For now, if no tweet_id, we'll try to count, but it might be misleading if rows are not unique tweets.
            # Let's try to create a simple count if tweet_id is missing:
            df_grouped = df.groupby(['class_label', 'sentiment']).size().reset_index(name='counts')
            pivot = df_grouped.pivot_table(
                index="class_label",
                columns="sentiment",
                values="counts", 
                fill_value=0,
            )
            if 'temp_id_for_pivot' in df.columns: del df_copy['temp_id_for_pivot'] # Clean up temp col
    else:
        pivot = df.pivot_table(
            index="class_label",
            columns="sentiment",
            values="tweet_id", 
            aggfunc="count", # Count unique tweet_ids if 'tweet_id' is a unique identifier
            fill_value=0,
        )
    
    if pivot.empty:
        log.warning("Pivot table for sentiment heatmap is empty. Skipping plot.")
        return plt.figure() # Return an empty figure

    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_pct, annot=True, fmt=".1f", cmap="YlGnBu")
    plt.title("Heatmap de Porcentaje: Sentimiento RoBERTa vs Clase")
    plt.xlabel("Sentiment (RoBERTa)")
    plt.ylabel("class_label")
    plt.tight_layout()
    return plt.gcf()

def plot_confidence_by_sentiment(df):
    """Genera y retorna un boxplot de confianza por sentimiento."""
    plt.figure(figsize=(7, 5))
    df_copy = df.copy()
    # Ensure sentiment is categorical for correct ordering in boxplot if desired
    df_copy['sentiment'] = pd.Categorical(df_copy['sentiment'], categories=["Negative", "Neutral", "Positive"], ordered=True)
    
    sns.boxplot(
        x="sentiment",
        y="confidence",
        # hue="sentiment", # Not needed if x is already sentiment
        data=df_copy,
        palette="Set2",
        # legend=False, # No legend if hue is not used or same as x
    )
    plt.title("Confianza del Modelo por Sentimiento (RoBERTa)")
    plt.xlabel("Sentiment (RoBERTa)")
    plt.ylabel("Confidence (RoBERTa)")
    plt.tight_layout()
    return plt.gcf()

# --- STUBS for functions from utils.py ---
# YOU NEED TO IMPLEMENT THESE FUNCTIONS BASED ON YOUR utils.py SCRIPT

def clean_text(df: pd.DataFrame, input_text_col: str, output_cleaned_text_col: str = 'cleaned_text') -> pd.DataFrame:
    """
    Cleans the text in the specified column of the DataFrame.
    This function should add a new column, e.g., 'cleaned_text'.
    """
    log.warning(f"STUB: `clean_text` function is not fully implemented. Using original text from '{input_text_col}' as '{output_cleaned_text_col}' (lowercased).")
    if input_text_col not in df.columns:
        raise ValueError(f"Input text column '{input_text_col}' not found in DataFrame for clean_text.")
    df[output_cleaned_text_col] = df[input_text_col].astype(str).str.lower().fillna('') # Ensure no NaNs for TF-IDF
    return df

def compute_features(df: pd.DataFrame, roberta_sentiment_col: str = 'sentiment', roberta_confidence_col: str = 'confidence', text_col_for_other_meta_features: str = 'cleaned_text') -> pd.DataFrame:
    """
    Computes meta-features for the classification model.
    IMPORTANT: This function MUST use the RoBERTa 'sentiment' and 'confidence' columns.
    The RoBERTa 'sentiment' (textual) will need to be numerically encoded.
    It should add new columns for each computed feature.
    It should NOT use VADER.
    """
    log.warning("STUB: `compute_features` function is not fully implemented. Basic RoBERTa feature encoding and placeholder meta-features will be used.")
    
    # 1. Encode RoBERTa sentiment
    if roberta_sentiment_col not in df.columns:
        raise ValueError(f"RoBERTa sentiment column '{roberta_sentiment_col}' not found.")
    sentiment_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2} # Consistent mapping
    df['roberta_sentiment_encoded'] = df[roberta_sentiment_col].map(sentiment_mapping).fillna(1).astype(int) # Fill unmapped as Neutral

    # 2. Use RoBERTa confidence (already in 'confidence' column from RoBERTa analysis)
    if roberta_confidence_col not in df.columns:
        raise ValueError(f"RoBERTa confidence column '{roberta_confidence_col}' not found.")
    # Ensure it's numeric, fill NaNs if any (should be handled by RoBERTa part, but good to be safe)
    df[roberta_confidence_col] = pd.to_numeric(df[roberta_confidence_col], errors='coerce').fillna(0.0)

    # 3. Placeholder for other meta-features (derived from text_col_for_other_meta_features)
    if text_col_for_other_meta_features not in df.columns:
        log.warning(f"Text column '{text_col_for_other_meta_features}' for other meta-features not found. Skipping these features.")
        # Add placeholder columns anyway to avoid downstream errors if they are expected
        for col in ["urgency", "tweet_len", "num_exclaims", "num_questions", "num_hashtags", "num_mentions"]:
            df[col] = 0
    else:
        df['urgency'] = 0 # Placeholder: User should implement logic
        df['tweet_len'] = df[text_col_for_other_meta_features].apply(lambda x: len(str(x)))
        df['num_exclaims'] = df[text_col_for_other_meta_features].apply(lambda x: str(x).count('!'))
        df['num_questions'] = df[text_col_for_other_meta_features].apply(lambda x: str(x).count('?'))
        df['num_hashtags'] = df[text_col_for_other_meta_features].apply(lambda x: str(x).count('#'))
        df['num_mentions'] = df[text_col_for_other_meta_features].apply(lambda x: str(x).count('@'))

    # For features like Tweet_Retweets, Tweet_Likes, ensure they exist or are created
    for col in ['Tweet_Retweets', 'Tweet_Likes']:
        if col not in df.columns:
            log.warning(f"Meta column '{col}' not found in input CSV, adding as zeros for STUB compute_features.")
            df[col] = 0 
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

def extract_features(df: pd.DataFrame, max_feats: int, cleaned_text_col: str = 'cleaned_text'):
    """
    Extracts TF-IDF features from the 'cleaned_text' column.
    Returns the feature matrix (sparse) and the fitted vectorizer.
    """
    log.warning(f"STUB: `extract_features` function. Basic TF-IDF on '{cleaned_text_col}'.")
    if cleaned_text_col not in df.columns:
        raise ValueError(f"Cleaned text column '{cleaned_text_col}' not found for TF-IDF.")
    
    vectorizer = TfidfVectorizer(max_features=max_feats, stop_words='english', min_df=2) # Added min_df
    
    texts_for_tfidf = df[cleaned_text_col].fillna('').astype(str)
    
    if texts_for_tfidf.empty or texts_for_tfidf.str.strip().eq('').all():
        log.warning(f"All texts in '{cleaned_text_col}' are empty or whitespace. TF-IDF will produce empty features.")
        return csr_matrix((len(df), 0)), vectorizer # Return empty sparse matrix

    try:
        X_text = vectorizer.fit_transform(texts_for_tfidf)
    except ValueError as e:
        log.error(f"TF-IDF fitting failed: {e}. All documents might be empty or only stop words. Returning empty features.")
        return csr_matrix((len(df), 0)), vectorizer
        
    if X_text.shape[1] == 0:
        log.warning("TF-IDF resulted in 0 features. Check text data and TF-IDF parameters (max_feats, min_df, vocabulary).")
    return X_text, vectorizer

# --- New Classification Pipeline Functions ---

def plot_metrics_and_confusion(metrics, results_dir, y_true=None, y_pred=None, label_mapping=None, class_names_ordered=None):
    """
    Simplified placeholder for plotting metrics and confusion matrix.
    User should replace this with their detailed plotting function.
    """
    log.info(f"Executing STUB plot_metrics_and_confusion. Metrics received: {list(metrics.keys()) if metrics else 'None'}")
    os.makedirs(results_dir, exist_ok=True)

    if 'cv_results' in metrics:
        log.info("STUB: Plotting CV results (bar chart).")
        # Example: Create a simple bar chart for CV F1 macro
        cv_f1_mean = metrics['cv_results'].get('test_f1_macro_mean', 0)
        cv_f1_std = metrics['cv_results'].get('test_f1_macro_std', 0)
        plt.figure(figsize=(6,4))
        plt.bar(['F1 Macro (CV)'], [cv_f1_mean], yerr=[cv_f1_std], capsize=5)
        plt.title('CV F1 Macro (Stub Plot)')
        plt.ylabel('Score')
        plt.ylim(0,1.1)
        plt.savefig(os.path.join(results_dir, "stub_cv_metrics_bar.png"))
        plt.close()

    elif 'f1_macro' in metrics and y_true is not None and y_pred is not None:
        log.info("STUB: Plotting train/test metrics (bar chart) and confusion matrix.")
        # Example: Simple bar chart for F1 macro
        f1_macro = metrics.get('f1_macro', 0)
        plt.figure(figsize=(6,4))
        plt.bar(['F1 Macro (Test)'], [f1_macro])
        plt.title('Test F1 Macro (Stub Plot)')
        plt.ylabel('Score')
        plt.ylim(0,1.05)
        plt.savefig(os.path.join(results_dir, "stub_test_metrics_bar.png"))
        plt.close()

        # Example: Simple confusion matrix
        if label_mapping and class_names_ordered:
            # Ensure labels for confusion_matrix are the numeric values
            numeric_labels = [label_mapping[name] for name in class_names_ordered if name in label_mapping]
            cm = confusion_matrix(y_true, y_pred, labels=numeric_labels)
            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=class_names_ordered, yticklabels=class_names_ordered)
            plt.title("Confusion Matrix (Stub Plot)")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "stub_confusion_matrix.png"))
            plt.close()
        else:
            log.warning("Stub plot_metrics_and_confusion: label_mapping or class_names_ordered missing for CM.")
    else:
        log.info("Stub plot_metrics_and_confusion: Not enough data to plot (e.g. no CV results or no test predictions).")

    log.info(f"STUB plot_metrics_and_confusion finished. Plots (if any) saved in {results_dir}")


def add_shap_explainability(model, X_test_np, results_dir, feature_names_list=None, class_names_list=None):
    """ Generate SHAP explanations for the model predictions. """
    if shap is None:
        log.warning("SHAP library not installed. Skipping SHAP plots. Install with: pip install shap")
        return
    try:
        log.info("Generating SHAP explanations...")
        os.makedirs(results_dir, exist_ok=True)

        if not isinstance(X_test_np, (np.ndarray, pd.DataFrame)):
             # SHAP TreeExplainer expects numpy array or pandas DataFrame.
             # If X_test_np is sparse, convert to dense.
            if hasattr(X_test_np, "toarray"): # Check if it's a sparse matrix
                log.info("Converting sparse X_test to dense for SHAP TreeExplainer.")
                X_test_np = X_test_np.toarray()
            else: # If not sparse and not ndarray/DataFrame, this is an issue.
                log.error(f"X_test_np for SHAP is of unexpected type: {type(X_test_np)}. SHAP may fail.")
                # Attempt to convert to numpy array if possible, as a last resort
                try:
                    X_test_np = np.array(X_test_np)
                except Exception as e_conv:
                    log.error(f"Could not convert X_test_np to numpy array for SHAP: {e_conv}")
                    return


        # For TreeExplainer, X_test_np (numpy array) is fine.
        # If feature_names_list is provided, create a DataFrame for prettier plots.
        x_display = X_test_np
        if feature_names_list:
            if X_test_np.shape[1] == len(feature_names_list):
                x_display = pd.DataFrame(X_test_np, columns=feature_names_list)
            else:
                log.warning(f"SHAP: Mismatch between X_test_np columns ({X_test_np.shape[1]}) and feature_names_list length ({len(feature_names_list)}). Using generic feature names.")
        else:
            log.warning("No feature names provided to SHAP, plots may use generic names.")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_np) # For RF, this is often a list of arrays for multi-class

        dump(shap_values, os.path.join(results_dir, "shap_values.joblib"))
        if hasattr(explainer, 'expected_value'):
             dump(explainer.expected_value, os.path.join(results_dir, "shap_expected_value.joblib"))
        log.info(f"SHAP values and expected_value (if available) saved to {results_dir}")

        if isinstance(shap_values, list) and len(shap_values) > 0: # Multi-class classification
            num_classes_shap = len(shap_values)
            for i in range(num_classes_shap):
                class_title_name = class_names_list[i] if class_names_list and i < len(class_names_list) else f"Class_{i}"
                plt.figure()
                # Use x_display for plotting, which might be a DataFrame with feature names
                shap.summary_plot(shap_values[i], x_display, show=False, plot_type="bar")
                plt.title(f"SHAP Summary Plot - {class_title_name}")
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f"shap_summary_{class_title_name.replace(' ', '_')}.png"), bbox_inches='tight')
                plt.close()
        else: # Binary classification or single output from shap_values
            plt.figure()
            shap.summary_plot(shap_values, x_display, show=False, plot_type="bar") # Default plot
            plt.title("SHAP Summary Plot")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "shap_summary.png"), bbox_inches='tight')
            plt.close()
        log.info(f"SHAP summary plots saved to {results_dir}")

    except Exception as e:
        log.error(f"SHAP explainability failed: {e}", exc_info=True)


def train_and_evaluate_model(X, y, model_type, results_dir, feature_names_list=None, class_names_list=None, label_mapping_new_task=None, explain=False, cv_folds=0):
    if model_type == "rf":
        model = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    elif model_type == "svm":
        model = SVC(class_weight="balanced", probability=True, random_state=42) # probability=True for SHAP if KernelExplainer was used
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    metrics = {}
    os.makedirs(results_dir, exist_ok=True)

    if cv_folds > 1:
        log.info(f"Performing {cv_folds}-fold cross-validation for model '{model_type}'...")
        scoring = {
            'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
            'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
            'f1_macro': make_scorer(f1_score, average='macro', zero_division=0)
        }
        start_cv = time.time()
        # Ensure X is suitable for sklearn (e.g. numpy array or sparse matrix, not list of lists)
        cv_results = cross_validate(model, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1, error_score='raise')
        cv_time = time.time() - start_cv
        
        metrics['cv_results'] = {
            'fit_time_mean': float(np.mean(cv_results['fit_time'])),
            'score_time_mean': float(np.mean(cv_results['score_time'])),
            'test_precision_macro_mean': float(np.mean(cv_results['test_precision_macro'])),
            'test_precision_macro_std': float(np.std(cv_results['test_precision_macro'])),
            'test_recall_macro_mean': float(np.mean(cv_results['test_recall_macro'])),
            'test_recall_macro_std': float(np.std(cv_results['test_recall_macro'])),
            'test_f1_macro_mean': float(np.mean(cv_results['test_f1_macro'])),
            'test_f1_macro_std': float(np.std(cv_results['test_f1_macro'])),
        }
        metrics['train_time_cv_total_seconds'] = cv_time
        log.info(f"Cross-validation finished in {cv_time:.2f}s. Mean F1 Macro: {metrics['cv_results']['test_f1_macro_mean']:.4f}")
        plot_metrics_and_confusion(metrics, results_dir) # Only CV bar plot
        return metrics, None # Return metrics and no model for CV

    # Standard train/test split evaluation (if cv_folds <= 1)
    log.info(f"Performing train/test split evaluation for model '{model_type}'...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)
    
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    
    preds = model.predict(X_test)
    
    # Calculate f1_micro separately for accuracy fallback
    f1_micro_val = float(f1_score(y_test, preds, average='micro', zero_division=0))
    report_dict = classification_report(y_test, preds, output_dict=True, zero_division=0, target_names=class_names_list)
    
    metrics = {
        "f1_macro": float(f1_score(y_test, preds, average='macro', zero_division=0)),
        "f1_micro": f1_micro_val, # Same as accuracy for multi-class if calculated this way
        "f1_weighted": float(f1_score(y_test, preds, average='weighted', zero_division=0)),
        "accuracy": report_dict.get("accuracy", f1_micro_val), # Use f1_micro_val as fallback
        "classification_report": report_dict,
        "train_time_seconds": train_time,
        "test_set_size": len(y_test)
    }

    pd.DataFrame({"y_true": y_test, "y_pred": preds}).to_csv(os.path.join(results_dir, f"{model_type}_predictions.csv"), index=False)
    model_path = os.path.join(results_dir, f"{model_type}_final_model.joblib")
    dump(model, model_path)
    log.info(f"Trained model saved to {model_path}")

    plot_metrics_and_confusion(metrics, results_dir, y_true=y_test, y_pred=preds, label_mapping=label_mapping_new_task, class_names_ordered=class_names_list)

    if explain and model_type == "rf": # SHAP only for RF for now
        add_shap_explainability(model, X_test, results_dir, feature_names_list=feature_names_list, class_names_list=class_names_list)
            
    return metrics, model


def train_and_evaluate_with_predefined_splits(X, y, split_indices, model_type, results_dir, feature_names_list=None, class_names_list=None, label_mapping_new_task=None, explain=False):
    if model_type == "rf":
        model = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    elif model_type == "svm":
        model = SVC(class_weight="balanced", probability=True, random_state=42)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    metrics = {}
    os.makedirs(results_dir, exist_ok=True)
    
    # Validate indices
    max_index = X.shape[0] -1 if hasattr(X, "shape") else len(X) -1 # Works for numpy/sparse and lists
    
    train_idx = [i for i in split_indices['train'] if i <= max_index]
    test_idx = [i for i in split_indices['test'] if i <= max_index]
    
    if not train_idx or not test_idx:
        raise ValueError("Train or Test indices are empty after validation or out of bounds.")

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    log.info(f"Training with predefined splits for model '{model_type}': {len(X_train)} train, {len(X_test)} test samples")
    
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    
    preds = model.predict(X_test)
    
    f1_micro_val = float(f1_score(y_test, preds, average='micro', zero_division=0))
    report_dict = classification_report(y_test, preds, output_dict=True, zero_division=0, target_names=class_names_list)
    
    metrics = {
        "f1_macro": float(f1_score(y_test, preds, average='macro', zero_division=0)),
        "f1_micro": f1_micro_val,
        "f1_weighted": float(f1_score(y_test, preds, average='weighted', zero_division=0)),
        "accuracy": report_dict.get("accuracy", f1_micro_val),
        "classification_report": report_dict,
        "train_time_seconds": train_time,
        "split_info": {
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "dev_size": len(split_indices.get('dev', [])), # Assuming 'dev' might be in split_indices
        }
    }

    pd.DataFrame({"y_true": y_test, "y_pred": preds}).to_csv(os.path.join(results_dir, f"{model_type}_predictions_predefined_split.csv"), index=False)
    model_path = os.path.join(results_dir, f"{model_type}_final_model_predefined_split.joblib")
    dump(model, model_path)
    log.info(f"Trained model from predefined split saved to {model_path}")

    plot_metrics_and_confusion(metrics, results_dir, y_true=y_test, y_pred=preds, label_mapping=label_mapping_new_task, class_names_ordered=class_names_list)

    if explain and model_type == "rf":
        add_shap_explainability(model, X_test, results_dir, feature_names_list=feature_names_list, class_names_list=class_names_list)
    
    return metrics, model


# --- Main Execution ---
def main():
    """Ejecuta el análisis de sentimiento RoBERTa y luego un pipeline de clasificación secundario."""
    # 1) Parámetros del modelo RoBERTa
    roberta_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    roberta_labels = ["Negative", "Neutral", "Positive"]

    # 2) Argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description="Pipeline de análisis de sentimiento (RoBERTa) y clasificación de tweets (RF/SVM)."
    )
    parser.add_argument(
        "--input_csv", type=str, default=os.path.join("data", "HumAID_maria_tweets_clean.csv"),
        help="Ruta al archivo CSV de entrada para el análisis de RoBERTa."
    )
    parser.add_argument(
        "--roberta_output_csv", type=str, default=os.path.join("results", "roberta_sentiment_results.csv"),
        help="Ruta al archivo CSV de salida para los resultados de RoBERTa."
    )
    parser.add_argument(
        "--text_col", type=str, default=None,
        help="Nombre de la columna de texto en el CSV (opcional, se auto-detectará)."
    )
    parser.add_argument(
        '--results_dir', type=str, default='results/classification_pipeline',
        help='Directorio para guardar todos los resultados de la pipeline de clasificación.'
    )
    parser.add_argument('--max_feats', type=int, default=1000, help='Max features para TF-IDF.')
    parser.add_argument('--model_type', type=str, default='rf', choices=['rf', 'svm'], help='Tipo de modelo para clasificación secundaria.')
    parser.add_argument('--explain', action='store_true', help='Habilitar explicabilidad SHAP (si es RF).')
    parser.add_argument('--cv', type=int, default=0, help='Folds para cross-validation (0/1 para train/test split).')
    parser.add_argument('--nan_threshold_label', type=float, default=0.5, help='Umbral de NaNs en etiquetas para error.')
    parser.add_argument('--use_predefined_splits', action='store_true', help="Usar columna 'split' si existe.")
    parser.add_argument('--force_random_split', action='store_true', help="Forzar split aleatorio.")
    parser.add_argument('--class_label_col', type=str, default='class_label', help="Columna de etiquetas originales.")
    parser.add_argument('--target_label_col', type=str, default='target_label_secondary_task', help="Columna para etiquetas numéricas mapeadas.")
    parser.add_argument('--cleaned_text_col_name', type=str, default='cleaned_text_for_tfidf', help="Columna para texto limpiado para TF-IDF.")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.roberta_output_csv), exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    log.info(f"Argumentos: {args}")

    # --- Parte 1: Análisis de Sentimiento con RoBERTa ---
    log.info("--- Iniciando Parte 1: Análisis de Sentimiento con RoBERTa ---")
    roberta_tokenizer, roberta_model, roberta_device = load_model(roberta_model_name)
    log.info(f"Modelo RoBERTa '{roberta_model_name}' cargado en {roberta_device}.")

    try:
        df = pd.read_csv(args.input_csv)
        log.info(f"CSV '{args.input_csv}' cargado. Shape: {df.shape}")
    except FileNotFoundError:
        log.error(f"Archivo CSV no encontrado: {args.input_csv}"); return
    except Exception as e:
        log.error(f"Error al cargar CSV '{args.input_csv}': {e}"); return

    possible_text_cols_roberta = ('clean_text', 'tweet', 'text', 'Tweet', 'Text', 'full_text')
    text_col_roberta = args.text_col if args.text_col and args.text_col in df.columns else \
                       next((col for col in possible_text_cols_roberta if col in df.columns), None)
    
    if text_col_roberta is None:
        log.error(f"No se encontró columna de texto para RoBERTa (buscadas: {possible_text_cols_roberta}). Especifique con --text_col."); return
    log.info(f"Columna de texto para RoBERTa: '{text_col_roberta}'")

    sentiments, confidences = [], []
    log.info(f"Iniciando análisis RoBERTa en {len(df)} tweets de '{text_col_roberta}'...")
    for txt in tqdm(df[text_col_roberta].astype(str), desc="Análisis RoBERTa"): # Ensure string type
        s, c = analyze_sentiment(txt, roberta_tokenizer, roberta_model, roberta_device, roberta_labels)
        sentiments.append(s); confidences.append(c)

    df["sentiment"] = sentiments  # RoBERTa sentiment
    df["confidence"] = confidences # RoBERTa confidence
    df.to_csv(args.roberta_output_csv, index=False)
    log.info(f"✅ Análisis RoBERTa completado. Resultados en: {args.roberta_output_csv}")

    roberta_plots_dir = os.path.join(args.results_dir, "roberta_visualizations")
    os.makedirs(roberta_plots_dir, exist_ok=True)
    base_name_roberta = os.path.splitext(os.path.basename(args.input_csv))[0]

    if args.class_label_col in df.columns and 'tweet_id' in df.columns: # tweet_id for pivot
        try:
            fig1 = plot_sentiment_class_heatmap(df.copy()) # Pass copy
            fig1.savefig(os.path.join(roberta_plots_dir, f"roberta_heatmap_{base_name_roberta}.png"), bbox_inches="tight"); plt.close(fig1)
        except Exception as e: log.warning(f"Error heatmap RoBERTa: {e}")
    else: log.info(f"Saltando heatmap RoBERTa (requiere '{args.class_label_col}' y 'tweet_id').")
    try:
        fig2 = plot_confidence_by_sentiment(df.copy()) # Pass copy
        fig2.savefig(os.path.join(roberta_plots_dir, f"roberta_boxplot_confidence_{base_name_roberta}.png"), bbox_inches="tight"); plt.close(fig2)
    except Exception as e: log.warning(f"Error boxplot RoBERTa: {e}")
    log.info("--- Fin Parte 1 ---")

    # --- Parte 2: Pipeline de Clasificación ---
    log.info("--- Iniciando Parte 2: Pipeline de Clasificación ---")
    classification_label_mapping = {
        "infrastructure_and_utility_damage": "informative", "injured_or_dead_people": "informative",
        "requests_or_urgent_needs": "informative", "rescue_volunteering_or_donation_effort": "informative",
        "sympathy_and_support": "informative", "other_relevant_information": "neutral",
        "not_humanitarian": "non_informative"
    }
    target_numerical_mapping = {"informative": 0, "neutral": 1, "non_informative": 2}
    class_names_for_plots = [name for name, _ in sorted(target_numerical_mapping.items(), key=lambda item: item[1])]

    if args.class_label_col not in df.columns:
        log.error(f"Columna '{args.class_label_col}' no encontrada. No se puede continuar Parte 2."); return
    
    df[args.target_label_col + "_str"] = df[args.class_label_col].map(classification_label_mapping)
    nan_prop = df[args.target_label_col + "_str"].isna().mean()
    if nan_prop > args.nan_threshold_label:
        log.error(f"NaNs en etiquetas ({nan_prop:.2%}) > umbral. Verifique mapeo."); return
    df = df.dropna(subset=[args.target_label_col + "_str"]).reset_index(drop=True)
    if df.empty: log.error("No quedan datos después de mapeo/NaN drop."); return
    
    y = df[args.target_label_col + "_str"].map(target_numerical_mapping).to_numpy(dtype=int)

    df = clean_text(df, input_text_col=text_col_roberta, output_cleaned_text_col=args.cleaned_text_col_name)
    df = compute_features(df, roberta_sentiment_col="sentiment", roberta_confidence_col="confidence", 
                          text_col_for_other_meta_features=args.cleaned_text_col_name) # Use cleaned text for meta
    X_text, vectorizer = extract_features(df, args.max_feats, cleaned_text_col=args.cleaned_text_col_name)
    dump(vectorizer, os.path.join(args.results_dir, f"{args.model_type}_tfidf_vectorizer.joblib"))

    meta_cols = ['roberta_sentiment_encoded', 'confidence', 'urgency', 'tweet_len', 
                 'num_exclaims', 'num_questions', 'num_hashtags', 'num_mentions', 
                 'Tweet_Retweets', 'Tweet_Likes']
    actual_meta_cols = [col for col in meta_cols if col in df.columns]
    X_meta = df[actual_meta_cols].values if actual_meta_cols else np.empty((len(df), 0))
    X_meta = np.nan_to_num(X_meta.astype(float), nan=0.0)

    log.info(f"Shape X_text: {X_text.shape}, Shape X_meta: {X_meta.shape}")

    if X_text.shape[0] != len(df) or X_meta.shape[0] != len(df):
        log.error("Desajuste de filas entre features y DataFrame. Abortando."); return

    # Combine features
    if X_text.shape[1] > 0 and X_meta.shape[1] > 0:
        X_combined = hstack([X_text, csr_matrix(X_meta)]).tocsr() # Ensure X_meta is sparse for hstack with sparse X_text
    elif X_text.shape[1] > 0: X_combined = X_text
    elif X_meta.shape[1] > 0: X_combined = csr_matrix(X_meta)
    else: log.error("Ambas matrices de features están vacías. No se puede entrenar."); return
    
    log.info(f"Shape X_combined: {X_combined.shape}")
    if X_combined is None or X_combined.shape[0] == 0: # Check if X_combined is valid
        log.error("X_combined es None o vacío. No se puede entrenar."); return

    feature_names_combined = []
    if X_text.shape[1] > 0 and hasattr(vectorizer, 'get_feature_names_out'):
        try: feature_names_combined.extend(vectorizer.get_feature_names_out().tolist())
        except: feature_names_combined.extend([f"tfidf_{i}" for i in range(X_text.shape[1])])
    if X_meta.shape[1] > 0: feature_names_combined.extend(actual_meta_cols)

    if X_combined.shape[0] != len(y):
        log.error(f"Desajuste X_combined ({X_combined.shape[0]}) e y ({len(y)}). Abortando."); return

    # Entrenamiento y Evaluación
    model_metrics, trained_model = None, None
    cv_to_use = args.cv # Initialize cv_to_use
    use_predefined = False

    if args.force_random_split:
        log.info("Forzando train/test split aleatorio.")
        cv_to_use = 0 
    elif args.use_predefined_splits and 'split' in df.columns and df['split'].isin(['train', 'test']).any():
        use_predefined = True
    
    if use_predefined:
        train_idx = df.index[df['split'] == 'train'].tolist()
        test_idx = df.index[df['split'] == 'test'].tolist()
        if not train_idx or not test_idx:
            log.warning("Splits predefinidos incompletos. Cayendo a CV/random split.")
            use_predefined = False # Fallback
        else:
            split_indices = {'train': train_idx, 'test': test_idx, 'dev': df.index[df['split'] == 'dev'].tolist()}
            model_metrics, trained_model = train_and_evaluate_with_predefined_splits(
                X_combined, y, split_indices, args.model_type, args.results_dir,
                feature_names_list=feature_names_combined, class_names_list=class_names_for_plots,
                label_mapping_new_task=target_numerical_mapping, explain=args.explain
            )
    if not use_predefined: # Handles fallback or default case
        model_metrics, trained_model = train_and_evaluate_model(
            X_combined, y, args.model_type, args.results_dir,
            feature_names_list=feature_names_combined, class_names_list=class_names_for_plots,
            label_mapping_new_task=target_numerical_mapping, explain=args.explain, cv_folds=cv_to_use
        )

    if model_metrics:
        model_metrics['timestamp'] = datetime.datetime.now().isoformat()
        model_metrics['git_hash'] = get_git_hash()
        # ... (add other relevant args to metrics)
        metrics_fname = f"metrics_{args.model_type}{'_predef' if use_predefined else ('_'+str(cv_to_use)+'cv' if cv_to_use > 1 else '_traintest')}.json"
        with open(os.path.join(args.results_dir, metrics_fname), "w", encoding="utf-8") as f:
            json.dump(model_metrics, f, indent=2, cls=NpEncoder)
        log.info(f"Métricas guardadas en {os.path.join(args.results_dir, metrics_fname)}")
    
    log.info("--- Fin Parte 2 ---")
    log.info("✅ Pipeline completado.")

if __name__ == "__main__":
    main()
