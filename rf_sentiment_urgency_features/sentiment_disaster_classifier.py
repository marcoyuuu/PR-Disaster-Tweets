
# =============================================
# Clasificador de Sentimientos y Desastres en Tweets
# Descripción: Script profesional para clasificación de tweets de desastres
#              con modelos Random Forest, SVM y Regresión Logística (BoW).
# =============================================

import os
import pandas as pd  # type: ignore
import numpy as np
import logging
import time
import json
import argparse
import datetime
import subprocess
import matplotlib.pyplot as plt  # type: ignore
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
from sklearn.model_selection import train_test_split, cross_validate  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.svm import SVC  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
from sklearn.metrics import (classification_report, f1_score, confusion_matrix,
                             make_scorer, precision_score, recall_score)  # type: ignore
from joblib import dump

# Utilidades propias
from utils import extract_features, compute_features

# Configuración de logging
log = logging.getLogger(__name__)


def load_data(data_file):
    """
    Carga un archivo CSV y valida que no esté vacío.
    Args:
        data_file (str): Ruta al archivo CSV.
    Returns:
        pd.DataFrame: DataFrame cargado.
    """
    log.info(f"Cargando archivo: {data_file}")
    df = pd.read_csv(data_file)
    if df.empty:
        log.error(f"No se cargaron datos desde {data_file}")
        raise RuntimeError(f"No se cargaron datos desde {data_file}")
    return df

def plot_metrics_and_confusion(metrics, results_dir, y_true=None, y_pred=None, label_mapping=None):
    """
    Genera y guarda visualizaciones profesionales de métricas y matriz de confusión (PNG, normalizada y absoluta).
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns
    import numpy as np
    import os
    
    # Establecer estilo general para todas las visualizaciones
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.3)
    
    # Paletas de colores personalizadas para mayor profesionalismo
    palette_viridis = sns.color_palette("viridis", 3)
    palette_cool = sns.color_palette("cool", 3)
    palette_bright = sns.color_palette("bright", 3)
    
    # 1. Métricas de clasificación (bar plot)
    if 'cv_results' in metrics:
        # Cross-validation: mostrar medias y std
        cv = metrics['cv_results']
        labels = ['F1 Macro', 'Precisión Macro', 'Recall Macro']
        means = [cv['test_f1_macro_mean'], cv['test_precision_macro_mean'], cv['test_recall_macro_mean']]
        stds = [cv['test_f1_macro_std'], cv['test_precision_macro_std'], cv['test_recall_macro_std']]
        
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)
        bars = ax.bar(labels, means, yerr=stds, capsize=8, color=palette_viridis, 
                     edgecolor='black', linewidth=1.2, alpha=0.8)
        
        # Estilo refinado
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Puntuación', fontweight='bold')
        ax.set_title('Métricas de Validación Cruzada (media ± desviación estándar)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Añadir valores en las barras con sombra para mejorar legibilidad
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.03, f"{mean:.3f}±{std:.3f}", ha='center', 
                   fontsize=12, fontweight='bold', 
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        
        # Añadir líneas de cuadrícula personalizadas para mayor claridad
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Añadir borde al gráfico para un aspecto profesional
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            
        fig.tight_layout()
        plt.savefig(os.path.join(results_dir, "cv_metrics_bar.png"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # Train/test split: mostrar F1 macro, F1 weighted, Accuracy
        f1_macro = metrics.get('f1_macro', None)
        f1_weighted = metrics.get('f1_weighted', None)
        f1_micro = metrics.get('f1_micro', None)
        
        if f1_macro is not None and f1_weighted is not None and f1_micro is not None:
            # Crear figura con dos subplots: métricas globales y F1 por tipo
            fig = plt.figure(figsize=(16, 10))
            gs = GridSpec(1, 2, width_ratios=[1, 1.2], figure=fig)
            
            # 1. Métricas globales (izquierda)
            ax1 = fig.add_subplot(gs[0])
            labels = ['F1 Macro', 'F1 Weighted', 'Accuracy']
            values = [f1_macro, f1_weighted, f1_micro]
            
            bars = ax1.bar(labels, values, color=palette_cool, 
                         edgecolor='black', linewidth=1.2, alpha=0.8)
            
            # Estilo refinado
            ax1.set_ylim(0, 1.05)
            ax1.set_ylabel('Puntuación', fontweight='bold')
            ax1.set_title('Métricas Globales de Clasificación', 
                        fontsize=16, fontweight='bold', pad=20)
            
            # Añadir valores en las barras con sombra para mejorar legibilidad
            for i, v in enumerate(values):
                ax1.text(i, v + 0.02, f"{v:.3f}", ha='center', 
                       fontsize=12, fontweight='bold',
                       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
            
            # Añadir líneas de cuadrícula personalizadas para mayor claridad
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            ax1.set_axisbelow(True)
            
            # Destacar F1-macro como métrica clave para datasets desbalanceados
            bars[0].set_color('darkred')
            bars[0].set_alpha(1.0)
            bars[0].set_edgecolor('black')
            bars[0].set_linewidth(2.0)
            
            # 2. Métricas por clase (derecha) - si hay classification_report disponible
            if 'classification_report' in metrics:
                ax2 = fig.add_subplot(gs[1])
                report = metrics['classification_report']
                
                classes = []
                precision_vals = []
                recall_vals = []
                f1_vals = []
                
                # Extraer métricas por clase
                for class_name, class_metrics in report.items():
                    if isinstance(class_metrics, dict) and 'precision' in class_metrics:
                        if class_name in label_mapping.values():
                            # Convertir etiquetas numéricas a texto
                            for name, val in label_mapping.items():
                                if val == int(class_name):
                                    classes.append(name)
                                    break
                        else:
                            classes.append(class_name)
                            
                        precision_vals.append(class_metrics['precision'])
                        recall_vals.append(class_metrics['recall'])
                        f1_vals.append(class_metrics['f1-score'])
                
                # Crear DataFrame para visualización
                import pandas as pd
                class_metrics_df = pd.DataFrame({
                    'Precisión': precision_vals,
                    'Recall': recall_vals,
                    'F1-Score': f1_vals
                }, index=classes)
                
                # Generar gráfico de barras agrupadas
                class_metrics_df.plot(kind='bar', ax=ax2, width=0.8)
                ax2.set_ylim(0, 1.05)
                ax2.set_title('Métricas por Clase', fontsize=16, fontweight='bold', pad=20)
                ax2.set_ylabel('Puntuación', fontweight='bold')
                ax2.set_xlabel('Clase', fontweight='bold')
                
                # Añadir leyenda en posición óptima
                ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                          ncol=3, frameon=True, shadow=True)
                
                # Añadir líneas de cuadrícula personalizadas para mayor claridad
                ax2.grid(axis='y', linestyle='--', alpha=0.7)
                ax2.set_axisbelow(True)
                
                # Guardar figura adicional con solo métricas por clase (más grande)
                plt.figure(figsize=(12, 8))
                class_metrics_df.plot(kind='bar', width=0.8)
                plt.ylim(0, 1.05)
                plt.title('Precisión, Recall y F1-Score por Clase', 
                         fontsize=18, fontweight='bold', pad=20)
                plt.ylabel('Puntuación', fontweight='bold', fontsize=14)
                plt.xlabel('Clase', fontweight='bold', fontsize=14)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), 
                          ncol=3, frameon=True, shadow=True, fontsize=12)
                plt.savefig(os.path.join(results_dir, "metrics_by_class.png"), 
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            # Añadir borde al gráfico para un aspecto profesional
            for spine in ax1.spines.values():
                spine.set_linewidth(1.5)
                
            # Título general
            fig.suptitle('Evaluación del Modelo de Clasificación', 
                       fontsize=20, fontweight='bold', y=0.98)
            
            fig.tight_layout()
            plt.subplots_adjust(top=0.9) # Ajustar para el título general
            plt.savefig(os.path.join(results_dir, "metrics_bar.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Gráfico adicional destacando solo F1-Macro
            plt.figure(figsize=(8, 6))
            plt.bar(['F1-Macro (Métrica para datasets desbalanceados)'], [f1_macro], 
                   color='darkred', edgecolor='black', linewidth=1.5)
            plt.ylim(0, 1.05)
            plt.ylabel('Puntuación', fontweight='bold', fontsize=14)
            plt.title('F1-Macro Score', fontsize=18, fontweight='bold', pad=20)
            plt.text(0, f1_macro + 0.03, f"{f1_macro:.4f}", ha='center', 
                   fontsize=16, fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.4'))
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(results_dir, "f1_macro_highlight.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 2. Matriz de confusión profesional (si aplica)
    if y_true is not None and y_pred is not None and label_mapping is not None:
        from sklearn.metrics import confusion_matrix
        
        # Etiquetas y nombres
        inv_label_map = {v: k for k, v in label_mapping.items()}
        class_names = [inv_label_map[i] for i in sorted(inv_label_map.keys())]
        cm = confusion_matrix(y_true, y_pred, labels=sorted(inv_label_map.keys()))
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        
        # Crear mapas de color personalizados para matrices de confusión
        cmap_counts = sns.color_palette("Blues", as_cmap=True)
        cmap_norm = sns.diverging_palette(230, 20, as_cmap=True)
        
        # Figura con dos matrices de confusión (absoluta y normalizada)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot recuentos absolutos
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap_counts, 
                   xticklabels=class_names, yticklabels=class_names, 
                   cbar=True, linewidths=1.5, linecolor='black', 
                   annot_kws={"size":14, "weight":"bold"}, ax=ax1)
        
        ax1.set_title("Matriz de Confusión (Recuentos)", 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel("Predicción", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Real", fontsize=14, fontweight='bold')
        
        # Añadir cuadrado alrededor de la matriz
        for _, spine in ax1.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(2)
            
        # Plot normalizado (porcentajes)
        sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap=cmap_norm, 
                   xticklabels=class_names, yticklabels=class_names, 
                   cbar=True, linewidths=1.5, linecolor='black', 
                   annot_kws={"size":14, "weight":"bold"}, ax=ax2)
        
        ax2.set_title("Matriz de Confusión Normalizada (%)", 
                     fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel("Predicción", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Real", fontsize=14, fontweight='bold')
        
        # Añadir cuadrado alrededor de la matriz
        for _, spine in ax2.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(2)
            
        # Título general
        fig.suptitle('Matrices de Confusión del Modelo', 
                   fontsize=20, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig(os.path.join(results_dir, "confusion_matrices.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Mantener también las matrices individuales para compatibilidad
        plt.figure(figsize=(8, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap_counts, 
                   xticklabels=class_names, yticklabels=class_names, 
                   cbar=True, linewidths=1.5, linecolor='black', 
                   annot_kws={"size":14, "weight":"bold"})
        plt.title("Matriz de Confusión (Recuentos)", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Predicción", fontsize=14, fontweight='bold')
        plt.ylabel("Real", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "confusion_matrix_counts.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(8, 7))
        sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap=cmap_norm, 
                   xticklabels=class_names, yticklabels=class_names, 
                   cbar=True, linewidths=1.5, linecolor='black', 
                   annot_kws={"size":14, "weight":"bold"})
        plt.title("Matriz de Confusión Normalizada (%)", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Predicción", fontsize=14, fontweight='bold')
        plt.ylabel("Real", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "confusion_matrix_normalized.png"), dpi=300, bbox_inches='tight')
        plt.close()

def add_shap_explainability(model, X_test, results_dir):
    """
    Generate SHAP explanations for the model predictions.
    """
    try:
        import shap
        # Ensure X_test is a DataFrame for SHAP if feature names are desired in plot
        # If X_test is numpy, shap will work but might not have feature names.
        # For TreeExplainer, numpy is fine.
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test) # This will be a list of arrays for multi-class
        
        # Save SHAP values and expected value
        dump(shap_values, os.path.join(results_dir, "shap_values.pkl"))
        dump(explainer.expected_value, os.path.join(results_dir, "shap_expected_value.pkl"))
        log.info(f"SHAP values and expected_value saved to {results_dir}")

        # For multi-class, shap_values is a list of arrays (one per class)
        if isinstance(shap_values, list) and len(shap_values) > 0:
             # Check if X_test needs to be a DataFrame for feature names
            if not isinstance(X_test, pd.DataFrame):
                # Create a dummy DataFrame if X_test is numpy, for feature names in plot
                num_features = X_test.shape[1]
                feature_names = [f"feature_{i}" for i in range(num_features)]
                x_test_df_for_shap = pd.DataFrame(X_test, columns=feature_names)
            else:
                x_test_df_for_shap = X_test

            # Plot for each class
            for i in range(len(shap_values)): # Assuming y has classes 0, 1, 2...
                plt.figure()
                shap.summary_plot(shap_values[i], x_test_df_for_shap, show=False, plot_type="bar")
                plt.title(f"SHAP Summary Plot - Class {i}")
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f"shap_summary_class_{i}.png"))
                plt.close()
            log.info(f"SHAP summary plots saved to {results_dir}")
        else: # Single class case (binary) or if shap_values is not a list
            plt.figure()
            shap.summary_plot(shap_values, X_test, show=False) # Default plot
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "shap_summary.png"))
            plt.close()
            log.info(f"SHAP summary plot saved to {results_dir}")

    except ImportError:
        log.warning("SHAP library not installed. Skipping SHAP plots.")
    except Exception as e:
        log.warning(f"SHAP explainability failed: {e}")

def train_and_evaluate_model(X, y, model_type, results_dir, explain=False, cv_folds=0, text_data=None):
    """
    Train and evaluate a model using either cross-validation or train/test split.
    
    Parameters:
    -----------
    X : array-like
        Features matrix (TF-IDF or other features)
    y : array-like
        Target labels
    model_type : str
        Type of model to train ('rf', 'svm', or 'logreg_bow')
    results_dir : str
        Directory to save results
    explain : bool
        Whether to generate model explanations
    cv_folds : int
        Number of cross-validation folds (if > 1)
    text_data : Series or None
        Text data for BoW vectorization if model_type is 'logreg_bow'
    """
    if model_type == "rf":
        model = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    elif model_type == "svm":
        model = SVC(class_weight="balanced", probability=True, random_state=42)
    elif model_type == "logreg_bow":
        # For Logistic Regression with BoW, we'll create a new CountVectorizer
        # and transform the text data to BoW representation
        if text_data is None:
            raise ValueError("Text data must be provided for BoW + Logistic Regression model")
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, solver='liblinear')
        # We'll handle the BoW transformation separately below
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    metrics = {}
    os.makedirs(results_dir, exist_ok=True)

    if cv_folds > 1:
        log.info(f"Performing {cv_folds}-fold cross-validation...")
        scoring = {
            'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
            'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
            'f1_macro': make_scorer(f1_score, average='macro', zero_division=0)
        }
        start_cv = time.time()
        cv_results = cross_validate(model, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
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
        metrics['train_time'] = cv_time # Overall CV time
        log.info(f"Cross-validation finished in {cv_time:.2f}s. Mean F1 Macro: {metrics['cv_results']['test_f1_macro_mean']:.4f}")
        # Visualización de métricas de CV
        plot_metrics_and_confusion(metrics, results_dir)
        return metrics # Return only CV metrics    # Standard train/test split evaluation (if cv_folds <= 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)
    
    start_train = time.time()
    
    # Special handling for BoW + Logistic Regression
    if model_type == "logreg_bow":
        # Split the text data the same way we split X and y
        text_train, text_test = train_test_split(text_data, stratify=y, random_state=42, test_size=0.2)
        
        # Create and fit the BoW vectorizer
        bow_vectorizer = CountVectorizer(stop_words='english', max_features=1000, min_df=2)
        X_train_bow = bow_vectorizer.fit_transform(text_train)
        
        # Train the model on BoW features
        model.fit(X_train_bow, y_train)
        train_time = time.time() - start_train
        
        # Transform test data and predict
        X_test_bow = bow_vectorizer.transform(text_test)
        preds = model.predict(X_test_bow)
        
        # Save the vectorizer for later use
        dump(bow_vectorizer, os.path.join(results_dir, "bow_vectorizer.joblib"))
    else:
        # Standard model training and prediction
        model.fit(X_train, y_train)
        train_time = time.time() - start_train
        preds = model.predict(X_test)
    
    report_dict = classification_report(y_test, preds, output_dict=True, zero_division=0)
    
    metrics = {
        "f1_macro": float(f1_score(y_test, preds, average='macro', zero_division=0)),
        "f1_micro": float(f1_score(y_test, preds, average='micro', zero_division=0)),
        "f1_weighted": float(f1_score(y_test, preds, average='weighted', zero_division=0)),
        "classification_report": report_dict, # Already includes precision, recall, support per class
        "train_time": train_time
    }

    pd.DataFrame({"y_true": y_test, "y_pred": preds}).to_csv(os.path.join(results_dir, "predictions.csv"), index=False)
    dump(model, os.path.join(results_dir, "final_model.joblib"))

    # Visualización de métricas y matriz de confusión
    label_mapping = {"informative": 0, "neutral": 1, "non_informative": 2}
    plot_metrics_and_confusion(metrics, results_dir, y_true=y_test, y_pred=preds, label_mapping=label_mapping)

    # Explainability (SHAP)
    if explain and model_type == "rf":
        add_shap_explainability(model, X_test, results_dir)
            
    return metrics # Return metrics for single train/test split

def train_and_evaluate_with_predefined_splits(X, y, split_indices, model_type, results_dir, explain=False, text_data=None):
    """
    Train and evaluate a model using predefined train/test/dev splits.
    
    Parameters:
    -----------
    X : array-like
        Features matrix
    y : array-like
        Target vector
    split_indices : dict
        Dictionary with indices for each split: {'train': [...], 'test': [...], 'dev': [...]}
    model_type : str
        Type of model to train ('rf', 'svm', or 'logreg_bow')
    results_dir : str
        Directory to save results
    explain : bool
        Whether to generate SHAP explanations
    text_data : Series or None
        Text data for BoW vectorization if model_type is 'logreg_bow'
    
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    if model_type == "rf":
        model = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    elif model_type == "svm":
        model = SVC(class_weight="balanced", probability=True, random_state=42)
    elif model_type == "logreg_bow":
        if text_data is None:
            raise ValueError("Text data must be provided for BoW + Logistic Regression model")
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, solver='liblinear')
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    metrics = {}
    os.makedirs(results_dir, exist_ok=True)
      # Get train and test data using provided indices
    X_train = X[split_indices['train']]
    y_train = y[split_indices['train']]
    X_test = X[split_indices['test']]
    y_test = y[split_indices['test']]
    
    log.info(f"Training with predefined splits: {len(X_train)} training samples, {len(X_test)} test samples")
    
    # Train the model
    start_train = time.time()
    
    # Special handling for BoW + Logistic Regression
    if model_type == "logreg_bow":
        # Get text data using indices
        text_train = text_data.iloc[split_indices['train']].values if hasattr(text_data, 'iloc') else text_data[split_indices['train']]
        text_test = text_data.iloc[split_indices['test']].values if hasattr(text_data, 'iloc') else text_data[split_indices['test']]
        
        # Create and fit the BoW vectorizer
        bow_vectorizer = CountVectorizer(stop_words='english', max_features=1000, min_df=2)
        X_train_bow = bow_vectorizer.fit_transform(text_train)
        
        # Train the model on BoW features
        model.fit(X_train_bow, y_train)
        train_time = time.time() - start_train
        
        # Transform test data and predict
        X_test_bow = bow_vectorizer.transform(text_test)
        preds = model.predict(X_test_bow)
        
        # Save the vectorizer for later use
        dump(bow_vectorizer, os.path.join(results_dir, "bow_vectorizer.joblib"))
    else:
        # Standard model training and prediction
        model.fit(X_train, y_train)
        train_time = time.time() - start_train
        preds = model.predict(X_test)
    
    report_dict = classification_report(y_test, preds, output_dict=True, zero_division=0)
    
    metrics = {
        "f1_macro": float(f1_score(y_test, preds, average='macro', zero_division=0)),
        "f1_micro": float(f1_score(y_test, preds, average='micro', zero_division=0)),
        "f1_weighted": float(f1_score(y_test, preds, average='weighted', zero_division=0)),
        "classification_report": report_dict,
        "train_time": train_time,
        "split_sizes": {
            "train": len(split_indices['train']),
            "test": len(split_indices['test']),
            "dev": len(split_indices.get('dev', [])),
        }
    }

    # Save predictions to CSV
    pd.DataFrame({"y_true": y_test, "y_pred": preds}).to_csv(os.path.join(results_dir, "predictions.csv"), index=False)
    
    # Save trained model
    dump(model, os.path.join(results_dir, "final_model.joblib"))

    # Visualize metrics and confusion matrix
    label_mapping = {"informative": 0, "neutral": 1, "non_informative": 2}
    plot_metrics_and_confusion(metrics, results_dir, y_true=y_test, y_pred=preds, label_mapping=label_mapping)

    # Explainability (SHAP)
    if explain and model_type == "rf":
        add_shap_explainability(model, X_test, results_dir)
    
    return metrics

def get_git_hash():
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], universal_newlines=True).strip()
        return git_hash
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Disaster Tweet Classification")
    parser.add_argument('--data', type=str, required=True, help='Path to CSV dataset')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--max_feats', type=int, default=1000, help='Max features for TF-IDF')
    parser.add_argument('--model_type', type=str, default='rf', choices=['rf', 'svm', 'logreg_bow'], help='Model type (rf: Random Forest, svm: Support Vector Machine, logreg_bow: Logistic Regression with Bag of Words)')
    parser.add_argument('--explain', action='store_true', help='Enable SHAP explainability')
    parser.add_argument('--cv', type=int, default=0, help='Number of cross-validation folds (e.g., 5). If 0 or 1, uses train/test split. Ignored if dataset has predefined splits.')
    parser.add_argument('--nan_threshold', type=float, default=0.5, help='Threshold for NaN proportion in labels after mapping, above which an error is raised.')
    parser.add_argument('--use_predefined_splits', action='store_true', help='Use predefined splits from the dataset if available (split column).')
    parser.add_argument('--force_random_split', action='store_true', help='Force using random train/test split even if dataset has predefined splits.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    
    os.makedirs(args.results_dir, exist_ok=True) # Ensure results_dir exists early

    df = load_data(args.data)
    
    # Check if the dataset has predefined splits
    has_split_column = 'split' in df.columns
    use_predefined_splits = has_split_column and (args.use_predefined_splits or not args.force_random_split)
    
    if has_split_column:
        log.info(f"Dataset has predefined splits: {df['split'].value_counts().to_dict()}")
        if use_predefined_splits:
            log.info("Using predefined splits from dataset")
        else:
            log.info("Ignoring predefined splits (using random split or cross-validation as specified)")
    
    # Map labels
    mapping = {
        "infrastructure_and_utility_damage": "informative",
        "injured_or_dead_people": "informative",
        "requests_or_urgent_needs": "informative",
        "rescue_volunteering_or_donation_effort": "informative",
        "sympathy_and_support": "informative",
        "other_relevant_information": "neutral",
        "not_humanitarian": "non_informative",
    }
    if "class_label" in df.columns:
        original_label_count = len(df)
        df["label"] = df["class_label"].map(mapping)
        # Validate NaN values after mapping
        nan_count = df["label"].isna().sum()
        nan_proportion = nan_count / original_label_count
        log.info(f"NaNs after label mapping: {nan_count} ({nan_proportion:.2%})")
        if nan_proportion > args.nan_threshold:
            raise ValueError(f"Proportion of NaN values in label column ({nan_proportion:.2%}) exceeds threshold ({args.nan_threshold:.2%}). Check label mapping or data quality.")
        df = df[df["label"].notna()].reset_index(drop=True)
        log.info(f"Data size after removing NaNs in label: {len(df)}")
        if df.empty:
            raise ValueError("No data remaining after label mapping and NaN removal. Check label mapping and data.")
    else:
        # Ensure 'label' column exists if 'class_label' is not used for mapping
        if "label" not in df.columns:
             raise ValueError("No 'class_label' column for mapping and no 'label' column found. Please provide a dataset with a 'class_label' or 'label' column.")
        log.info("Using existing 'label' column.")

    # Compute features for the whole dataset
    df = compute_features(df)
    X_text, vectorizer = extract_features(df, args.max_feats)
    dump(vectorizer, os.path.join(args.results_dir, "tfidf_vectorizer.joblib"))
    
    meta_cols = [
        "sentiment", "urgency", "Tweet_Retweets", "Tweet_Likes", "tweet_len",
        "num_exclaims", "num_questions", "num_hashtags", "num_mentions"
    ]
    # Ensure meta columns exist, fill with 0 if not (already handled in compute_features for some)
    for col in meta_cols:
        if col not in df.columns:
            df[col] = 0 
            log.warning(f"Meta column '{col}' not found, adding as zeros.")

    X_meta = df[meta_cols].values
    
    # Check if X_text is empty (e.g., if all texts were empty strings)
    if X_text.shape[0] == 0:
        raise ValueError("TF-IDF features (X_text) are empty. This might happen if 'cleaned' text column is empty or all texts are stop words.")

    # Ensure X_text and X_meta have the same number of rows
    if X_text.shape[0] != X_meta.shape[0]:
        raise ValueError(f"Mismatch in number of samples between TF-IDF features ({X_text.shape[0]}) and meta features ({X_meta.shape[0]}). This should not happen.")

    X = np.hstack([X_text.toarray(), X_meta]) # Use .toarray() instead of .todense() for sparse matrix
    
    # Map string labels to numerical representation
    label_mapping = {"informative": 0, "neutral": 1, "non_informative": 2}
    if not df["label"].isin(label_mapping.keys()).all():
        unknown_labels = df[~df["label"].isin(label_mapping.keys())]["label"].unique()
        log.warning(f"Unknown values found in 'label' column: {unknown_labels}. These will become NaN after mapping and might be dropped or cause errors if not handled.")

    y = df["label"].map(label_mapping).to_numpy()

    # Handle potential NaNs in y that could arise from unmapped labels
    if np.isnan(y).any():
        log.warning(f"NaNs found in target variable 'y' after mapping. Count: {np.isnan(y).sum()}. These samples will be dropped.")
        valid_indices = ~np.isnan(y)
        X = X[valid_indices]
        y = y[valid_indices].astype(int) # Ensure y is integer type after dropping NaNs
        if X.shape[0] == 0:
            raise ValueError("No samples remaining after removing NaNs from target variable 'y'. Check label mapping.")
    else:
        y = y.astype(int) # Ensure y is integer type

    if X.shape[0] == 0:
        raise ValueError("Feature matrix X is empty before training. Check data processing steps.")
    if len(y) == 0:
        raise ValueError("Target variable y is empty before training. Check data processing steps.")
    if X.shape[0] != len(y):
        raise ValueError(f"Mismatch between X samples ({X.shape[0]}) and y samples ({len(y)}) before training.")

    # Choose evaluation strategy based on dataset and args
    if use_predefined_splits:
        # Get indices for each split
        split_indices = {}
        all_indices = np.arange(len(df))
        
        # Handle the case where there's an 'all' split value
        train_indices = all_indices[(df['split'] == 'train') | (df['split'] == 'all')]
        test_indices = all_indices[df['split'] == 'test'] 
        dev_indices = all_indices[df['split'] == 'dev']
        
        split_indices = {
            'train': train_indices,
            'test': test_indices,
            'dev': dev_indices
        }
        
        log.info(f"Using predefined splits - Train: {len(train_indices)}, Test: {len(test_indices)}, Dev: {len(dev_indices)} samples")
        
        # For the logistic regression with BoW, we need to pass the cleaned text
        text_data = None
        if args.model_type == "logreg_bow":
            text_data = df["cleaned"].values if "cleaned" in df.columns else df["clean_text"].values
            log.info(f"Using BoW features from text column with {len(text_data)} entries")
        
        # Train and evaluate using predefined splits
        model_metrics = train_and_evaluate_with_predefined_splits(
            X, y, split_indices, args.model_type, args.results_dir, 
            explain=args.explain, text_data=text_data
        )
    else:
        # Use cross-validation or random train/test split
        cv_folds = args.cv
        log.info(f"Using {'cross-validation' if cv_folds > 1 else 'random train/test split'}")
        
        # For the logistic regression with BoW, we need to pass the cleaned text
        text_data = None
        if args.model_type == "logreg_bow":
            text_data = df["cleaned"].values if "cleaned" in df.columns else df["clean_text"].values
            log.info(f"Using BoW features from text column with {len(text_data)} entries")
        
        model_metrics = train_and_evaluate_model(
            X, y, args.model_type, args.results_dir, 
            explain=args.explain, cv_folds=cv_folds, text_data=text_data
        )

    # Add timestamp and git hash to metrics
    model_metrics['timestamp'] = datetime.datetime.now().isoformat()
    model_metrics['git_hash'] = get_git_hash()
    
    # Save all metrics to metrics.json
    metrics_path = os.path.join(args.results_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(model_metrics, f, indent=2, cls=NpEncoder) # Use NpEncoder for numpy types
    log.info(f"Metrics saved to {metrics_path}")

# Custom JSON encoder to handle numpy types that may appear in metrics
class NpEncoder(json.JSONEncoder):
    def default(self, o): # Changed 'obj' to 'o'
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super(NpEncoder, self).default(o)

if __name__ == "__main__":
    main()