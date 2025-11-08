"""
Visualization functions for label transfer results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple
from sklearn.preprocessing import LabelEncoder
import anndata as ad


def plot_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    true_label_names: List[str],
    pred_label_names: List[str],
    save_path: Path,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (14, 12),
    row_order: Optional[List[str]] = None,
    col_order: Optional[List[str]] = None,
    valid_mask: Optional[np.ndarray] = None,
    cmap: str = "Blues",
    fmt: str = ".2f",
    query_data: Optional[ad.AnnData] = None
) -> None:
    """
    Plot confusion matrix with customizable ordering.
    
    If query_data is provided with 'celltype_original', uses ALL original query labels
    for rows (true labels), not just those matching reference.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted label IDs
    labels : np.ndarray
        True label IDs
    true_label_names : List[str]
        Names for true labels (rows) - reference categories
    pred_label_names : List[str]
        Names for predicted labels (columns) - reference categories
    save_path : Path
        Path to save figure
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
    row_order : Optional[List[str]]
        Custom order for rows (true labels)
    col_order : Optional[List[str]]
        Custom order for columns (predicted labels)
    valid_mask : Optional[np.ndarray]
        Boolean mask for valid predictions (not used for plotting, all cells shown)
    cmap : str
        Colormap name
    fmt : str
        Format string for annotations
    query_data : Optional[ad.AnnData]
        Query data with celltype_original for full label preservation
    """
    # Use original query labels if available, otherwise use the mapped labels
    if query_data is not None and 'celltype_original' in query_data.obs.columns:
        true_labels_str = query_data.obs['celltype_original'].values
    else:
        # Fallback: try to map label IDs to names
        true_labels_str = np.array([true_label_names[i] if 0 <= i < len(true_label_names) else f"Unknown_{i}" 
                                    for i in labels])
    
    # Map predicted IDs to names
    pred_labels_str = np.array([pred_label_names[i] if 0 <= i < len(pred_label_names) else f"Unknown_{i}" 
                                for i in predictions])
    
    # Get unique labels
    # Rows: ALL query labels that appear in data
    unique_true = sorted(set(true_labels_str))
    # Columns: ALL reference labels (whether predicted or not)
    unique_pred = pred_label_names
    
    # Encode labels
    true_encoder = LabelEncoder()
    true_encoder.fit(unique_true)
    true_encoded = true_encoder.transform(true_labels_str)
    
    pred_encoder = LabelEncoder()
    pred_encoder.fit(unique_pred)
    pred_encoded = pred_encoder.transform(pred_labels_str)
    
    # Build confusion matrix manually to handle non-square case
    cm = np.zeros((len(unique_true), len(unique_pred)), dtype=int)
    for t_idx, p_idx in zip(true_encoded, pred_encoded):
        cm[t_idx, p_idx] += 1
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float')
    row_sums = cm.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    cm_normalized = cm_normalized / row_sums[:, np.newaxis]
    
    # Create DataFrame with proper labels
    cm_df = pd.DataFrame(
        cm_normalized,
        index=true_encoder.classes_,
        columns=pred_encoder.classes_
    )
    
    # Apply custom ordering if specified
    if row_order is not None:
        # Filter to only include labels that exist in the data
        row_order_filtered = [r for r in row_order if r in cm_df.index]
        if row_order_filtered:
            cm_df = cm_df.reindex(row_order_filtered)
    else:
        # Default: alphabetical
        cm_df = cm_df.sort_index()
    
    if col_order is not None:
        # Filter to only include labels that exist in the data
        col_order_filtered = [c for c in col_order if c in cm_df.columns]
        if col_order_filtered:
            cm_df = cm_df[col_order_filtered]
    else:
        # Default: alphabetical
        cm_df = cm_df[sorted(cm_df.columns)]
    print("====== Confusion Matrix ======")
    print(cm_df)
    print("==============================")
    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm_df,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        cbar_kws={"label": "Proportion"},
        vmin=0,
        vmax=1
    )
    plt.title(title, fontsize=14, pad=20)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(
    metrics_dict: dict,
    save_path: Path,
    title: str = "Metrics Comparison",
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot bar chart comparing metrics across different runs/models.
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary mapping run names to metrics dictionaries
    save_path : Path
        Path to save figure
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
    """
    # Extract metrics to compare
    metric_names = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    
    data = []
    for run_name, metrics in metrics_dict.items():
        for metric_name in metric_names:
            if metric_name in metrics:
                data.append({
                    'Run': run_name,
                    'Metric': metric_name,
                    'Value': metrics[metric_name]
                })
    
    df = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(data=df, x='Metric', y='Value', hue='Run')
    plt.title(title, fontsize=14, pad=20)
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Metric", fontsize=12)
    plt.ylim(0, 1)
    plt.legend(title="Model/Run", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_f1(
    per_class_metrics: dict,
    save_path: Path,
    title: str = "Per-Class F1 Scores",
    figsize: Tuple[int, int] = (12, 8),
    top_n: Optional[int] = None
) -> None:
    """
    Plot per-class F1 scores.
    
    Parameters
    ----------
    per_class_metrics : dict
        Dictionary mapping class names to metrics
    save_path : Path
        Path to save figure
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
    top_n : Optional[int]
        If specified, only plot top N classes by F1 score
    """
    # Extract F1 scores
    data = []
    for class_name, metrics in per_class_metrics.items():
        data.append({
            'Class': class_name,
            'F1': metrics['f1'],
            'Support': metrics['support']
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('F1', ascending=False)
    
    if top_n is not None:
        df = df.head(top_n)
    
    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(data=df, x='F1', y='Class', palette='viridis')
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel("F1 Score", fontsize=12)
    plt.ylabel("Class", fontsize=12)
    plt.xlim(0, 1)
    
    # Add support as text
    for i, row in enumerate(df.itertuples()):
        plt.text(row.F1 + 0.02, i, f"n={row.Support}", va='center', fontsize=9)
    
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
