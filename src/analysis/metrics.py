"""
Evaluation metrics for label transfer.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Optional, List


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted labels (integers)
    labels : np.ndarray
        True labels (integers)
    valid_mask : Optional[np.ndarray]
        Boolean mask indicating which predictions are valid for evaluation
    label_names : Optional[List[str]]
        Names of the labels for reporting
        
    Returns
    -------
    metrics : Dict[str, float]
        Dictionary of metric names to values
    """
    # Apply valid mask if provided
    if valid_mask is not None:
        predictions = predictions[valid_mask]
        labels = labels[valid_mask]
    
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'precision_macro': precision_score(labels, predictions, average='macro', zero_division=0),
        'recall_macro': recall_score(labels, predictions, average='macro', zero_division=0),
        'f1_macro': f1_score(labels, predictions, average='macro', zero_division=0),
        'precision_weighted': precision_score(labels, predictions, average='weighted', zero_division=0),
        'recall_weighted': recall_score(labels, predictions, average='weighted', zero_division=0),
        'f1_weighted': f1_score(labels, predictions, average='weighted', zero_division=0),
        'n_samples': len(predictions),
    }
    
    return metrics


def compute_per_class_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: List[str],
    valid_mask: Optional[np.ndarray] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class metrics.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted labels (integers)
    labels : np.ndarray
        True labels (integers)
    label_names : List[str]
        Names of the labels
    valid_mask : Optional[np.ndarray]
        Boolean mask for valid predictions
        
    Returns
    -------
    per_class_metrics : Dict[str, Dict[str, float]]
        Nested dictionary: label_name -> metric_name -> value
    """
    # Apply valid mask if provided
    if valid_mask is not None:
        predictions = predictions[valid_mask]
        labels = labels[valid_mask]
    
    # Get per-class precision, recall, f1
    precision = precision_score(labels, predictions, average=None, zero_division=0)
    recall = recall_score(labels, predictions, average=None, zero_division=0)
    f1 = f1_score(labels, predictions, average=None, zero_division=0)
    
    # Count samples per class
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    
    per_class_metrics = {}
    for i, label_name in enumerate(label_names):
        per_class_metrics[label_name] = {
            'precision': float(precision[i]) if i < len(precision) else 0.0,
            'recall': float(recall[i]) if i < len(recall) else 0.0,
            'f1': float(f1[i]) if i < len(f1) else 0.0,
            'support': int(label_counts.get(i, 0)),
        }
    
    return per_class_metrics


def compute_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted labels (integers)
    labels : np.ndarray
        True labels (integers)
    valid_mask : Optional[np.ndarray]
        Boolean mask for valid predictions
    normalize : bool
        Whether to normalize by true label counts
        
    Returns
    -------
    cm : np.ndarray
        Confusion matrix
    """
    # Apply valid mask if provided
    if valid_mask is not None:
        predictions = predictions[valid_mask]
        labels = labels[valid_mask]
    
    cm = confusion_matrix(labels, predictions)
    
    if normalize:
        # Normalize by row (true labels)
        cm = cm.astype('float')
        row_sums = cm.sum(axis=1)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums[:, np.newaxis]
    
    return cm


def get_classification_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: List[str],
    valid_mask: Optional[np.ndarray] = None
) -> str:
    """
    Generate classification report string.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted labels (integers)
    labels : np.ndarray
        True labels (integers)
    label_names : List[str]
        Names of the labels
    valid_mask : Optional[np.ndarray]
        Boolean mask for valid predictions
        
    Returns
    -------
    report : str
        Classification report
    """
    # Apply valid mask if provided
    if valid_mask is not None:
        predictions = predictions[valid_mask]
        labels = labels[valid_mask]
    
    report = classification_report(
        labels,
        predictions,
        target_names=label_names,
        zero_division=0
    )
    
    return report