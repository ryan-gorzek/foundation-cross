"""
Analysis and visualization utilities.
"""
from .metrics import (
    compute_metrics,
    compute_per_class_metrics,
    compute_confusion_matrix,
    get_classification_report,
)
from .visualization import (
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_per_class_f1,
)
from .comparison import compare_model_runs, generate_comparison_report

__all__ = [
    'compute_metrics',
    'compute_per_class_metrics',
    'compute_confusion_matrix',
    'get_classification_report',
    'plot_confusion_matrix',
    'plot_metrics_comparison',
    'plot_per_class_f1',
    'compare_model_runs',
    'generate_comparison_report',
]