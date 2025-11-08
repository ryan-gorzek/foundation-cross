"""
Cross-model comparison utilities.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from .metrics import compute_metrics, compute_per_class_metrics
from .visualization import plot_confusion_matrix, plot_metrics_comparison


def load_run_results(run_dir: Path) -> Dict[str, Any]:
    """
    Load results from a model run directory.
    
    Parameters
    ----------
    run_dir : Path
        Directory containing run results
        
    Returns
    -------
    results : Dict
        Dictionary containing predictions, labels, metrics, etc.
    """
    results = {
        'run_dir': run_dir,
        'run_name': run_dir.name,
    }
    
    # Load predictions
    matches = list(run_dir.glob("predictions_*.csv"))
    if len(matches) != 1:
        raise FileExistsError(f"Expected 1 predictions_*.csv, found {len(matches)}: {matches}")
    predictions_path = matches[0]
    if predictions_path.exists():
        predictions_df = pd.read_csv(predictions_path)
        results['predicted_label'] = predictions_df['predicted_label'].values
        results['true_label'] = predictions_df['true_label'].values
        results['predicted_label_id'] = predictions_df['predicted_label_id'].values
        results['true_label_id'] = predictions_df['true_label_id'].values
    
    # Load metrics
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            results['metrics'] = json.load(f)
    
    # Load config
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        from ..utils import load_yaml
        results['config'] = load_yaml(config_path)
    
    return results


def compare_model_runs(
    run_dirs: List[Path],
    output_dir: Path,
    experiment_name: str = "comparison",
    logger=None
) -> Dict[str, Any]:
    """
    Compare multiple model runs on the same experiment.
    
    Parameters
    ----------
    run_dirs : List[Path]
        List of run directories to compare
    output_dir : Path
        Directory to save comparison results
    experiment_name : str
        Name for the comparison
    logger : Optional
        Logger instance
        
    Returns
    -------
    comparison_results : Dict
        Comparison results and statistics
    """
    if logger:
        logger.info(f"Comparing {len(run_dirs)} model runs...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    all_results = []
    for run_dir in run_dirs:
        if logger:
            logger.info(f"  Loading: {run_dir.name}")
        results = load_run_results(run_dir)
        all_results.append(results)

    # Compare metrics
    metrics_comparison = {}
    for results in all_results:
        if 'metrics' in results:
            model_name = results['run_name'].split('_')[0]  # Extract model name
            metrics_comparison[model_name] = results['metrics']
    
    # Plot metrics comparison
    if metrics_comparison:
        if logger:
            logger.info("Generating metrics comparison plot...")
        plot_metrics_comparison(
            metrics_comparison,
            output_dir / f"{experiment_name}_metrics_comparison.png",
            title=f"{experiment_name}: Model Comparison"
        )
    
    # Create comparison table
    comparison_data = []
    for results in all_results:
        model_name = results['run_name'].split('_')[0]
        if 'metrics' in results:
            row = {'model': model_name}
            row.update(results['metrics'])
            comparison_data.append(row)
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_csv = output_dir / f"{experiment_name}_metrics_table.csv"
        comparison_df.to_csv(comparison_csv, index=False)
        if logger:
            logger.info(f"Saved metrics table to {comparison_csv}")
    
    # Generate side-by-side confusion matrices
    if len(all_results) <= 3:  # Only if we have a reasonable number
        if logger:
            logger.info("Generating side-by-side confusion matrices...")
        plot_side_by_side_confusion_matrices(
            all_results,
            output_dir / f"{experiment_name}_confusion_matrices.png",
            experiment_name
        )
    else:
        if logger:
            logger.info(f"Skipped side-by-side confusion matrices: too many results...")
    
    # Save comparison summary
    comparison_summary = {
        'experiment_name': experiment_name,
        'n_models': len(all_results),
        'models': [r['run_name'] for r in all_results],
        'metrics_comparison': metrics_comparison,
    }
    
    summary_path = output_dir / f"{experiment_name}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(comparison_summary, f, indent=2)
    
    if logger:
        logger.info(f"Comparison complete. Results saved to {output_dir}")
    
    return comparison_summary


def plot_side_by_side_confusion_matrices(
    results_list: List[Dict],
    save_path: Path,
    experiment_name: str,
    figsize_per_plot: tuple = (6, 5)
) -> None:
    """
    Plot confusion matrices side-by-side for comparison.
    
    Parameters
    ----------
    results_list : List[Dict]
        List of results dictionaries from different runs
    save_path : Path
        Path to save figure
    experiment_name : str
        Experiment name for title
    figsize_per_plot : tuple
        Size of each subplot
    """

    n_models = len(results_list)
    ncols = min(n_models, 2)
    nrows = (n_models + ncols - 1) // ncols
    
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
    )
    
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, results in enumerate(results_list):
        print(results.keys())
        if 'true_label' not in results or 'predicted_label' not in results:
            continue

        predictions = results['predicted_label']
        labels = results['true_label']
        model_name = results['run_name'].split('_')[0]
        
        # Get unique true and predicted labels
        unique_true = sorted(set(labels))
        unique_pred = sorted(set(predictions))

        # Encode labels
        true_encoder = LabelEncoder()
        true_encoder.fit(unique_true)
        true_encoded = true_encoder.transform(labels)
        
        pred_encoder = LabelEncoder()
        pred_encoder.fit(unique_pred)
        pred_encoded = pred_encoder.transform(predictions)
        
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
        print(cm_df)
        # Apply custom ordering if specified
        if 'row_order' in results['config']:
            # Filter to only include labels that exist in the data
            row_order = results['config']['row_order']
            row_order_filtered = [r for r in row_order if r in cm_df.index]
            if row_order_filtered:
                cm_df = cm_df.reindex(row_order_filtered)
        else:
            # Default: alphabetical
            cm_df = cm_df.sort_index()
        
        if 'col_order' in results['config']:
            # Filter to only include labels that exist in the data
            col_order = results['config']['col_order']
            col_order_filtered = [c for c in col_order if c in cm_df.columns]
            if col_order_filtered:
                cm_df = cm_df[col_order_filtered]
        else:
            # Default: alphabetical
            cm_df = cm_df[sorted(cm_df.columns)]
        
        # Plot
        ax = axes[idx]
        sns.heatmap(
            cm_df,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            ax=ax,
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Proportion"}
        )
        
        # Get accuracy if available
        accuracy = results.get('metrics', {}).get('accuracy', None)
        title = f"{model_name}"
        if accuracy is not None:
            title += f"\nAccuracy: {accuracy:.3f}"
        
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("True Label", fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"{experiment_name}: Confusion Matrix Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_comparison_report(
    run_dirs: List[Path],
    output_dir: Path,
    experiment_name: str = "comparison",
    logger=None
) -> Path:
    """
    Generate a comprehensive comparison report in markdown format.
    
    Parameters
    ----------
    run_dirs : List[Path]
        List of run directories to compare
    output_dir : Path
        Directory to save report
    experiment_name : str
        Name for the comparison
    logger : Optional
        Logger instance
        
    Returns
    -------
    report_path : Path
        Path to generated report
    """
    if logger:
        logger.info("Generating comparison report...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{experiment_name}_report.md"
    
    # Load all results
    all_results = [load_run_results(run_dir) for run_dir in run_dirs]
    
    # Generate report
    with open(report_path, 'w') as f:
        f.write(f"# {experiment_name}: Model Comparison Report\n\n")
        f.write(f"Comparing {len(all_results)} model runs.\n\n")
        
        # Models
        f.write("## Models\n\n")
        for i, results in enumerate(all_results, 1):
            f.write(f"{i}. **{results['run_name']}**\n")
        f.write("\n")
        
        # Metrics table
        f.write("## Performance Metrics\n\n")
        f.write("| Model | Accuracy | Precision | Recall | F1 (Macro) | F1 (Weighted) |\n")
        f.write("|-------|----------|-----------|--------|------------|---------------|\n")
        
        for results in all_results:
            model_name = results['run_name'].split('_')[0]
            if 'metrics' in results:
                m = results['metrics']
                f.write(
                    f"| {model_name} | "
                    f"{m.get('accuracy', 0):.4f} | "
                    f"{m.get('precision_macro', 0):.4f} | "
                    f"{m.get('recall_macro', 0):.4f} | "
                    f"{m.get('f1_macro', 0):.4f} | "
                    f"{m.get('f1_weighted', 0):.4f} |\n"
                )
        f.write("\n")
        
        # Best model
        f.write("## Best Model\n\n")
        best_model = None
        best_f1 = -1
        for results in all_results:
            if 'metrics' in results:
                f1 = results['metrics'].get('f1_macro', 0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = results['run_name'].split('_')[0]
        
        if best_model:
            f.write(f"**{best_model}** achieved the highest macro F1 score: {best_f1:.4f}\n\n")
        
        # Links to visualizations
        f.write("## Visualizations\n\n")
        metrics_plot = f"{experiment_name}_metrics_comparison.png"
        cm_plot = f"{experiment_name}_confusion_matrices.png"
        
        if (output_dir / metrics_plot).exists():
            f.write(f"- [Metrics Comparison]({metrics_plot})\n")
        if (output_dir / cm_plot).exists():
            f.write(f"- [Confusion Matrices]({cm_plot})\n")
        f.write("\n")
        
        # Individual model details
        f.write("## Individual Model Results\n\n")
        for results in all_results:
            model_name = results['run_name']
            f.write(f"### {model_name}\n\n")
            
            if 'metrics' in results:
                f.write("**Metrics:**\n\n")
                for key, value in results['metrics'].items():
                    if isinstance(value, float):
                        f.write(f"- {key}: {value:.4f}\n")
                    else:
                        f.write(f"- {key}: {value}\n")
                f.write("\n")
            
            confusion_matrix_path = results['run_dir'] / "confusion_matrix.png"
            if confusion_matrix_path.exists():
                rel_path = confusion_matrix_path.relative_to(output_dir.parent)
                f.write(f"[View Confusion Matrix]({rel_path})\n\n")
    
    if logger:
        logger.info(f"Report saved to {report_path}")
    
    return report_path
