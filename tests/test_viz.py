import pandas as pd
from pathlib import Path
from src.analysis.visualization import plot_confusion_matrix

results_dir = Path("results/mouse_opossum")
run_dirs = sorted(results_dir.glob("scgpt_*"))
latest_run = run_dirs[-1]

# Load predictions
predictions_df = pd.read_csv(latest_run / "predictions_opossum.csv")
predictions = predictions_df['predicted_label_id'].values
true_labels = predictions_df['true_label_id'].values
valid_mask = predictions_df['has_valid_label'].values

# Simulate what the pipeline does - use categorical Index, NOT list
true_label_names = pd.Categorical(predictions_df['true_label']).categories
pred_label_names = pd.Categorical(predictions_df['predicted_label']).categories

# This should reproduce the error
plot_confusion_matrix(
    predictions=predictions,
    labels=true_labels,
    true_label_names=true_label_names,  # Index object, not list
    pred_label_names=pred_label_names,  # Index object, not list
    save_path=latest_run / "confusion_matrix_test.png",
    title="Test",
    figsize=(14, 12),
    valid_mask=valid_mask,
)
