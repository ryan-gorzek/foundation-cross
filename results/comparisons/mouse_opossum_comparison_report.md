# mouse_opossum_comparison: Model Comparison Report

Comparing 2 model runs.

## Models

1. **scgpt_Nov02-09-02**
2. **seurat_mapquery_Nov04-15-27**

## Performance Metrics

| Model | Accuracy | Precision | Recall | F1 (Macro) | F1 (Weighted) |
|-------|----------|-----------|--------|------------|---------------|
| scgpt | 0.8213 | 0.5012 | 0.5308 | 0.4886 | 0.8248 |
| seurat | 0.8634 | 0.6542 | 0.6117 | 0.5997 | 0.8708 |

## Best Model

**seurat** achieved the highest macro F1 score: 0.5997

## Visualizations

- [Metrics Comparison](mouse_opossum_comparison_metrics_comparison.png)
- [Confusion Matrices](mouse_opossum_comparison_confusion_matrices.png)

## Individual Model Results

### scgpt_Nov02-09-02

**Metrics:**

- accuracy: 0.8213
- precision_macro: 0.5012
- recall_macro: 0.5308
- f1_macro: 0.4886
- precision_weighted: 0.8682
- recall_weighted: 0.8213
- f1_weighted: 0.8248
- n_samples: 18305

### seurat_mapquery_Nov04-15-27

**Metrics:**

- accuracy: 0.8634
- precision_macro: 0.6542
- recall_macro: 0.6117
- f1_macro: 0.5997
- precision_weighted: 0.9209
- recall_weighted: 0.8634
- f1_weighted: 0.8708
- n_samples: 18305

