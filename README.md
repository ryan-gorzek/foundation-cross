# foundation-cross

This repository provides a modular framework for benchmarking label transfer methods across species (easily adapted for cross-modality experiments). It currently compares a transformer-based foundation model (scGPT) with a traditional reference mapping approach (Seurat MapQuery) for transferring cell type annotations between mouse and opossum primary visual cortex.

## Overview

This pipeline enables systematic comparison of different label transfer approaches across the same reference and query datasets. Key features:

- **Modular architecture**: Easy to add new models, datasets, or species
- **Reproducible experiments**: YAML-based configuration with version tracking
- **Comprehensive evaluation**: Confusion matrices, per-class metrics, and cross-model comparison tools

### Current Supported Models

1. **scGPT** - Pretrained transformer model for single-cell transcriptomics
2. **Seurat MapQuery** - Reference-based label transfer using CCA anchors

## Repository Structure

```
foundation-cross/
├── configs/
│   ├── datasets/          # Dataset specifications (mouse.yaml, opossum.yaml)
│   ├── experiments/       # Experiment configs (mouse_to_opossum.yaml)
│   └── models/            # Model hyperparameters (scgpt.yaml, seurat_mapquery.yaml)
│
├── data/
│   └── raw/              # H5AD files (Mouse_V1_P38_All.h5ad, Opossum_V1_All.h5ad)
│
├── src/
│   ├── data/             # Data loading, preprocessing, validation
│   ├── models/           # Model implementations (base class + specific models)
│   │   ├── scgpt/        # scGPT implementation
│   │   └── seurat/       # Seurat MapQuery with R integration
│   ├── analysis/         # Metrics computation, visualization, comparison
│   └── utils/            # Configuration, logging, reproducibility
│
├── scripts/
│   ├── run_experiment.py     # Main entry point
│   └── compare_models.py     # Cross-model comparison
│
├── results/              # Experiment outputs (organized by reference_query/model_timestamp)
│
├── environment_scgpt.yaml    # Conda environment for scGPT
└── environment_seurat.yaml   # Conda environment for Seurat
```

## Installation

### Note

These environments can be incredibly difficult to build and may not work straightforwardly on your system with these instructions. Feel free to reach out with questions.

### Requirements

- Python 3.11
- R 4.3+ (for Seurat)
- CUDA-capable GPU (optional, for scGPT)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/ryan-gorzek/foundation-cross.git
cd foundation-cross
```

2. **Create conda environments**

For scGPT experiments:
```bash
conda env create -f environment_scgpt.yaml
conda activate scgpt_env
```

For Seurat experiments:
```bash
conda env create -f environment_seurat.yaml
conda activate seurat_env
```

3. **Install R packages** (for Seurat only)
```bash
# In R console or via Rscript
R -e "
options(repos = c(CRAN = 'https://cloud.r-project.org'))
install.packages(c('Seurat', 'optparse', 'jsonlite', 'remotes'))
remotes::install_github('mojaveazure/seurat-disk')
"
```

4. **Download pretrained models** (for scGPT)
```bash
# Download scGPT pretrained model
mkdir -p models/scgpt/whole-human
# Download from scGPT repository or provide your own pretrained weights
```

## Quick Start

### Run a single experiment

```bash
conda activate scgpt_env
python scripts/run_experiment.py --config configs/experiments/mouse_to_opossum_scgpt.yaml
```

Or with Seurat:
```bash
conda activate seurat_env
python scripts/run_experiment.py --config configs/experiments/mouse_to_opossum_seurat.yaml
```

### Compare multiple models

```bash
python scripts/compare_models.py \
  --experiment mouse_opossum_comparison \
  --runs results/mouse_opossum/scgpt_Nov04-10-30 \
         results/mouse_opossum/seurat_mapquery_Nov04-13-45 \
  --output results/comparisons \
  --report
```

## Configuration

Experiments are defined through hierarchical YAML configs:

### Experiment Config (`configs/experiments/mouse_to_opossum.yaml`)
```yaml
name: mouse_opossum_transfer
description: "Cross-species label transfer from mouse to opossum V1"

reference:
  dataset: mouse
  labels: null  # Use all labels

query:
  - dataset: opossum
    labels: null

model: scgpt
model_config: configs/models/scgpt.yaml

training:
  seed: 0
  epochs: 10
  batch_size: 16
  validation_split: 0.1

output:
  save_dir: results
  confusion_matrix:
    figsize: [14, 12]
    row_order: [...] # Specify custom ordering
```

### Dataset Config (`configs/datasets/mouse.yaml`)
```yaml
name: mouse
species: Mus musculus
data_path: data/raw/Mouse_V1_P38_All.h5ad
celltype_column: Subclass

preprocessing:
  min_genes: 8
  min_counts: 3
  normalize_total: 1e4
  n_bins: 51
```

## Output Structure

Each experiment run creates a timestamped directory:

```
results/mouse_opossum/scgpt_Nov04-10-30/
├── config.yaml                 # Complete config snapshot
├── run.log                     # Detailed execution log
├── predictions_opossum.csv     # Cell-level predictions
├── metrics_opossum.json        # Overall + per-class metrics
├── confusion_matrix_opossum.png
├── per_class_f1_opossum.png
├── model_outputs/              # Model-specific artifacts
│   ├── best_model.pt
│   └── vocab.json
└── reproducibility.json        # Git hash, environment info
```

## Key Features

### Robust Label Handling

- **Mismatched labels**: Query cells with labels not in reference are tracked but not penalized
- **Original label preservation**: Full confusion matrix shows all query and reference cell types, not just matched ones

### Cross-Platform Integration

- **R ↔ Python bridge**: Integration with Seurat via subprocess calls
- **Format conversion**: Automatic H5AD → H5Seurat → RDS conversion with metadata preservation
- **Seurat v5 compatibility**: Handles layer joining and assay structure changes

### Reproducibility

- **Git tracking**: Automatic commit hash recording
- **Config hashing**: Unique fingerprint for each parameter set
- **Environment logging**: Captures Python, PyTorch, CUDA versions
- **Seed management**: Deterministic RNG for NumPy, PyTorch, Python

## Development

### Adding a New Model

1. Create model directory: `src/models/your_model/`
2. Implement `BaseLabelTransferModel`:
```python
class YourModel(BaseLabelTransferModel):
    def train(self, reference_data, **kwargs): ...
    def predict(self, query_data, **kwargs): ...
    def save_model(self): ...
```
3. Register in `src/models/__init__.py`:
```python
MODEL_REGISTRY = {
    'your_model': YourModel,
    ...
}
```
4. Create config: `configs/models/your_model.yaml`
5. Run: `python scripts/run_experiment.py --config your_experiment.yaml`

### Adding a New Dataset

1. Create dataset config: `configs/datasets/new_species.yaml`
2. Specify data path, celltype column, preprocessing parameters
3. Reference in experiment config:
```yaml
query:
  - dataset: new_species
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{foundation_cross,
  author = {Ryan Gorzek},
  title = {foundation-cross: Benchmarking single-cell foundation models for cross-species label transfer},
  year = {2025},
  url = {https://github.com/ryan-gorzek/foundation-cross}
}
```

### Model Citations

**scGPT:**
```bibtex
@article{cui2023scGPT,
title={scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI},
author={Cui, Haotian and Wang, Chloe and Maan, Hassaan and Pang, Kuan and Luo, Fengning and Wang, Bo},
journal={bioRxiv},
year={2023},
publisher={Cold Spring Harbor Laboratory}
}
```

**Seurat:**
```bibtex
@article{hao2023dictionary,
title={Dictionary learning for integrative, multimodal and scalable single-cell analysis},
author={Hao, Yuhan and Stuart, Tim and Kowalski, Madeline H. and Choudhary, Saket and Hoffman, Paul and Hartman, Austin and Srivastava, Avi and Molla, Gesmira and Madad, Shaista and Fernandez-Granda, Carlos and Satija, Rahul},
journal={Nature Biotechnology},
volume={42},
pages={293--304},
year={2024},
publisher={Nature Publishing Group},
doi={10.1038/s41587-023-01875-1}
}
```

## License

MIT. See `LICENSE`.
