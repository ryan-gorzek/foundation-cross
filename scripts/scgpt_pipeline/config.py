"""
Configuration file for scGPT cross-species label transfer pipeline.
Modify these parameters as needed for your experiments.
"""
from pathlib import Path

# ============================================================================
# DATA PATHS
# ============================================================================
DATA_DIR = Path("../data")
MOUSE_H5AD = DATA_DIR / "raw/Mouse_V1_P38_All.h5ad"
OPOSSUM_H5AD = DATA_DIR / "raw/Opossum_V1_All_Labeled.h5ad"

# Pre-trained model
PRETRAINED_MODEL_DIR = Path("../save/scGPT_human")

# Output directory
SAVE_DIR = Path("./save")

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================
# Training
SEED = 0
DO_TRAIN = True
EPOCHS = 1
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
SCHEDULE_RATIO = 0.9  # Learning rate decay ratio

# Model architecture
LAYER_SIZE = 128  # Embedding dimension
N_LAYERS = 4  # Number of transformer encoder layers
N_HEAD = 4  # Number of attention heads
DROPOUT = 0.2

# Training objectives
MASK_RATIO = 0.0  # Masked language modeling ratio (0.0 = disabled)
MVC = False  # Masked value prediction for cell embedding
ECS_THRES = 0.0  # Elastic cell similarity threshold (0.0 = disabled)
DAB_WEIGHT = 0.0  # Domain adaptation weight

# Data processing
N_BINS = 51  # Number of bins for gene expression
INCLUDE_ZERO_GENE = False  # Include zero-expression genes
MAX_SEQ_LEN = 3001  # Maximum sequence length

# Optimization
AMP = True  # Automatic mixed precision
FAST_TRANSFORMER = True
FREEZE_ENCODER = False  # Freeze encoder weights during fine-tuning

# ============================================================================
# EVALUATION
# ============================================================================
SAVE_EVAL_INTERVAL = 5  # Save/evaluate every N epochs
LOG_INTERVAL = 100  # Log training stats every N batches

# ============================================================================
# WANDB (optional, set to None to disable)
# ============================================================================
WANDB_PROJECT = "scGPT_cross_species"
WANDB_ENTITY = None  # Set to your wandb username/team
DATASET_NAME = "mouse_opossum"
