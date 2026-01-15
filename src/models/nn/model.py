"""
Neural network classifier implementation for label transfer.
"""
import os
import json
import copy
import shutil
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any
import anndata as ad
from scipy.sparse import issparse
from torch.utils.data import DataLoader

from ..base import BaseLabelTransferModel
from .architecture import MLPClassifier
from .utils import make_dataloaders_group_split

class NNModel(BaseLabelTransferModel):
    """Neural network model for label transfer."""

    def __init__(self, config: Dict[str, Any], save_dir: Path, logger=None):
        super().__init__(config, save_dir, logger)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_info(f"Using device: {self.device}")

        self.model = None

    def _initialize_model(self, reference_data: ad.AnnData) -> nn.Module:
        
        # Inputs
        if 'higly_variable' in reference_data.var.columns:
            n_inputs = int(reference_data.var["highly_variable"].sum())
            self.log_info(f"Using {n_inputs} genes")
        else:
            raise ValueError("Reference data must have highly variable genes")
        # Outputs
        if 'celltype' in reference_data.obs.columns:
            self.reference_categories = reference_data.obs['celltype'].cat.categories.tolist()
            n_outputs = len(self.reference_categories)
            self.log_info(f"Using {len(self.reference_categories)} reference categories")
        else:
            raise ValueError("Reference data must have 'celltype' column")
        # Model
        self.model = MLPClassifier(n_inputs, n_outputs)

    def train(self):
        pass

    def predict(self):
        pass

