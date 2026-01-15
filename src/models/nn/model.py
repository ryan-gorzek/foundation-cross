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
from .train import train_epoch, evaluate

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

    def train(self, reference_data: ad.AnnData, query_data: Optional[ad.AnnData] = None, **kwargs):
        """
        Train neural network model on reference data.
        
        Parameters
        ----------
        reference_data : AnnData
            Reference dataset with 'celltype_id' in obs
        """
        self.log_info("Training neural network model")

        # Make the train and validation DataLoaders
        (
            train_loader, 
            val_loader, 
            train_idx, 
            val_idx,
            class_weights
        ) = make_dataloaders_group_split(
            reference_data,
            var_col='highly_variable',
            obs_col='celltype_id',
            make_dense=False,
            sample_col='sample',
            val_frac=kwargs.get('val_frac', 0.2),
            batch_size = kwargs.get('batch_size', 512)
        )

        self.log_info(f"Training samples: {len(train_data)}")
        self.log_info(f"Validation samples: {len(valid_data)}")

        criterion = nn.CrossEntropyLoss(weights=class_weights.to(self.device))
        lr = kwargs.get('learning_rate', 3e-4)
        wd = kwargs.get('weight_decay', 1e-4)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

        # Training loop
        num_epochs = kwargs.get('num_epochs', 25)
        model.train()
        for ep in num_epochs:
            
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            self.log_info(
                    f"Epoch {epoch:3d} | train loss {train_loss:5.4f} | "
                    f"valid loss {val_loss:5.4f} | valid accuracy {val_acc:5.4f}"
                )

        self.log_info("Training complete")

    def predict(self):
        pass
