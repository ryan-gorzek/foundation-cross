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
from typing import Dict, Any, Optional
import scanpy as sc
import anndata as ad
from scipy.sparse import issparse
from torch.utils.data import DataLoader

from ..base import BaseLabelTransferModel
from .architecture import MLPClassifier
from .utils import make_dataloaders_group_split, make_dataloader
from .train import train_epoch, evaluate

class NNModel(BaseLabelTransferModel):
    """Neural network model for label transfer."""

    def __init__(self, config: Dict[str, Any], save_dir: Path, logger=None):
        super().__init__(config, save_dir, logger)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_info(f"Using device: {self.device}")

        # Extract config values
        self.genes = config.get('pretrained', {})
        self.architecture = config.get('architecture', {})
        # Store model
        self.model = None

    def _initialize_model(self, reference_data: ad.AnnData) -> nn.Module:
        
        # Inputs
        var_col = self.genes.get("var_col", "highly_variable")
        if var_col in reference_data.var.columns:
            n_inputs = int(reference_data.var[var_col].sum())
            self.log_info(f"Using {n_inputs} genes")
        else:
            raise ValueError(f"Reference data lacks {var_col} in var")

        # Outputs
        if 'celltype' in reference_data.obs.columns:
            self.reference_categories = reference_data.obs['celltype'].cat.categories.tolist()
            n_outputs = len(self.reference_categories)
            self.log_info(f"Using {len(self.reference_categories)} reference categories")
        else:
            raise ValueError("Reference data must have 'celltype' column in obs")
        # Model
        hidden = self.architecture.get("hidden", (512, 512, 256))
        dropout = self.architecture.get("dropout", 0.2)
        self.model = MLPClassifier(n_inputs,
                                   n_outputs,
                                   hidden=hidden,
                                   dropout = dropout).to(self.device)

    def _preprocess_data(self, reference_data: ad.AnnData, query_data: Optional[ad.AnnData] = None):
        """
        Compute HVGs for reference data within common gene space.
        """
        reference_genes = reference_data.var["gene_name"].values
        query_genes = query_data.var["gene_name"].values
        
        # Find common genes (intersection)
        common_genes = [g for g in reference_genes if g in query_genes]
        reference_data = reference_data[:, common_genes].copy()
        
        # Get HVGs
        sc.pp.highly_variable_genes(reference_data, n_top_genes=self.genes.get("num", 3000))
        n_hvgs = reference_data.var['highly_variable'].sum()
        self.log_info(f"Found {n_hvgs} HVGs out of {reference_data.shape[1]} common genes")

        return reference_data

    def train(self, reference_data: ad.AnnData, query_data: Optional[ad.AnnData] = None, **kwargs):
        """
        Train neural network model on reference data.
        
        Parameters
        ----------
        reference_data : AnnData
            Reference dataset with 'celltype_id' in obs
        """
        self.log_info("Training neural network model")

        var_col = self.genes.get("var_col", "highly_variable")
        if var_col == "highly_variable":
            reference_data = self._preprocess_data(reference_data, query_data)
        self._initialize_model(reference_data)

        # Store the training gene names for indexing query
        gene_idx = reference_data.var[var_col]
        self.training_genes = reference_data.var.loc[gene_idx].index.to_numpy()

        # Make the train and validation DataLoaders
        (
            train_loader, 
            val_loader, 
            train_idx, 
            val_idx,
            class_weights
        ) = make_dataloaders_group_split(
            reference_data,
            var_col=var_col,
            obs_col="celltype_id",
            make_dense=False,
            batch_col=kwargs.get("batch_col", "sample"),
            val_frac=kwargs.get("val_frac", 0.2),
            batch_size = kwargs.get("batch_size", 512),
            use_weighted_sampler=kwargs.get("use_weighted_sampler", False),
        )

        self.log_info(f"Training samples: {len(train_idx)}")
        self.log_info(f"Validation samples: {len(val_idx)}")

        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        lr = kwargs.get("learning_rate", 3e-4)
        wd = kwargs.get("weight_decay", 1e-4)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(lr), weight_decay=float(wd))

        # Training loop
        num_epochs = kwargs.get("num_epochs", 25)
        self.model.train()
        for ep in range(1, num_epochs + 1):
            
            train_loss, train_acc = train_epoch(self.model, train_loader, criterion, optimizer, self.device)
            val_loss, val_acc = evaluate(self.model, val_loader, criterion, self.device)

            self.log_info(
                    f"Epoch {ep:3d} | train loss {train_loss:5.4f} | "
                    f"valid loss {val_loss:5.4f} | valid accuracy {val_acc:5.4f}"
                )

        self.log_info("Training complete")

    @torch.no_grad()
    def predict(self, query_data: ad.AnnData, **kwargs):
        """
        Predict labels for query data.
        
        Parameters
        ----------
        query_data : AnnData
            Query dataset
            
        Returns
        -------
        predictions : np.ndarray
            Predicted cell type IDs
        """
        self.log_info("Predicting labels for query data")
        
        # Ensure query has exact same genes as training (in same order)
        if hasattr(self, 'training_genes'):
            training_gene_set = set(self.training_genes)
            query_genes = query_data.var["gene_name"].values
            
            # Find common genes (intersection)
            common_genes = [g for g in self.training_genes if g in query_genes]
            n_common = len(common_genes)
            n_missing = len(self.training_genes) - n_common
            
            if n_missing > 0:
                self.log_warning(f"Query missing {n_missing}/{len(self.training_genes)} training genes")
            
            if n_common == 0:
                raise ValueError("No common genes between training and query")
            
            # Subset query to common genes and reorder to match training
            query_data = query_data[:, common_genes].copy()
            query_data.var['prediction_genes'] = True # Mark all genes for make_dataloader
            
            self.log_info(f"Using {n_common} common genes for prediction")
        
        query_loader = make_dataloader(
            query_data,
            var_col='prediction_genes',
            obs_col='celltype_id',
            make_dense=False,
            batch_size = kwargs.get('batch_size', 512)
            )

        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        predictions = evaluate(self.model, query_loader, criterion, self.device, return_raw=True)
        
        self.log_info(f"Generated predictions for {len(predictions)} cells")
        return predictions

    def save_model(self):
        """Save model weights."""
        if self.model is not None:
            model_path = self.model_outputs_dir / "best_model.pt"
            torch.save(self.model.state_dict(), model_path)
            self.log_info(f"Saved model weights to {model_path}")
