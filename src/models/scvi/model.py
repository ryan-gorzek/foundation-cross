"""
scVI/scANVI model implementation for integration and label transfer.
"""
import json
import pandas as pd
import numpy as np
import anndata as ad
from pathlib import Path
from typing import Dict, Any
import scanpy as sc
import scvi
import torch

from ..base import BaseLabelTransferModel

class ScVIModel(BaseLabelTransferModel):
    """scVI/scANVI pipeline for cross-species integration and label transfer."""

    def __init__(self, config: Dict[str, Any], save_dir: Path, logger=None):
        super().__init__(config, save_dir, logger)

        # scVI-specific config
        self.scvi_config = config.get('scvi', {})
        self.n_top_genes = self.scvi_config.get('n_top_genes', 2000)
        self.batch_key = self.scvi_config.get('batch_key', 'dataset')
        self.n_layers = self.scvi_config.get('n_layers', 2)
        self.n_latent = self.scvi_config.get('n_latent', 30)
        # scANVI-specific config
        self.scanvi_config = config.get('scanvi', {})
        self.max_epochs = self.scanvi_config.get('max_epochs', 20)
        self.n_samples_per_label = self.scanvi_config.get('n_samples_per_label', 100)

        # Store reference categories for prediction mapping
        self.reference_categories = None
        # Store merged object for sharing across methods
        self.merged_data = None

        # Models (initialized during train)
        self.scvi_model = None
        self.scanvi_model = None

    def train(self, reference_data: ad.AnnData, query_data: Optional[ad.AnnData] = None, **kwargs) -> None:
        """
        Prepare reference data and train scVI/scANVI integration and label transfer models.
        
        Parameters
        ----------
        reference_data : AnnData
            Reference dataset with 'celltype' in obs
        query_data : AnnData
            Query dataset with 'celltype' in obs
        """
        self.log_info("Merging and processing reference and query objects for scVI integration")

        # CRITICAL: Store reference categories for prediction mapping
        if 'celltype' in reference_data.obs.columns:
            self.reference_categories = reference_data.obs['celltype'].cat.categories.tolist()
            self.log_info(f"Stored {len(self.reference_categories)} reference categories")
        else:
            raise ValueError("Reference data must have 'celltype' column")

        # Merge and process objects
        reference_data.obs['dataset'] = 'reference'
        query_data.obs['dataset'] = 'query'
        self.merged_data = anndata.concat([reference_data, query_data])
        self.merged_data.layers['counts'] = adata.X.copy()
        sc.pp.normalize_total(self.merged_data, target_sum=1e4)
        sc.pp.log1p(self.merged_data)
        adata.raw = self.merged_data  # keep full dimension safe
        sc.pp.highly_variable_genes(
            self.merged_data,
            flavor='seurat_v3',
            n_top_genes=self.n_top_genes,
            layer='counts',
            batch_key='dataset',
            subset=True,
        )
    
        # Prepare and train the scVI model, store the embeddings
        self.log_info("Training scVI model")
        scvi.model.SCVI.setup_anndata(self.merged_data, layer="counts", batch_key='dataset')
        self.scvi_model = scvi.model.SCVI(self.merged_data, n_layers=2, n_latent=30)
        self.scvi_model.train()
        self.merged_data.obsm['X_scVI'] = self.scvi_model.get_latent_representation()

        # Prepare and train the scANVI model, store the embeddings
        self.log_info("Training scANVI model")
        self.merged_data.obs['celltype_scanvi'] = -1
        query_mask = self.merged_data.obs['dataset'] == 'query'
        self.merged_data.obs['celltype_scanvi'][ss2_mask] = self.merged_data.obs['celltype_id'][query_mask].values
        self.scanvi_model = scvi.model.SCANVI.from_scvi_model(
            self.scvi_model,
            adata=self.merged_data,
            unlabeled_category=-1,
            labels_key='celltype_scanvi',
        )
        self.scanvi_model.train(max_epochs=self.max_epochs, n_samples_per_label=self.n_samples_per_label)
        self.log_info("Training complete")

    def predict(self, query_data: ad.AnnData, **kwargs) -> np.ndarray:
        """
        Predict labels for query data using scANVI.
        
        Parameters
        ----------
        query_data : AnnData
            Query dataset
            
        Returns
        -------
        predictions : np.ndarray
            Predicted cell type IDs (as integers matching reference categories)
        """
        self.log_info("Predicting labels for query data")
        
        # Predict labels with scANVI
        self.merge_data.obsm['X_scANVI'] = scanvi_model.get_latent_representation(self.merge_data)
        self.merge_data.obs['C_scANVI'] = scanvi_model.predict(self.merge_data)
        query_mask = self.merged_data.obs['dataset'] == 'query'
        predictions = np.array(self.merge_data.obs['C_scANVI'][query_mask]).astype(int)
        # IMPORTANT: Verify query cells match input, in case something happened internally
        query_cells_merged = self.merged_data.obs_names[query_mask]
        query_cells_input = query_data.obs_names
        if len(query_cells_merged) != len(query_cells_input):
            raise ValueError(
                f"Cell count mismatch: {len(query_cells_merged)} in merged vs {len(query_cells_input)} in input"
            )
        if not np.array_equal(query_cells_merged, query_cells_input):
            if set(query_cells_merged) != set(query_cells_input):
                raise ValueError("Cell names differ between merged data and query input")
            # Same cells, wrong order -> reindex (nothing was appended by ad.concat)
            self.log_warning("Reordering predictions to match input cell order")
            reindex_map = pd.Series(range(len(query_cells_merged)), index=query_cells_merged)
            reorder_idx = reindex_map.loc[query_cells_input].values
            predictions = predictions[reorder_idx]
        
        self.log_info(f"Generated predictions for {len(predictions)} cells")
        return predictions

    def save_model(self) -> None:
        """
        Don't save these models, for now.
        
        ...
        """
        pass