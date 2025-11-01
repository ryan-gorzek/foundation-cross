"""
Preprocessing utilities for single-cell data.
"""
import numpy as np
import scanpy as sc
import anndata as ad
from typing import Dict, Any, Optional
from scipy.sparse import issparse


class SingleCellPreprocessor:
    """Preprocess single-cell data for model input."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """
        Initialize preprocessor with configuration.
        
        Parameters
        ----------
        config : Dict
            Preprocessing configuration with keys:
            - filter_cell_by_counts: min counts per cell
            - normalize_total: target sum for normalization
            - log1p: whether to apply log1p
            - n_bins: number of bins for expression values
        logger : Optional
            Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Set defaults
        self.filter_cell_by_counts = config.get('filter_cell_by_counts', 3)
        self.normalize_total = config.get('normalize_total', 1e4)
        self.log1p = config.get('log1p', False)
        self.n_bins = config.get('n_bins', 51)
    
    def preprocess(self, adata: ad.AnnData, batch_key: Optional[str] = None) -> ad.AnnData:
        """
        Apply preprocessing pipeline to data.
        
        Parameters
        ----------
        adata : AnnData
            Input data (will be modified in place)
        batch_key : Optional[str]
            Batch key for batch-aware processing
            
        Returns
        -------
        adata : AnnData
            Preprocessed data with new layers
        """
        if self.logger:
            self.logger.info("Preprocessing data...")
        
        # Store raw counts
        if 'X' not in adata.layers:
            adata.layers['X'] = adata.X.copy()
        
        # Filter cells by counts
        if self.filter_cell_by_counts > 0:
            sc.pp.filter_cells(adata, min_counts=self.filter_cell_by_counts)
            if self.logger:
                self.logger.info(f"  After filtering: {adata.n_obs} cells")
        
        # Normalize
        if self.logger:
            self.logger.info(f"  Normalizing to {self.normalize_total}")
        sc.pp.normalize_total(adata, target_sum=self.normalize_total)
        adata.layers['X_normed'] = adata.X.copy()
        
        # Log transform
        if self.log1p:
            if self.logger:
                self.logger.info("  Applying log1p transform")
            sc.pp.log1p(adata)
            adata.layers['X_log1p'] = adata.X.copy()
        
        # Bin expression values
        if self.n_bins > 0:
            if self.logger:
                self.logger.info(f"  Binning expression values into {self.n_bins} bins")
            adata.layers['X_binned'] = self._bin_expression(adata.X, self.n_bins)
        
        return adata
    
    def _bin_expression(self, X, n_bins: int) -> np.ndarray:
        """
        Bin expression values into discrete bins.
        
        Parameters
        ----------
        X : array-like
            Expression matrix
        n_bins : int
            Number of bins
            
        Returns
        -------
        X_binned : np.ndarray
            Binned expression values
        """
        if issparse(X):
            X_dense = X.A
        else:
            X_dense = X
        
        # Create bins from 0 to max value
        X_binned = np.zeros_like(X_dense, dtype=int)
        
        # Bin non-zero values
        nonzero_mask = X_dense > 0
        if nonzero_mask.any():
            nonzero_vals = X_dense[nonzero_mask]
            
            # Create bins from min to max of non-zero values
            min_val = nonzero_vals.min()
            max_val = nonzero_vals.max()
            
            # Edge case: all non-zero values are the same
            if min_val == max_val:
                X_binned[nonzero_mask] = 1
            else:
                # Bin into n_bins-1 bins (bin 0 is reserved for zeros)
                bins = np.linspace(min_val, max_val, n_bins)
                X_binned[nonzero_mask] = np.digitize(nonzero_vals, bins, right=True)
                
                # Ensure values are in [1, n_bins-1]
                X_binned[nonzero_mask] = np.clip(X_binned[nonzero_mask], 1, n_bins - 1)
        
        return X_binned


def prepare_for_transfer(
    reference: ad.AnnData,
    query: ad.AnnData,
    reference_labels: np.ndarray,
    logger=None
) -> tuple:
    """
    Prepare reference and query data for label transfer.
    
    Creates celltype_id mappings such that:
    - Reference cells get IDs [0, n_types-1]
    - Query cells get IDs based on reference categories
    - Query cells with labels not in reference get ID -1 (invalid)
    
    Parameters
    ----------
    reference : AnnData
        Reference dataset
    query : AnnData
        Query dataset
    reference_labels : np.ndarray
        Reference cell type labels
    logger : Optional
        Logger instance
        
    Returns
    -------
    reference : AnnData
        Reference with celltype_id added to obs
    query : AnnData
        Query with celltype_id and metadata added to obs
    metadata : dict
        Metadata about cell types and mappings
    """
    # Create categorical labels for reference
    reference.obs['celltype_id'] = reference_labels
    
    # Get unique cell types from reference
    train_categories = reference.obs['celltype'].cat.categories
    
    # Store original query labels
    query.obs['celltype_original'] = query.obs['celltype'].copy()
    
    # Set query categories to match reference
    query.obs['celltype'] = query.obs['celltype'].cat.set_categories(train_categories)
    query.obs['celltype_id'] = query.obs['celltype'].cat.codes.values
    
    # Mark which cells have valid labels for evaluation
    query.obs['has_valid_label'] = query.obs['celltype_id'] >= 0
    n_valid = query.obs['has_valid_label'].sum()
    n_invalid = (~query.obs['has_valid_label']).sum()
    
    if logger:
        logger.info(f"Label matching summary:")
        logger.info(f"  {n_valid} query cells have labels matching reference categories")
        logger.info(f"  {n_invalid} query cells have labels NOT in reference")
    
    # For cells without valid labels, set dummy ID to avoid errors
    query.obs.loc[~query.obs['has_valid_label'], 'celltype_id'] = 0
    
    # Create metadata
    id2type = dict(enumerate(train_categories))
    metadata = {
        'celltypes': train_categories.tolist(),
        'num_types': len(train_categories),
        'id2type': id2type,
        'n_valid_query_cells': int(n_valid),
        'n_invalid_query_cells': int(n_invalid),
    }
    
    return reference, query, metadata