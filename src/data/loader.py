"""
Data loading utilities for cross-species experiments.
"""
import numpy as np
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import Optional, List, Dict, Any
from scipy.sparse import issparse


class DatasetLoader:
    """Load and prepare datasets for label transfer experiments."""
    
    def __init__(self, dataset_config: Dict[str, Any], logger=None):
        self.config = dataset_config
        self.logger = logger
        self.name = dataset_config['name']
        self.data_path = Path(dataset_config['data_path'])
        self.celltype_column = dataset_config['celltype_column']
        self.preprocessing = dataset_config.get('preprocessing', {})
    
    def load(self, label_subset: Optional[List[str]] = None) -> ad.AnnData:
        """
        Load dataset from h5ad file.
        
        Parameters
        ----------
        label_subset : Optional[List[str]]
            If provided, filter to only these cell type labels
            
        Returns
        -------
        adata : AnnData
            Loaded and optionally filtered dataset
        """
        if self.logger:
            self.logger.info(f"Loading {self.name} data from {self.data_path}")
        
        adata = sc.read_h5ad(self.data_path)
        
        if self.logger:
            self.logger.info(f"  {adata.n_obs} cells, {adata.n_vars} genes")
        
        # Standardize celltype column name
        if self.celltype_column != 'celltype':
            if self.celltype_column not in adata.obs.columns:
                raise ValueError(
                    f"Celltype column '{self.celltype_column}' not found in data. "
                    f"Available columns: {list(adata.obs.columns)}"
                )
            adata.obs['celltype'] = adata.obs[self.celltype_column].copy()
        
        # Ensure celltype is categorical
        adata.obs['celltype'] = adata.obs['celltype'].astype('category')
        
        # Filter to label subset if specified
        if label_subset is not None:
            if self.logger:
                self.logger.info(f"  Filtering to {len(label_subset)} specified cell types")
            
            # Check which labels are available
            available_labels = set(adata.obs['celltype'].cat.categories)
            requested_labels = set(label_subset)
            missing_labels = requested_labels - available_labels
            
            if missing_labels:
                self.logger.warning(
                    f"  Requested labels not found in data: {sorted(missing_labels)}"
                )
            
            # Filter to cells with requested labels
            mask = adata.obs['celltype'].isin(label_subset)
            n_before = adata.n_obs
            adata = adata[mask].copy()
            n_after = adata.n_obs
            
            if self.logger:
                self.logger.info(f"  Kept {n_after}/{n_before} cells after label filtering")
            
            # Remove unused categories
            adata.obs['celltype'] = adata.obs['celltype'].cat.remove_unused_categories()
        
        # Ensure gene_name is in var
        if 'gene_name' not in adata.var.columns:
            adata.var['gene_name'] = adata.var_names.tolist()
        
        return adata
    
    def filter_cells_by_genes(self, adata: ad.AnnData, min_genes: int = 8) -> ad.AnnData:
        """
        Filter out cells with too few genes expressed.
        
        Parameters
        ----------
        adata : AnnData
            Input data
        min_genes : int
            Minimum number of genes that must be expressed
            
        Returns
        -------
        adata : AnnData
            Filtered data
        """
        if issparse(adata.X):
            n_genes = np.array((adata.X > 0).sum(axis=1)).flatten()
        else:
            n_genes = (adata.X > 0).sum(axis=1)
        
        keep = n_genes >= min_genes
        n_removed = (~keep).sum()
        
        if self.logger and n_removed > 0:
            self.logger.info(
                f"  Removing {n_removed} cells with < {min_genes} genes expressed"
            )
        
        return adata[keep].copy()
    
    def filter_zero_count_cells(self, adata: ad.AnnData) -> ad.AnnData:
        """Remove cells with zero total counts."""
        if issparse(adata.X):
            cell_counts = np.array(adata.X.sum(axis=1)).flatten()
        else:
            cell_counts = adata.X.sum(axis=1)
        
        keep = cell_counts > 0
        n_removed = (~keep).sum()
        
        if self.logger and n_removed > 0:
            self.logger.info(f"  Removing {n_removed} cells with zero counts")
        
        return adata[keep].copy()


def intersect_genes(
    reference: ad.AnnData,
    queries: List[ad.AnnData],
    logger=None
) -> tuple:
    """
    Find common genes across all datasets and subset to them.
    
    Parameters
    ----------
    reference : AnnData
        Reference dataset
    queries : List[AnnData]
        List of query datasets
    logger : Optional
        Logger instance
        
    Returns
    -------
    reference : AnnData
        Reference data subset to common genes
    queries : List[AnnData]
        Query datasets subset to common genes
    """
    # Start with reference genes
    common_genes = set(reference.var_names)
    
    # Intersect with each query
    for query in queries:
        common_genes = common_genes.intersection(query.var_names)
    
    common_genes = sorted(common_genes)
    
    if logger:
        logger.info(f"Found {len(common_genes)} common genes across all datasets")
    
    # Subset all datasets
    reference = reference[:, common_genes].copy()
    queries = [q[:, common_genes].copy() for q in queries]
    
    return reference, queries