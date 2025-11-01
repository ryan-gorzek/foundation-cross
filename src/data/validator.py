"""
Data validation utilities.
"""
import numpy as np
import anndata as ad
from typing import List, Optional


class DataValidator:
    """Validate datasets for label transfer experiments."""
    
    @staticmethod
    def validate_adata(adata: ad.AnnData, name: str = "dataset") -> None:
        """
        Validate basic AnnData structure.
        
        Parameters
        ----------
        adata : AnnData
            Dataset to validate
        name : str
            Name for error messages
        """
        if adata.n_obs == 0:
            raise ValueError(f"{name} has no cells")
        
        if adata.n_vars == 0:
            raise ValueError(f"{name} has no genes")
        
        if 'celltype' not in adata.obs.columns:
            raise ValueError(f"{name} missing 'celltype' column in obs")
        
        if adata.X is None:
            raise ValueError(f"{name} has no expression matrix (X is None)")
    
    @staticmethod
    def validate_gene_overlap(
        reference: ad.AnnData,
        queries: List[ad.AnnData],
        min_overlap: float = 0.5
    ) -> None:
        """
        Validate that datasets have sufficient gene overlap.
        
        Parameters
        ----------
        reference : AnnData
            Reference dataset
        queries : List[AnnData]
            Query datasets
        min_overlap : float
            Minimum fraction of reference genes that must overlap
        """
        ref_genes = set(reference.var_names)
        
        for i, query in enumerate(queries):
            query_genes = set(query.var_names)
            overlap = ref_genes.intersection(query_genes)
            overlap_frac = len(overlap) / len(ref_genes)
            
            if overlap_frac < min_overlap:
                raise ValueError(
                    f"Query dataset {i} has insufficient gene overlap with reference: "
                    f"{overlap_frac:.2%} (minimum: {min_overlap:.2%})"
                )
    
    @staticmethod
    def validate_label_format(adata: ad.AnnData, name: str = "dataset") -> None:
        """
        Validate cell type label format.
        
        Parameters
        ----------
        adata : AnnData
            Dataset to validate
        name : str
            Name for error messages
        """
        if not isinstance(adata.obs['celltype'].dtype.name, str) or \
           adata.obs['celltype'].dtype.name != 'category':
            raise ValueError(
                f"{name} celltype column must be categorical, "
                f"got {adata.obs['celltype'].dtype}"
            )
        
        # Check for NaN labels
        if adata.obs['celltype'].isna().any():
            n_na = adata.obs['celltype'].isna().sum()
            raise ValueError(f"{name} has {n_na} cells with NaN cell type labels")
        
        # Check for empty categories
        if len(adata.obs['celltype'].cat.categories) == 0:
            raise ValueError(f"{name} has no cell type categories")
    
    @staticmethod
    def check_data_quality(adata: ad.AnnData, name: str = "dataset", logger=None) -> None:
        """
        Check data quality and log warnings.
        
        Parameters
        ----------
        adata : AnnData
            Dataset to check
        name : str
            Name for messages
        logger : Optional
            Logger instance
        """
        from scipy.sparse import issparse
        
        # Check for zero-count cells
        if issparse(adata.X):
            cell_counts = np.array(adata.X.sum(axis=1)).flatten()
        else:
            cell_counts = adata.X.sum(axis=1)
        
        n_zero = (cell_counts == 0).sum()
        if n_zero > 0 and logger:
            logger.warning(f"{name} has {n_zero} cells with zero counts")
        
        # Check for zero-expression genes
        if issparse(adata.X):
            gene_counts = np.array(adata.X.sum(axis=0)).flatten()
        else:
            gene_counts = adata.X.sum(axis=0)
        
        n_zero_genes = (gene_counts == 0).sum()
        if n_zero_genes > 0 and logger:
            logger.warning(f"{name} has {n_zero_genes} genes with zero expression")
        
        # Check label distribution
        label_counts = adata.obs['celltype'].value_counts()
        min_cells_per_label = label_counts.min()
        if min_cells_per_label < 10 and logger:
            logger.warning(
                f"{name} has cell types with very few cells (min: {min_cells_per_label})"
            )