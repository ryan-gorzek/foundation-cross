"""
Data loading and preprocessing utilities for scGPT cross-species label transfer.
"""
import numpy as np
import scanpy as sc
from pathlib import Path
from typing import Tuple
import anndata as ad


def load_and_prepare_data(
    mouse_path: Path,
    opossum_path: Path,
    train_species: str = "mouse"
) -> Tuple[ad.AnnData, ad.AnnData, dict]:
    """
    Load mouse and opossum h5ad files and prepare them for training.
    
    Parameters
    ----------
    mouse_path : Path
        Path to mouse h5ad file
    opossum_path : Path
        Path to opossum h5ad file
    train_species : str
        Which species to use for training ("mouse" or "opossum")
        
    Returns
    -------
    adata_train : AnnData
        Training data (reference)
    adata_test : AnnData
        Test data (query)
    metadata : dict
        Dictionary containing celltypes, id mappings, etc.
    """
    print(f"Loading data from {mouse_path} and {opossum_path}")
    
    # Load data
    mouse = sc.read_h5ad(mouse_path)
    opossum = sc.read_h5ad(opossum_path)
    
    print(f"Mouse: {mouse.n_obs} cells, {mouse.n_vars} genes")
    print(f"Opossum: {opossum.n_obs} cells, {opossum.n_vars} genes")
    
    # Find common genes (already in ortholog space)
    common_genes = mouse.var_names.intersection(opossum.var_names)
    print(f"Common orthologous genes: {len(common_genes)}")
    
    # Subset to common genes
    mouse = mouse[:, common_genes].copy()
    opossum = opossum[:, common_genes].copy()
    
    # Set up training and test based on direction
    if train_species == "mouse":
        adata_train = mouse
        adata_test = opossum
        print("Training on mouse, testing on opossum")
    else:
        adata_train = opossum
        adata_test = mouse
        print("Training on opossum, testing on mouse")
    
    # Create celltype mappings
    # Check for 'Subclass' or 'celltype' field
    celltype_key = None
    for key in ['Subclass', 'celltype', 'cell_type']:
        if key in adata_train.obs.columns:
            celltype_key = key
            break
    
    if celltype_key is None:
        raise ValueError("No celltype annotation found. Expected 'Subclass', 'celltype', or 'cell_type' in obs")
    
    print(f"Using '{celltype_key}' as celltype annotation")
    
    # Standardize to 'celltype' column
    if celltype_key != 'celltype':
        adata_train.obs['celltype'] = adata_train.obs[celltype_key].astype('category')
        adata_test.obs['celltype'] = adata_test.obs[celltype_key].astype('category')
    
    # Create categorical IDs
    adata_train.obs['celltype'] = adata_train.obs['celltype'].astype('category')
    celltype_id_labels = adata_train.obs['celltype'].cat.codes.values
    adata_train.obs['celltype_id'] = celltype_id_labels
    
    # Map test data to same categories
    train_categories = adata_train.obs['celltype'].cat.categories
    adata_test.obs['celltype'] = adata_test.obs['celltype'].astype('category')
    adata_test.obs['celltype'] = adata_test.obs['celltype'].cat.set_categories(train_categories)
    adata_test.obs['celltype_id'] = adata_test.obs['celltype'].cat.codes.values
    
    # Create ID to type mapping
    id2type = dict(enumerate(train_categories))
    num_types = len(train_categories)
    
    # Create batch IDs
    adata_train.obs['batch_id'] = 0
    adata_train.obs['str_batch'] = '0'
    adata_test.obs['batch_id'] = 1
    adata_test.obs['str_batch'] = '1'
    
    # Ensure gene_name is in var
    adata_train.var['gene_name'] = adata_train.var_names.tolist()
    adata_test.var['gene_name'] = adata_test.var_names.tolist()
    
    # Store metadata
    metadata = {
        'celltypes': train_categories.tolist(),
        'num_types': num_types,
        'id2type': id2type,
        'celltype_key': celltype_key,
        'n_genes': len(common_genes),
        'train_species': train_species,
        'test_species': 'opossum' if train_species == 'mouse' else 'mouse'
    }
    
    print(f"\nData preparation complete:")
    print(f"  Training cells: {adata_train.n_obs}")
    print(f"  Test cells: {adata_test.n_obs}")
    print(f"  Cell types: {num_types}")
    print(f"  Genes: {len(common_genes)}")
    
    return adata_train, adata_test, metadata


def prepare_anndata(adata: ad.AnnData, adata_test: ad.AnnData) -> ad.AnnData:
    """
    Concatenate training and test data for preprocessing.
    
    Parameters
    ----------
    adata : AnnData
        Training data
    adata_test : AnnData
        Test data
        
    Returns
    -------
    adata_combined : AnnData
        Combined dataset
    """
    adata_combined = adata.concatenate(adata_test, batch_key='str_batch')
    return adata_combined
