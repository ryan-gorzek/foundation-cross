"""
Tests for data loading functionality.
"""
import pytest
import numpy as np
import anndata as ad
from pathlib import Path
import tempfile

from src.data import DatasetLoader, intersect_genes
from src.utils import DataValidator


@pytest.fixture
def mock_dataset_config():
    """Create a mock dataset configuration."""
    return {
        'name': 'test_dataset',
        'species': 'Test species',
        'data_path': 'test_data.h5ad',
        'celltype_column': 'celltype',
        'preprocessing': {
            'min_genes': 5,
            'min_counts': 3,
            'normalize_total': 1e4,
            'log1p': False,
            'n_bins': 51,
        }
    }


@pytest.fixture
def mock_adata():
    """Create a mock AnnData object."""
    n_obs = 100
    n_vars = 50
    
    X = np.random.rand(n_obs, n_vars)
    obs = {
        'celltype': ['type_A'] * 50 + ['type_B'] * 50
    }
    var = {
        'gene_name': [f'gene_{i}' for i in range(n_vars)]
    }
    
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obs['celltype'] = adata.obs['celltype'].astype('category')
    
    return adata


def test_dataset_loader_initialization(mock_dataset_config):
    """Test DatasetLoader initialization."""
    loader = DatasetLoader(mock_dataset_config)
    assert loader.name == 'test_dataset'
    assert loader.celltype_column == 'celltype'


def test_filter_cells_by_genes(mock_dataset_config, mock_adata):
    """Test filtering cells by gene count."""
    loader = DatasetLoader(mock_dataset_config)
    
    # Set some cells to have few genes
    mock_adata.X[:10, :] = 0
    
    filtered = loader.filter_cells_by_genes(mock_adata, min_genes=5)
    assert filtered.n_obs < mock_adata.n_obs


def test_filter_zero_count_cells(mock_dataset_config, mock_adata):
    """Test filtering zero-count cells."""
    loader = DatasetLoader(mock_dataset_config)
    
    # Set some cells to zero
    mock_adata.X[:5, :] = 0
    
    filtered = loader.filter_zero_count_cells(mock_adata)
    assert filtered.n_obs == mock_adata.n_obs - 5


def test_intersect_genes():
    """Test gene intersection across datasets."""
    # Create two datasets with overlapping genes
    genes1 = [f'gene_{i}' for i in range(100)]
    genes2 = [f'gene_{i}' for i in range(50, 150)]
    
    adata1 = ad.AnnData(X=np.random.rand(50, 100))
    adata1.var_names = genes1
    adata1.var['gene_name'] = genes1
    
    adata2 = ad.AnnData(X=np.random.rand(50, 100))
    adata2.var_names = genes2
    adata2.var['gene_name'] = genes2
    
    ref, queries = intersect_genes(adata1, [adata2])
    
    # Should have 50 overlapping genes (50-99)
    assert ref.n_vars == 50
    assert queries[0].n_vars == 50
    assert set(ref.var_names) == set(queries[0].var_names)


def test_data_validator(mock_adata):
    """Test data validation."""
    # Valid data should pass
    DataValidator.validate_adata(mock_adata, "test")
    DataValidator.validate_label_format(mock_adata, "test")
    
    # Invalid data should raise
    invalid_adata = mock_adata.copy()
    del invalid_adata.obs['celltype']
    
    with pytest.raises(ValueError):
        DataValidator.validate_adata(invalid_adata, "test")


def test_label_subset_filtering(mock_dataset_config, mock_adata, tmp_path):
    """Test filtering to label subset."""
    # Save mock data
    h5ad_path = tmp_path / "test.h5ad"
    mock_adata.write_h5ad(h5ad_path)
    
    # Update config with real path
    mock_dataset_config['data_path'] = str(h5ad_path)
    
    loader = DatasetLoader(mock_dataset_config)
    
    # Load with label subset
    adata_subset = loader.load(label_subset=['type_A'])
    
    assert adata_subset.n_obs == 50
    assert len(adata_subset.obs['celltype'].cat.categories) == 1