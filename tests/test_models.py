"""
Tests for model functionality.
"""
import pytest
import numpy as np
import anndata as ad
from pathlib import Path
import tempfile

from src.models import get_model, MODEL_REGISTRY
from src.models.base import BaseLabelTransferModel


@pytest.fixture
def mock_adata():
    """Create mock training data."""
    n_obs = 100
    n_vars = 50
    
    X = np.random.rand(n_obs, n_vars)
    obs = {
        'celltype': ['type_A'] * 50 + ['type_B'] * 50,
        'celltype_id': [0] * 50 + [1] * 50,
        'batch_id': [0] * n_obs,
    }
    var = {
        'gene_name': [f'GENE{i}' for i in range(n_vars)]
    }
    
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obs['celltype'] = adata.obs['celltype'].astype('category')
    
    # Add required layers for scGPT
    adata.layers['X_binned'] = np.random.randint(0, 51, size=(n_obs, n_vars))
    
    return adata


def test_model_registry():
    """Test that all models are registered."""
    assert 'scgpt' in MODEL_REGISTRY
    assert 'seurat_mapquery' in MODEL_REGISTRY


def test_get_model(tmp_path):
    """Test model retrieval from registry."""
    config = {
        'name': 'scgpt',
        'type': 'neural_network',
        'architecture': {},
        'training': {},
        'tokenization': {},
    }
    
    model = get_model('scgpt', config, tmp_path)
    assert isinstance(model, BaseLabelTransferModel)


def test_unknown_model(tmp_path):
    """Test that unknown model raises error."""
    with pytest.raises(ValueError):
        get_model('unknown_model', {}, tmp_path)


def test_base_model_interface(tmp_path):
    """Test that base model has required interface."""
    config = {'name': 'test', 'type': 'test'}
    
    # Create a minimal concrete implementation
    class TestModel(BaseLabelTransferModel):
        def train(self, reference_data, **kwargs):
            pass
        
        def predict(self, query_data, **kwargs):
            return np.zeros(query_data.n_obs, dtype=int)
        
        def save_model(self):
            pass
    
    model = TestModel(config, tmp_path)
    
    # Check required methods exist
    assert hasattr(model, 'train')
    assert hasattr(model, 'predict')
    assert hasattr(model, 'save_model')
    assert hasattr(model, 'get_model_outputs_dir')


def test_model_output_directory_creation(tmp_path):
    """Test that model creates output directories."""
    config = {'name': 'test', 'type': 'test'}
    
    class TestModel(BaseLabelTransferModel):
        def train(self, reference_data, **kwargs):
            pass
        def predict(self, query_data, **kwargs):
            return np.zeros(query_data.n_obs, dtype=int)
        def save_model(self):
            pass
    
    model = TestModel(config, tmp_path)
    
    assert model.save_dir.exists()
    assert model.model_outputs_dir.exists()