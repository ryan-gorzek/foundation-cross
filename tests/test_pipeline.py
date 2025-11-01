"""
Tests for pipeline functionality.
"""
import pytest
import numpy as np
import anndata as ad
from pathlib import Path
import yaml

from src.utils import load_experiment_config, ConfigValidator


@pytest.fixture
def mock_experiment_config(tmp_path):
    """Create a mock experiment configuration."""
    # Create mock dataset configs
    dataset_dir = tmp_path / "configs" / "datasets"
    dataset_dir.mkdir(parents=True)
    
    mouse_config = {
        'name': 'mouse',
        'species': 'Mus musculus',
        'data_path': 'data/mouse.h5ad',
        'celltype_column': 'celltype',
        'preprocessing': {
            'min_genes': 5,
            'min_counts': 3,
            'normalize_total': 1e4,
            'log1p': False,
            'n_bins': 51,
        }
    }
    
    with open(dataset_dir / "mouse.yaml", 'w') as f:
        yaml.dump(mouse_config, f)
    
    opossum_config = mouse_config.copy()
    opossum_config['name'] = 'opossum'
    opossum_config['species'] = 'Monodelphis domestica'
    opossum_config['data_path'] = 'data/opossum.h5ad'
    
    with open(dataset_dir / "opossum.yaml", 'w') as f:
        yaml.dump(opossum_config, f)
    
    # Create mock model config
    model_dir = tmp_path / "configs" / "models"
    model_dir.mkdir(parents=True)
    
    model_config = {
        'name': 'scgpt',
        'type': 'neural_network',
        'architecture': {},
        'training': {},
        'tokenization': {},
    }
    
    with open(model_dir / "scgpt.yaml", 'w') as f:
        yaml.dump(model_config, f)
    
    # Create experiment config
    exp_dir = tmp_path / "configs" / "experiments"
    exp_dir.mkdir(parents=True)
    
    exp_config = {
        'name': 'test_experiment',
        'description': 'Test experiment',
        'reference': {
            'dataset': 'mouse',
            'labels': None,
        },
        'query': [
            {'dataset': 'opossum', 'labels': None}
        ],
        'model': 'scgpt',
        'model_config': str(model_dir / "scgpt.yaml"),
        'output': {
            'save_dir': 'results',
            'confusion_matrix': {
                'figsize': [10, 10],
            }
        },
        'training': {
            'seed': 0,
            'epochs': 1,
            'batch_size': 16,
        },
        'reproducibility': {
            'git_commit': None,
        }
    }
    
    exp_path = exp_dir / "test.yaml"
    with open(exp_path, 'w') as f:
        yaml.dump(exp_config, f)
    
    return exp_path, tmp_path


def test_config_validation():
    """Test configuration validation."""
    # Valid dataset config
    valid_dataset = {
        'name': 'test',
        'species': 'Test',
        'data_path': __file__,  # Use this file as it exists
        'celltype_column': 'celltype',
    }
    ConfigValidator.validate_dataset_config(valid_dataset)
    
    # Missing required field
    invalid_dataset = valid_dataset.copy()
    del invalid_dataset['celltype_column']
    
    with pytest.raises(ValueError):
        ConfigValidator.validate_dataset_config(invalid_dataset)


def test_experiment_config_validation():
    """Test experiment configuration validation."""
    valid_config = {
        'name': 'test',
        'reference': {'dataset': 'mouse'},
        'query': [{'dataset': 'opossum'}],
        'model': 'scgpt',
    }
    ConfigValidator.validate_experiment_config(valid_config)
    
    # Missing required field
    invalid_config = valid_config.copy()
    del invalid_config['model']
    
    with pytest.raises(ValueError):
        ConfigValidator.validate_experiment_config(invalid_config)


def test_model_config_validation():
    """Test model configuration validation."""
    valid_config = {
        'name': 'scgpt',
        'type': 'neural_network',
    }
    ConfigValidator.validate_model_config(valid_config)
    
    # Missing required field
    invalid_config = {'name': 'scgpt'}
    
    with pytest.raises(ValueError):
        ConfigValidator.validate_model_config(invalid_config)