"""
Configuration loading and validation utilities.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import hashlib
import json


class ConfigValidator:
    """Validate experiment configurations."""
    
    @staticmethod
    def validate_dataset_config(config: Dict[str, Any]) -> None:
        """Validate dataset configuration."""
        required = ['name', 'species', 'data_path', 'celltype_column']
        for field in required:
            if field not in config:
                raise ValueError(f"Dataset config missing required field: {field}")
        
        # Check data file exists
        data_path = Path(config['data_path'])
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
    
    @staticmethod
    def validate_experiment_config(config: Dict[str, Any]) -> None:
        """Validate experiment configuration."""
        required = ['name', 'reference', 'query', 'model']
        for field in required:
            if field not in config:
                raise ValueError(f"Experiment config missing required field: {field}")
        
        # Validate reference
        if 'dataset' not in config['reference']:
            raise ValueError("Reference must specify 'dataset'")
        
        # Validate query (can be list or single dict)
        queries = config['query']
        if not isinstance(queries, list):
            raise ValueError("Query must be a list of dataset specifications")
        
        for q in queries:
            if 'dataset' not in q:
                raise ValueError("Each query must specify 'dataset'")
    
    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> None:
        """Validate model configuration."""
        required = ['name', 'type']
        for field in required:
            if field not in config:
                raise ValueError(f"Model config missing required field: {field}")


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(config: Dict[str, Any], path: Path) -> None:
    """Save configuration to YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute hash of configuration for reproducibility."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def load_dataset_config(dataset_name: str, config_dir: Path = Path("configs/datasets")) -> Dict[str, Any]:
    """Load dataset configuration by name."""
    config_path = config_dir / f"{dataset_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    
    config = load_yaml(config_path)
    ConfigValidator.validate_dataset_config(config)
    return config


def load_model_config(model_name: str, config_dir: Path = Path("configs/models")) -> Dict[str, Any]:
    """Load model configuration by name."""
    config_path = config_dir / f"{model_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    
    config = load_yaml(config_path)
    ConfigValidator.validate_model_config(config)
    return config


def load_experiment_config(experiment_path: Path) -> Dict[str, Any]:
    """Load and validate full experiment configuration."""
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {experiment_path}")
    
    config = load_yaml(experiment_path)
    ConfigValidator.validate_experiment_config(config)
    
    # Load referenced configs
    config['_reference_dataset_config'] = load_dataset_config(config['reference']['dataset'])
    config['_query_dataset_configs'] = [
        load_dataset_config(q['dataset']) for q in config['query']
    ]
    
    # Load model config if path specified, otherwise load by name
    if 'model_config' in config:
        model_config_path = Path(config['model_config'])
        config['_model_config'] = load_yaml(model_config_path)
    else:
        config['_model_config'] = load_model_config(config['model'])
    
    ConfigValidator.validate_model_config(config['_model_config'])
    
    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configurations, with override taking precedence."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged