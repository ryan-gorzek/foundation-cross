"""
Label transfer models.
"""
from .base import BaseLabelTransferModel

# Model registry (string references, not imported classes)
MODEL_REGISTRY = {
    'scgpt': 'scgpt',
    'seurat_mapquery': 'seurat',
    'scvi': 'scvi'
}


def get_model(model_name: str, config: dict, save_dir, logger=None):
    """
    Get model instance by name.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    config : dict
        Model configuration
    save_dir : Path
        Directory to save model outputs
    logger : Optional
        Logger instance
        
    Returns
    -------
    model : BaseLabelTransferModel
        Model instance
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )
    
    # Lazy import based on model name
    if model_name == 'scgpt':
        from .scgpt import ScGPTModel
        return ScGPTModel(config, save_dir, logger)
    elif model_name == 'seurat_mapquery':
        from .seurat import SeuratMapQuery
        return SeuratMapQuery(config, save_dir, logger)
    elif model_name == 'scvi':
        from .scvi import ScVIModel
        return ScVIModel(config, save_dir, logger)
    else:
        raise ValueError(f"Unknown model: {model_name}")


__all__ = [
    'BaseLabelTransferModel',
    'MODEL_REGISTRY',
    'get_model',
]
