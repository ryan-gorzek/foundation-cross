"""
Label transfer models.
"""
from .base import BaseLabelTransferModel
from .scgpt import ScGPTModel
from .seurat import SeuratMapQuery

# Model registry
MODEL_REGISTRY = {
    'scgpt': ScGPTModel,
    'seurat_mapquery': SeuratMapQuery,
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
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(config, save_dir, logger)


__all__ = [
    'BaseLabelTransferModel',
    'ScGPTModel',
    'SeuratMapQuery',
    'MODEL_REGISTRY',
    'get_model',
]