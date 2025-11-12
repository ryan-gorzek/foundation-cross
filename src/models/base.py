"""
Abstract base class for label transfer models.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import anndata as ad


class BaseLabelTransferModel(ABC):
    """
    Abstract base class for all label transfer models.
    
    All models must implement train, predict, and save_model methods.
    Models handle their own preprocessing quirks and output to standard format.
    """
    
    def __init__(self, config: Dict[str, Any], save_dir: Path, logger=None):
        """
        Initialize the model.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Model configuration dictionary
        save_dir : Path
            Directory to save model outputs
        logger : Optional
            Logger instance
        """
        self.config = config
        self.save_dir = save_dir
        self.logger = logger
        
        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_outputs_dir = self.save_dir / "model_outputs"
        self.model_outputs_dir.mkdir(parents=True, exist_ok=True)
        
        if self.logger:
            self.logger.info(f"Initialized {self.config['name']} model")
            self.logger.info(f"  Output directory: {self.save_dir}")
    
    @abstractmethod
    def train(
        self,
        reference_data: ad.AnnData,
        query_data: Optional[ad.AnnData] = None,
        **kwargs
    ) -> None:
        """
        Train or fine-tune the model on reference data.
        
        Parameters
        ----------
        reference_data : AnnData
            Reference dataset with 'celltype_id' in obs
        query_data : Optional[AnnData]
            Query dataset (only used by models that integrate first)
        **kwargs
            Additional model-specific arguments
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        query_data: ad.AnnData,
        **kwargs
    ) -> np.ndarray:
        """
        Predict labels for query data.
        
        Parameters
        ----------
        query_data : AnnData
            Query dataset
        **kwargs
            Additional model-specific arguments
            
        Returns
        -------
        predictions : np.ndarray
            Predicted cell type IDs (integers)
        """
        pass
    
    @abstractmethod
    def save_model(self) -> None:
        """
        Save model artifacts (weights, vocabulary, etc.).
        
        Model-specific files should be saved to self.model_outputs_dir
        """
        pass
    
    def get_model_outputs_dir(self) -> Path:
        """Return directory for model-specific outputs."""
        return self.model_outputs_dir
    
    def log_info(self, message: str) -> None:
        """Log info message if logger available."""
        if self.logger:
            self.logger.info(message)
    
    def log_warning(self, message: str) -> None:
        """Log warning message if logger available."""
        if self.logger:
            self.logger.warning(message)
    
    def log_error(self, message: str) -> None:
        """Log error message if logger available."""
        if self.logger:
            self.logger.error(message)