"""
Seurat MapQuery model implementation for label transfer.
"""
import subprocess
import json
import pandas as pd
import numpy as np
import anndata as ad
from pathlib import Path
from typing import Dict, Any

from ..base import BaseLabelTransferModel


class SeuratMapQuery(BaseLabelTransferModel):
    """Seurat MapQuery for cross-species label transfer."""
    
    def __init__(self, config: Dict[str, Any], save_dir: Path, logger=None):
        super().__init__(config, save_dir, logger)
        
        # Seurat-specific config
        self.seurat_config = config.get('seurat', {})
        self.dims = self.seurat_config.get('dims', 30)
        self.k_anchor = self.seurat_config.get('k_anchor', 5)
        self.k_filter = self.seurat_config.get('k_filter', 200)
        self.k_weight = self.seurat_config.get('k_weight', 50)
        
        # Paths to R scripts
        self.script_dir = Path(__file__).parent
        self.convert_script = self.script_dir / "anndata_to_seurat.R"
        self.mapquery_script = self.script_dir / "run_mapquery.R"
        
        # Check if R scripts exist
        if not self.convert_script.exists():
            raise FileNotFoundError(f"R script not found: {self.convert_script}")
        if not self.mapquery_script.exists():
            raise FileNotFoundError(f"R script not found: {self.mapquery_script}")
        
        # Storage paths
        self.reference_h5ad = None
        self.reference_rds = None
        self.query_h5ad = None
        self.query_rds = None
    
    def _write_h5ad(self, adata: ad.AnnData, name: str) -> Path:
        """Write AnnData to H5AD file."""
        h5ad_path = self.model_outputs_dir / f"{name}.h5ad"
        adata.write_h5ad(h5ad_path)
        self.log_info(f"Wrote {name} data to {h5ad_path}")
        return h5ad_path
    
    def _convert_to_seurat(self, h5ad_path: Path, name: str) -> Path:
        """Convert H5AD to Seurat RDS format."""
        rds_path = self.model_outputs_dir / f"{name}.rds"
        
        self.log_info(f"Converting {name} H5AD to Seurat RDS...")
        
        cmd = [
            "Rscript",
            str(self.convert_script),
            "--input", str(h5ad_path),
            "--output", str(rds_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            self.log_info(result.stdout)
        except subprocess.CalledProcessError as e:
            self.log_error(f"R script failed with error:\n{e.stderr}")
            raise RuntimeError(f"Failed to convert {name} to Seurat format") from e
        
        if not rds_path.exists():
            raise RuntimeError(f"Conversion failed: {rds_path} not created")
        
        self.log_info(f"Conversion complete: {rds_path}")
        return rds_path
    
    def train(self, reference_data: ad.AnnData, **kwargs) -> None:
        """
        Prepare reference data for Seurat MapQuery.
        
        Note: Seurat MapQuery doesn't require explicit training,
        but we prepare the reference data here.
        
        Parameters
        ----------
        reference_data : AnnData
            Reference dataset with 'celltype' in obs
        """
        self.log_info("Preparing reference data for Seurat MapQuery")
        
        # Write reference to H5AD
        self.reference_h5ad = self._write_h5ad(reference_data, "reference")
        
        # Convert to Seurat RDS
        self.reference_rds = self._convert_to_seurat(self.reference_h5ad, "reference")
        
        self.log_info("Reference data preparation complete")
    
    def predict(self, query_data: ad.AnnData, **kwargs) -> np.ndarray:
        """
        Predict labels for query data using Seurat MapQuery.
        
        Parameters
        ----------
        query_data : AnnData
            Query dataset
            
        Returns
        -------
        predictions : np.ndarray
            Predicted cell type IDs (as integers matching reference categories)
        """
        self.log_info("Running Seurat MapQuery for label transfer")
        
        if self.reference_rds is None:
            raise RuntimeError("Reference data not prepared. Call train() first.")
        
        # Write query to H5AD
        self.query_h5ad = self._write_h5ad(query_data, "query")
        
        # Convert to Seurat RDS
        self.query_rds = self._convert_to_seurat(self.query_h5ad, "query")
        
        # Prepare config for R script
        seurat_config = {
            "dims": self.dims,
            "k_anchor": self.k_anchor,
            "k_filter": self.k_filter,
            "k_weight": self.k_weight,
            "celltype_column": "celltype"
        }
        
        config_path = self.model_outputs_dir / "seurat_config.json"
        with open(config_path, 'w') as f:
            json.dump(seurat_config, f, indent=2)
        
        # Run MapQuery R script
        self.log_info("Running MapQuery (this may take a few minutes)...")
        
        cmd = [
            "Rscript",
            str(self.mapquery_script),
            "--reference", str(self.reference_rds),
            "--query", str(self.query_rds),
            "--output", str(self.model_outputs_dir),
            "--config", str(config_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            self.log_info(result.stdout)
        except subprocess.CalledProcessError as e:
            self.log_error(f"MapQuery failed with error:\n{e.stderr}")
            raise RuntimeError("Seurat MapQuery failed") from e
        
        # Read predictions
        predictions_path = self.model_outputs_dir / "predictions.csv"
        if not predictions_path.exists():
            raise RuntimeError(f"Predictions file not found: {predictions_path}")
        
        predictions_df = pd.read_csv(predictions_path)
        self.log_info(f"Loaded predictions for {len(predictions_df)} cells")
        
        # Convert predicted labels to IDs matching reference categories
        # Get reference categories from query_data (which has been aligned to reference)
        if 'celltype' in query_data.obs.columns:
            reference_categories = query_data.obs['celltype'].cat.categories
        else:
            # Fallback: extract from predictions
            reference_categories = predictions_df['predicted_label'].unique()
        
        # Create mapping
        label_to_id = {label: i for i, label in enumerate(reference_categories)}
        
        # Convert predictions to IDs
        predicted_labels = predictions_df['predicted_label'].values
        predictions = np.array([
            label_to_id.get(label, -1) for label in predicted_labels
        ])
        
        # Store prediction scores for later analysis
        self.prediction_scores = predictions_df['prediction_score'].values
        
        self.log_info(f"Generated predictions for {len(predictions)} cells")
        return predictions
    
    def save_model(self) -> None:
        """
        Save model artifacts.
        
        For Seurat, this includes the RDS files and transfer metadata.
        """
        # Metadata is already saved by R script
        metadata_path = self.model_outputs_dir / "transfer_metadata.json"
        if metadata_path.exists():
            self.log_info(f"Model metadata saved at {metadata_path}")
        
        # Save prediction scores if available
        if hasattr(self, 'prediction_scores'):
            scores_path = self.model_outputs_dir / "prediction_scores.npy"
            np.save(scores_path, self.prediction_scores)
            self.log_info(f"Prediction scores saved to {scores_path}")