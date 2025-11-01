"""
Main pipeline orchestration for cross-species label transfer.
"""
import time
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from .utils import (
    load_experiment_config,
    setup_logger,
    set_seed,
    get_git_commit,
    get_git_status,
    get_environment_info,
    compute_config_hash,
    save_yaml,
)
from .data import (
    DatasetLoader,
    intersect_genes,
    SingleCellPreprocessor,
    prepare_for_transfer,
    DataValidator,
)
from .models import get_model
from .analysis import (
    compute_metrics,
    compute_per_class_metrics,
    plot_confusion_matrix,
    get_classification_report,
    plot_per_class_f1,
)


class CrossSpeciesLabelTransferPipeline:
    """Main pipeline for cross-species label transfer experiments."""
    
    def __init__(self, experiment_config_path: Path):
        """
        Initialize pipeline with experiment configuration.
        
        Parameters
        ----------
        experiment_config_path : Path
            Path to experiment YAML config
        """
        # Load configuration
        self.config = load_experiment_config(experiment_config_path)
        self.experiment_name = self.config['name']
        
        # Setup output directory
        timestamp = time.strftime('%b%d-%H-%M')
        model_name = self.config['model']
        
        # Create directory: results/{reference}_{query}/{model}_{timestamp}
        ref_name = self.config['reference']['dataset']
        query_names = '_'.join([q['dataset'] for q in self.config['query']])
        
        output_config = self.config.get('output', {})
        save_dir_base = Path(output_config.get('save_dir', 'results'))
        
        self.save_dir = save_dir_base / f"{ref_name}_{query_names}" / f"{model_name}_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = setup_logger('pipeline', self.save_dir)
        self.logger.section("Cross-Species Label Transfer Pipeline")
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info(f"Output directory: {self.save_dir}")
        
        # Save configuration snapshot
        config_snapshot_path = self.save_dir / "config.yaml"
        save_yaml(self.config, config_snapshot_path)
        self.logger.info(f"Configuration saved to {config_snapshot_path}")
        
        # Reproducibility info
        self._log_reproducibility_info()
        
        # Initialize components
        self.reference_data = None
        self.query_datasets = []
        self.model = None
        self.metadata = {}
    
    def _log_reproducibility_info(self):
        """Log reproducibility information."""
        repro_config = self.config.get('reproducibility', {})
        
        # Git info
        git_commit = repro_config.get('git_commit', 'auto')
        if git_commit == 'auto':
            git_commit = get_git_commit()
        
        git_status = get_git_status()
        
        if git_commit:
            self.logger.info(f"Git commit: {git_commit}")
            if git_status == 'dirty':
                self.logger.warning("Git repository has uncommitted changes")
        
        # Environment info
        env_info = get_environment_info()
        self.logger.info(f"Python version: {env_info['python_version']}")
        self.logger.info(f"PyTorch version: {env_info['torch_version']}")
        self.logger.info(f"CUDA available: {env_info['cuda_available']}")
        
        # Config hash
        config_hash = compute_config_hash(self.config)
        self.logger.info(f"Config hash: {config_hash}")
        
        # Save to file
        repro_info = {
            'git_commit': git_commit,
            'git_status': git_status,
            'config_hash': config_hash,
            'environment': env_info,
        }
        
        with open(self.save_dir / "reproducibility.json", 'w') as f:
            json.dump(repro_info, f, indent=2)
    
    def load_datasets(self):
        """Load reference and query datasets."""
        self.logger.section("STEP 1: Loading Datasets")
        
        # Load reference
        ref_config = self.config['_reference_dataset_config']
        ref_labels = self.config['reference'].get('labels', None)
        
        ref_loader = DatasetLoader(ref_config, self.logger)
        self.reference_data = ref_loader.load(label_subset=ref_labels)
        
        # Validate reference
        DataValidator.validate_adata(self.reference_data, "Reference")
        DataValidator.validate_label_format(self.reference_data, "Reference")
        DataValidator.check_data_quality(self.reference_data, "Reference", self.logger)
        
        # Load queries
        self.query_datasets = []
        for i, query_spec in enumerate(self.config['query']):
            query_config = self.config['_query_dataset_configs'][i]
            query_labels = query_spec.get('labels', None)
            
            query_loader = DatasetLoader(query_config, self.logger)
            query_data = query_loader.load(label_subset=query_labels)
            
            # Validate query
            DataValidator.validate_adata(query_data, f"Query {i}")
            DataValidator.validate_label_format(query_data, f"Query {i}")
            DataValidator.check_data_quality(query_data, f"Query {i}", self.logger)
            
            self.query_datasets.append({
                'data': query_data,
                'name': query_config['name'],
                'config': query_spec,
            })
        
        # Find common genes
        self.logger.info("\nFinding common genes across datasets...")
        query_data_list = [q['data'] for q in self.query_datasets]
        self.reference_data, query_data_list = intersect_genes(
            self.reference_data,
            query_data_list,
            self.logger
        )
        
        # Update query datasets
        for i, query_data in enumerate(query_data_list):
            self.query_datasets[i]['data'] = query_data
        
        # Validate gene overlap
        DataValidator.validate_gene_overlap(
            self.reference_data,
            query_data_list,
            min_overlap=0.5
        )
    
    def preprocess_datasets(self):
        """Preprocess datasets."""
        self.logger.section("STEP 2: Preprocessing Datasets")
        
        # Get preprocessing config
        preproc_config = self.config['_reference_dataset_config'].get('preprocessing', {})
        
        # Preprocess reference
        self.logger.info("Preprocessing reference data...")
        ref_loader = DatasetLoader(self.config['_reference_dataset_config'], self.logger)
        self.reference_data = ref_loader.filter_zero_count_cells(self.reference_data)
        self.reference_data = ref_loader.filter_cells_by_genes(
            self.reference_data,
            min_genes=preproc_config.get('min_genes', 8)
        )
        
        preprocessor = SingleCellPreprocessor(preproc_config, self.logger)
        self.reference_data = preprocessor.preprocess(self.reference_data)
        
        # Preprocess queries (separately to avoid information leakage)
        for i, query_dict in enumerate(self.query_datasets):
            self.logger.info(f"\nPreprocessing query data: {query_dict['name']}...")
            query_config = self.config['_query_dataset_configs'][i]
            query_preproc_config = query_config.get('preprocessing', preproc_config)
            
            query_loader = DatasetLoader(query_config, self.logger)
            query_data = query_loader.filter_zero_count_cells(query_dict['data'])
            query_data = query_loader.filter_cells_by_genes(
                query_data,
                min_genes=query_preproc_config.get('min_genes', 8)
            )
            
            query_preprocessor = SingleCellPreprocessor(query_preproc_config, self.logger)
            query_data = query_preprocessor.preprocess(query_data)
            
            self.query_datasets[i]['data'] = query_data
    
    def prepare_for_training(self):
        """Prepare data for model training."""
        self.logger.section("STEP 3: Preparing Data for Training")
        
        # Create celltype_id for reference
        self.reference_data.obs['celltype_id'] = self.reference_data.obs['celltype'].cat.codes
        
        # Add batch IDs
        self.reference_data.obs['batch_id'] = 0
        
        # Prepare each query
        for i, query_dict in enumerate(self.query_datasets):
            query_data = query_dict['data']
            query_data.obs['batch_id'] = i + 1
            
            # Prepare for transfer (align labels)
            _, query_data, metadata = prepare_for_transfer(
                self.reference_data,
                query_data,
                self.reference_data.obs['celltype_id'].values,
                self.logger
            )
            
            self.query_datasets[i]['data'] = query_data
            self.query_datasets[i]['metadata'] = metadata
    
    def initialize_model(self):
        """Initialize the model."""
        self.logger.section("STEP 4: Initializing Model")
        
        model_name = self.config['model']
        model_config = self.config['_model_config']
        
        self.model = get_model(model_name, model_config, self.save_dir, self.logger)
    
    def train_model(self):
        """Train the model."""
        self.logger.section("STEP 5: Training Model")
        
        # Get training config
        training_config = self.config.get('training', {})
        
        # Set seed
        seed = training_config.get('seed', 0)
        set_seed(seed)
        self.logger.info(f"Random seed: {seed}")
        
        # Train
        self.model.train(
            self.reference_data,
            **training_config
        )
        
        # Save model
        self.model.save_model()
    
    def evaluate_model(self):
        """Evaluate model on query datasets."""
        self.logger.section("STEP 6: Evaluating Model")
        
        all_results = []
        
        for query_dict in self.query_datasets:
            query_name = query_dict['name']
            query_data = query_dict['data']
            metadata = query_dict['metadata']
            
            self.logger.subsection(f"Evaluating on {query_name}")
            
            # Predict
            predictions = self.model.predict(query_data, **self.config.get('training', {}))
            
            # Get true labels
            true_labels = query_data.obs['celltype_id'].values
            valid_mask = query_data.obs.get('has_valid_label', np.ones(len(true_labels), dtype=bool)).values
            
            # Compute metrics
            metrics = compute_metrics(predictions, true_labels, valid_mask)
            
            self.logger.info(f"Results ({valid_mask.sum()} cells with valid labels):")
            self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            self.logger.info(f"  Precision (macro): {metrics['precision_macro']:.4f}")
            self.logger.info(f"  Recall (macro): {metrics['recall_macro']:.4f}")
            self.logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
            
            # Per-class metrics
            label_names = metadata['celltypes']
            per_class_metrics = compute_per_class_metrics(
                predictions, true_labels, label_names, valid_mask
            )
            
            # Save results
            results = {
                'query_name': query_name,
                'predictions': predictions,
                'true_labels': true_labels,
                'valid_mask': valid_mask,
                'metrics': metrics,
                'per_class_metrics': per_class_metrics,
                'metadata': metadata,
            }
            all_results.append(results)
            
            # Save predictions to CSV
            self._save_predictions(query_data, predictions, query_name, metadata)
            
            # Save metrics
            self._save_metrics(metrics, per_class_metrics, query_name)
            
            # Generate visualizations
            self._generate_visualizations(results, query_name)
        
        return all_results
    
    def _save_predictions(self, query_data, predictions, query_name, metadata):
        """Save predictions to CSV."""
        id2type = metadata['id2type']
        
        predictions_df = pd.DataFrame({
            'cell_id': query_data.obs_names,
            'true_label': query_data.obs.get('celltype_original', query_data.obs['celltype']),
            'true_label_id': query_data.obs['celltype_id'],
            'predicted_label': [id2type.get(p, f'Unknown_{p}') for p in predictions],
            'predicted_label_id': predictions,
            'has_valid_label': query_data.obs.get('has_valid_label', True),
        })
        
        csv_path = self.save_dir / f"predictions_{query_name}.csv"
        predictions_df.to_csv(csv_path, index=False)
        self.logger.info(f"Predictions saved to {csv_path}")
    
    def _save_metrics(self, metrics, per_class_metrics, query_name):
        """Save metrics to JSON."""
        all_metrics = {
            'overall': metrics,
            'per_class': per_class_metrics,
        }
        
        metrics_path = self.save_dir / f"metrics_{query_name}.json"
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        self.logger.info(f"Metrics saved to {metrics_path}")
    
    def _generate_visualizations(self, results, query_name):
        """Generate visualizations."""
        self.logger.info("Generating visualizations...")
        
        predictions = results['predictions']
        true_labels = results['true_labels']
        valid_mask = results['valid_mask']
        metadata = results['metadata']
        
        # Get label names
        true_label_names = metadata['celltypes']
        pred_label_names = metadata['celltypes']
        
        # Get ordering from config
        output_config = self.config.get('output', {})
        cm_config = output_config.get('confusion_matrix', {})
        figsize = tuple(cm_config.get('figsize', [14, 12]))
        row_order = cm_config.get('row_order', None)
        col_order = cm_config.get('col_order', None)
        
        # Confusion matrix
        cm_path = self.save_dir / f"confusion_matrix_{query_name}.png"
        plot_confusion_matrix(
            predictions=predictions,
            labels=true_labels,
            true_label_names=true_label_names,
            pred_label_names=pred_label_names,
            save_path=cm_path,
            title=f"Cross-Species Label Transfer: {query_name}",
            figsize=figsize,
            row_order=row_order,
            col_order=col_order,
            valid_mask=valid_mask,
        )
        self.logger.info(f"Confusion matrix saved to {cm_path}")
        
        # Per-class F1 scores
        per_class_f1_path = self.save_dir / f"per_class_f1_{query_name}.png"
        plot_per_class_f1(
            results['per_class_metrics'],
            per_class_f1_path,
            title=f"Per-Class F1 Scores: {query_name}",
        )
        self.logger.info(f"Per-class F1 plot saved to {per_class_f1_path}")
    
    def run(self):
        """Run the complete pipeline."""
        start_time = time.time()
        
        try:
            self.load_datasets()
            self.preprocess_datasets()
            self.prepare_for_training()
            self.initialize_model()
            self.train_model()
            results = self.evaluate_model()
            
            elapsed = time.time() - start_time
            self.logger.section("PIPELINE COMPLETE")
            self.logger.info(f"Total time: {elapsed:.2f} seconds")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise