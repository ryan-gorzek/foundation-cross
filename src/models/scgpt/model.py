"""
scGPT model implementation for label transfer.
"""
import os
import json
import copy
import shutil
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any
import anndata as ad
from scipy.sparse import issparse
from torch.utils.data import DataLoader

# scGPT imports
from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor

from ..base import BaseLabelTransferModel
from .utils import SeqDataset, prepare_train_valid_split, prepare_data_loaders
from .train import train_epoch, evaluate


class ScGPTModel(BaseLabelTransferModel):
    """scGPT model for cross-species label transfer."""
    
    def __init__(self, config: Dict[str, Any], save_dir: Path, logger=None):
        super().__init__(config, save_dir, logger)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_info(f"Using device: {self.device}")
        
        # Model components (initialized during train)
        self.vocab = None
        self.model = None
        
        # Special tokens
        self.pad_token = "<pad>"
        self.special_tokens = [self.pad_token, "<cls>", "<eoc>"]
        
        # Extract config values
        self.pretrained_path = config.get('pretrained', {}).get('path', None)
        self.architecture = config.get('architecture', {})
        self.training_config = config.get('training', {})
        self.tokenization = config.get('tokenization', {})
        
    def _load_pretrained_vocab(self) -> GeneVocab:
        """Load pre-trained vocabulary."""
        if self.pretrained_path is None:
            return None
        
        pretrained_dir = Path(self.pretrained_path)
        vocab_file = pretrained_dir / "vocab.json"
        
        if not vocab_file.exists():
            self.log_warning(f"Vocab file not found: {vocab_file}")
            return None
        
        vocab = GeneVocab.from_file(vocab_file)
        for s in self.special_tokens:
            if s not in vocab:
                vocab.append_token(s)
        
        # Copy vocab to save directory
        shutil.copy(vocab_file, self.save_dir / "vocab.json")
        self.log_info(f"Loaded pre-trained vocabulary from {vocab_file}")
        
        return vocab
    
    def _build_vocab_from_data(self, adata: ad.AnnData) -> GeneVocab:
        """Build vocabulary from data genes."""
        genes = adata.var["gene_name"].tolist()
        vocab = GeneVocab.from_default(genes + self.special_tokens)
        vocab.save_json(self.save_dir / "vocab.json")
        self.log_info("Built vocabulary from data")
        return vocab
    
    def _match_genes_to_vocab(self, adata: ad.AnnData) -> ad.AnnData:
        """Match data genes to vocabulary."""
        # Convert gene names to uppercase to match pretrained vocabulary
        self.log_info("Converting gene names to uppercase to match vocabulary")
        adata.var["gene_name_original"] = adata.var["gene_name"].copy()
        adata.var["gene_name"] = adata.var["gene_name"].str.upper()
        
        # Match genes to vocabulary
        adata.var["id_in_vocab"] = [
            1 if gene in self.vocab else -1 for gene in adata.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        self.log_info(
            f"Matched {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(self.vocab)}"
        )
        
        # Subset to genes in vocabulary
        adata = adata[:, adata.var["id_in_vocab"] >= 0].copy()
        return adata
    
    def _load_pretrained_weights(self, model: nn.Module) -> nn.Module:
        """Load pre-trained model weights."""
        if self.pretrained_path is None:
            return model
        
        pretrained_dir = Path(self.pretrained_path)
        model_file = pretrained_dir / "best_model.pt"
        
        if not model_file.exists():
            self.log_warning(f"Pretrained weights not found: {model_file}")
            return model
        
        try:
            model.load_state_dict(torch.load(model_file, map_location=self.device))
            self.log_info(f"Loaded pre-trained weights from {model_file}")
        except:
            self.log_warning("Failed to load all weights. Loading compatible weights only.")
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_file, map_location=self.device)
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            self.log_info(f"Loaded {len(pretrained_dict)} compatible weight tensors")
        
        return model
    
    def _preprocess_data(self, adata: ad.AnnData) -> ad.AnnData:
        """Preprocess data for scGPT - creates fresh preprocessor each time."""
        # Remove zero-count cells
        if issparse(adata.X):
            cell_counts = np.array(adata.X.sum(axis=1)).flatten()
        else:
            cell_counts = adata.X.sum(axis=1)
        
        n_before = adata.n_obs
        adata = adata[cell_counts > 0].copy()
        n_removed = n_before - adata.n_obs
        
        if n_removed > 0:
            self.log_info(f"Filtered {n_removed} zero-count cells")
        
        # Create a FRESH preprocessor each time to avoid state leakage between datasets
        preprocessor = Preprocessor(
            use_key="X",
            filter_gene_by_counts=False,
            filter_cell_by_counts=3,
            normalize_total=1e4,
            result_normed_key="X_normed",
            log1p=False,
            result_log1p_key="X_log1p",
            subset_hvg=False,
            binning=self.architecture.get('n_bins', 51),
            result_binned_key="X_binned",
        )
        
        preprocessor(adata, batch_key=None)
        return adata
    
    def train(self, reference_data: ad.AnnData, **kwargs) -> None:
        """
        Train scGPT model on reference data.
        
        Parameters
        ----------
        reference_data : AnnData
            Reference dataset with 'celltype_id' in obs
        """
        self.log_info("Training scGPT model")
        
        # Load or build vocabulary
        self.vocab = self._load_pretrained_vocab()
        if self.vocab is None:
            self.vocab = self._build_vocab_from_data(reference_data)
        else:
            reference_data = self._match_genes_to_vocab(reference_data)
        
        self.vocab.set_default_index(self.vocab[self.pad_token])
        
        # Preprocess data
        reference_data = self._preprocess_data(reference_data)
        
        # Get training data
        input_layer_key = "X_binned"
        all_counts = (
            reference_data.layers[input_layer_key].A
            if issparse(reference_data.layers[input_layer_key])
            else reference_data.layers[input_layer_key]
        )
        
        genes = reference_data.var["gene_name"].tolist()
        self.gene_ids = np.array(self.vocab(genes), dtype=int)
        self.training_genes = genes  # Save for prediction matching
        
        celltypes_labels = reference_data.obs["celltype_id"].values
        batch_ids = reference_data.obs.get("batch_id", np.zeros(len(celltypes_labels))).values
        
        # Train/validation split
        (
            train_data,
            valid_data,
            train_celltype_labels,
            valid_celltype_labels,
            train_batch_labels,
            valid_batch_labels,
        ) = prepare_train_valid_split(
            all_counts, 
            celltypes_labels, 
            batch_ids, 
            test_size=0.1,
            random_state=kwargs.get('seed', 0)
        )
        
        self.log_info(f"Training samples: {len(train_data)}")
        self.log_info(f"Validation samples: {len(valid_data)}")
        
        # Tokenize
        mask_value = -1
        pad_value = -2
        max_seq_len = self.architecture.get('max_seq_len', 3001)
        include_zero_gene = self.tokenization.get('include_zero_gene', False)
        
        tokenized_train = tokenize_and_pad_batch(
            train_data,
            self.gene_ids,
            max_len=max_seq_len,
            vocab=self.vocab,
            pad_token=self.pad_token,
            pad_value=pad_value,
            append_cls=True,
            include_zero_gene=include_zero_gene,
        )
        
        tokenized_valid = tokenize_and_pad_batch(
            valid_data,
            self.gene_ids,
            max_len=max_seq_len,
            vocab=self.vocab,
            pad_token=self.pad_token,
            pad_value=pad_value,
            append_cls=True,
            include_zero_gene=include_zero_gene,
        )
        
        # Prepare data tensors
        train_data_pt = {
            "gene_ids": tokenized_train["genes"],
            "values": tokenized_train["values"],
            "target_values": tokenized_train["values"],
            "batch_labels": torch.from_numpy(train_batch_labels).long(),
            "celltype_labels": torch.from_numpy(train_celltype_labels).long(),
        }
        
        valid_data_pt = {
            "gene_ids": tokenized_valid["genes"],
            "values": tokenized_valid["values"],
            "target_values": tokenized_valid["values"],
            "batch_labels": torch.from_numpy(valid_batch_labels).long(),
            "celltype_labels": torch.from_numpy(valid_celltype_labels).long(),
        }
        
        # Create dataloaders
        batch_size = kwargs.get('batch_size', 16)
        num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)
        train_loader, valid_loader = prepare_data_loaders(
            train_data_pt, valid_data_pt, batch_size, num_workers
        )
        
        # Initialize model
        num_types = len(reference_data.obs['celltype'].cat.categories)
        self._initialize_model(num_types, kwargs)
        
        # Training setup
        criterion_cls = nn.CrossEntropyLoss()
        lr = self.training_config.get('learning_rate', 1e-4)
        amp = self.training_config.get('amp', True)
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(lr),
            eps=1e-4 if amp else 1e-8
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=self.training_config.get('schedule_ratio', 0.9)
        )
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        
        # Training loop
        epochs = kwargs.get('epochs', 10)
        log_interval = kwargs.get('log_interval', 100)
        best_val_loss = float("inf")
        best_model = None
        
        for epoch in range(1, epochs + 1):
            train_loss, train_error = train_epoch(
                self.model, train_loader, criterion_cls, optimizer, scaler,
                self.device, self.vocab, self.pad_token, epoch, log_interval,
                self.logger, amp, False, scheduler
            )
            
            val_loss, val_error = evaluate(
                self.model, valid_loader, criterion_cls, self.device,
                self.vocab, self.pad_token, amp, False
            )
            
            self.log_info(
                f"Epoch {epoch:3d} | train loss {train_loss:5.4f} | "
                f"valid loss {val_loss:5.4f} | valid error {val_error:5.4f}"
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.model)
                self.log_info(f"New best model at epoch {epoch}")
            
            scheduler.step()
        
        self.model = best_model
        self.log_info("Training complete")
    
    def _initialize_model(self, num_types: int, kwargs: dict) -> None:
        """Initialize the transformer model."""
        embsize = self.architecture.get('layer_size', 128)
        nhead = self.architecture.get('n_heads', 4)
        d_hid = self.architecture.get('layer_size', 128)
        nlayers = self.architecture.get('n_layers', 4)
        dropout = self.architecture.get('dropout', 0.2)
        n_bins = self.architecture.get('n_bins', 51)
        fast_transformer = self.training_config.get('fast_transformer', True)
        
        self.model = TransformerModel(
            len(self.vocab),
            embsize,
            nhead,
            d_hid,
            nlayers,
            nlayers_cls=3,
            n_cls=num_types,
            vocab=self.vocab,
            dropout=dropout,
            pad_token=self.pad_token,
            pad_value=-2,
            do_mvc=False,
            do_dab=False,
            use_batch_labels=False,
            num_batch_labels=1,
            domain_spec_batchnorm=False,
            input_emb_style="continuous",
            n_input_bins=n_bins,
            cell_emb_style="cls",
            mvc_decoder_style="inner product",
            ecs_threshold=0.0,
            explicit_zero_prob=False,
            use_fast_transformer=fast_transformer,
            fast_transformer_backend="flash",
            pre_norm=False,
        )
        
        # Load pretrained weights
        self.model = self._load_pretrained_weights(self.model)
        
        # Freeze encoder if specified
        if self.architecture.get('freeze_encoder', False):
            for name, param in self.model.named_parameters():
                if "encoder" in name and "transformer_encoder" not in name:
                    param.requires_grad = False
            self.log_info("Froze encoder weights")
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        self.log_info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
        
        self.model.to(self.device)
    
    def predict(self, query_data: ad.AnnData, **kwargs) -> np.ndarray:
        """
        Predict labels for query data.
        
        Parameters
        ----------
        query_data : AnnData
            Query dataset
            
        Returns
        -------
        predictions : np.ndarray
            Predicted cell type IDs
        """
        self.log_info("Predicting labels for query data")
        
        # Match genes to vocab and preprocess
        query_data = self._match_genes_to_vocab(query_data)
        
        # CRITICAL: Ensure query has exact same genes as training (in same order)
        if hasattr(self, 'training_genes'):
            training_gene_set = set(self.training_genes)
            query_genes = query_data.var["gene_name"].values
            
            # Find common genes (intersection)
            common_genes = [g for g in self.training_genes if g in query_genes]
            n_common = len(common_genes)
            n_missing = len(self.training_genes) - n_common
            
            if n_missing > 0:
                self.log_warning(f"Query missing {n_missing}/{len(self.training_genes)} training genes")
            
            if n_common == 0:
                raise ValueError("No common genes between training and query")
            
            # Subset query to common genes and reorder to match training
            print("======= QUERY NAMES ========")
            print(query_data.var_names)
            print("======= COMMON GENES ========")
            print(common_genes)
            query_data = query_data[:, common_genes].copy()
            
            # Update gene_ids for the reduced gene set
            self.gene_ids = np.array(self.vocab(common_genes), dtype=int)
            
            self.log_info(f"Using {n_common} common genes for prediction")
        
        query_data = self._preprocess_data(query_data)
        
        # Prepare test data
        input_layer_key = "X_binned"
        all_counts_test = (
            query_data.layers[input_layer_key].A
            if issparse(query_data.layers[input_layer_key])
            else query_data.layers[input_layer_key]
        )
        
        # Use celltype_id_for_model if available (handles cells with labels not in reference)
        if "celltype_id_for_model" in query_data.obs.columns:
            celltypes_labels_test = query_data.obs["celltype_id_for_model"].values
        else:
            celltypes_labels_test = query_data.obs["celltype_id"].values
        batch_ids_test = query_data.obs.get("batch_id", np.ones(len(celltypes_labels_test))).values
        
        # Tokenize
        max_seq_len = self.architecture.get('max_seq_len', 3001)
        include_zero_gene = self.tokenization.get('include_zero_gene', False)
        
        tokenized_test = tokenize_and_pad_batch(
            all_counts_test,
            self.gene_ids,
            max_len=max_seq_len,
            vocab=self.vocab,
            pad_token=self.pad_token,
            pad_value=-2,
            append_cls=True,
            include_zero_gene=include_zero_gene,
        )
        
        test_data_pt = {
            "gene_ids": tokenized_test["genes"],
            "values": tokenized_test["values"],
            "target_values": tokenized_test["values"],
            "batch_labels": torch.from_numpy(batch_ids_test).long(),
            "celltype_labels": torch.from_numpy(celltypes_labels_test).long(),
        }
        
        test_loader = DataLoader(
            dataset=SeqDataset(test_data_pt),
            batch_size=kwargs.get('batch_size', 16),
            shuffle=False,
            drop_last=False,
            num_workers=min(len(os.sched_getaffinity(0)), kwargs.get('batch_size', 16) // 2),
            pin_memory=True,
        )
        
        # Get predictions
        criterion_cls = nn.CrossEntropyLoss()
        amp = self.training_config.get('amp', True)
        predictions = evaluate(
            self.model, test_loader, criterion_cls, self.device,
            self.vocab, self.pad_token, amp, False, return_raw=True
        )
        
        self.log_info(f"Generated predictions for {len(predictions)} cells")
        return predictions
    
    def save_model(self) -> None:
        """Save model weights and vocabulary."""
        if self.model is not None:
            model_path = self.model_outputs_dir / "best_model.pt"
            torch.save(self.model.state_dict(), model_path)
            self.log_info(f"Saved model weights to {model_path}")
        
        if self.vocab is not None:
            vocab_path = self.model_outputs_dir / "vocab.json"
            self.vocab.save_json(vocab_path)
            self.log_info(f"Saved vocabulary to {vocab_path}")
