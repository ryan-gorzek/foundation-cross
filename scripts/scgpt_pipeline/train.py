#!/usr/bin/env python
"""
Main script for scGPT cross-species label transfer.
Fine-tune pre-trained scGPT on mouse data and transfer labels to opossum.
"""
import os
import sys
import json
import time
import copy
import shutil
import pickle
import logging
from pathlib import Path
from typing import Dict
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from scipy.sparse import issparse

# scGPT imports
import scgpt as scg
from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed
from scgpt.loss import masked_mse_loss, criterion_neg_log_bernoulli

# Try to import wandb (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available. Logging will be local only.")

# Local imports
import config
from data_utils import load_and_prepare_data, prepare_anndata
from train_utils import (
    SeqDataset, prepare_train_valid_split, train_epoch, evaluate
)


def setup_logger(save_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")
    return logger


def initialize_wandb(config_dict: Dict):
    """Initialize Weights & Biases logging."""
    if not WANDB_AVAILABLE:
        return None
    
    if config.WANDB_PROJECT is None:
        return None
    
    run = wandb.init(
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY,
        config=config_dict,
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
    )
    return run


def load_pretrained_model(model_dir: Path, adata, special_tokens, logger):
    """Load pre-trained model weights and vocabulary."""
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"
    
    # Load vocabulary
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    
    # Convert gene names to uppercase to match human vocabulary
    logger.info("Converting gene names to uppercase to match pretrained vocabulary")
    adata.var["gene_name_original"] = adata.var["gene_name"].copy()
    adata.var["gene_name"] = adata.var["gene_name"].str.upper()
    
    # Match genes to vocabulary
    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info(
        f"Matched {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    
    # Subset to genes in vocabulary
    adata = adata[:, adata.var["id_in_vocab"] >= 0]
    
    # Load model config
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    
    logger.info(f"Loading model config from {model_config_file}")
    
    return vocab, adata, model_configs, model_file


def prepare_dataloaders(
    train_data_pt: Dict,
    valid_data_pt: Dict,
    batch_size: int,
    eval_batch_size: int
) -> tuple:
    """Create PyTorch dataloaders."""
    num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)
    
    train_loader = DataLoader(
        dataset=SeqDataset(train_data_pt),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    valid_loader = DataLoader(
        dataset=SeqDataset(valid_data_pt),
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, valid_loader


def main():
    """Main training pipeline."""
    # Set random seed
    set_seed(config.SEED)
    
    # Create save directory
    save_dir = config.SAVE_DIR / f"{config.DATASET_NAME}-{time.strftime('%b%d-%H-%M')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to {save_dir}")
    
    # Setup logging
    logger = setup_logger(save_dir)
    logger.info("=" * 89)
    logger.info("scGPT Cross-Species Label Transfer Pipeline")
    logger.info("=" * 89)
    
    # Initialize wandb
    config_dict = {k: v for k, v in vars(config).items() if k.isupper()}
    wandb_run = initialize_wandb(config_dict)
    
    # ========================================================================
    # 1. LOAD AND PREPARE DATA
    # ========================================================================
    logger.info("\n" + "=" * 89)
    logger.info("STEP 1: Loading and preparing data")
    logger.info("=" * 89)
    
    adata_train, adata_test, metadata = load_and_prepare_data(
        config.MOUSE_H5AD,
        config.OPOSSUM_H5AD,
        train_species="mouse"
    )
    
    # Keep species separate for preprocessing (no cross-species information leakage)
    
    # ========================================================================
    # 2. SETUP TOKENIZATION AND PREPROCESSING
    # ========================================================================
    logger.info("\n" + "=" * 89)
    logger.info("STEP 2: Preprocessing and tokenization")
    logger.info("=" * 89)
    
    # Special tokens
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    
    # Load pre-trained model if specified
    if config.PRETRAINED_MODEL_DIR and config.PRETRAINED_MODEL_DIR.exists():
        vocab, adata_train, model_configs, model_file = load_pretrained_model(
            config.PRETRAINED_MODEL_DIR, adata_train, special_tokens, logger
        )
        # Copy vocab to save directory
        shutil.copy(config.PRETRAINED_MODEL_DIR / "vocab.json", save_dir / "vocab.json")
        
        # Apply same gene name conversion to test data
        logger.info("Applying same gene name conversion to test data")
        adata_test.var["gene_name_original"] = adata_test.var["gene_name"].copy()
        adata_test.var["gene_name"] = adata_test.var["gene_name"].str.upper()
        
        # Subset test data to same genes as training
        common_genes = adata_train.var_names.intersection(adata_test.var_names)
        logger.info(f"Test data: subsetting to {len(common_genes)} genes that matched vocabulary")
        adata_test = adata_test[:, common_genes].copy()
        adata_train = adata_train[:, common_genes].copy()  # Ensure train also has same genes
    else:
        logger.info("No pre-trained model specified. Building vocabulary from data.")
        # Build vocabulary from genes (training data only)
        genes = adata_train.var["gene_name"].tolist()
        vocab = GeneVocab.from_default(genes + special_tokens)
        # Save vocab
        vocab.save_json(save_dir / "vocab.json")
        model_configs = None
        model_file = None
    
    vocab.set_default_index(vocab[pad_token])
    
    # Preprocessing - IMPORTANT: Process train and test separately to avoid cross-species leakage
    logger.info("Preprocessing training data (mouse) separately from test data (opossum)")
    
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=3,  # Filter cells with < 3 genes expressed
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=False,
        result_log1p_key="X_log1p",
        subset_hvg=False,
        binning=config.N_BINS,
        result_binned_key="X_binned",
    )
    
    # Remove cells with all zeros BEFORE preprocessing (already done in data_utils, but double-check)
    from scipy.sparse import issparse
    if issparse(adata_train.X):
        cell_counts = np.array(adata_train.X.sum(axis=1)).flatten()
        test_cell_counts = np.array(adata_test.X.sum(axis=1)).flatten()
    else:
        cell_counts = adata_train.X.sum(axis=1)
        test_cell_counts = adata_test.X.sum(axis=1)
    
    n_before_train = adata_train.n_obs
    n_before_test = adata_test.n_obs
    
    adata_train = adata_train[cell_counts > 0].copy()
    adata_test = adata_test[test_cell_counts > 0].copy()
    
    logger.info(f"Filtered {n_before_train - adata_train.n_obs} zero-count cells from training data")
    logger.info(f"Filtered {n_before_test - adata_test.n_obs} zero-count cells from test data")
    
    # Preprocess train and test SEPARATELY (no cross-species information leakage)
    logger.info("Preprocessing training data...")
    preprocessor(adata_train, batch_key=None)
    
    logger.info("Preprocessing test data...")
    preprocessor(adata_test, batch_key=None)
    
    logger.info(f"Preprocessed {adata_train.n_obs} training cells and {adata_test.n_obs} test cells")
    
    # Store original adata_test for later (with predictions)
    adata_test_raw = adata_test.copy()
    
    # ========================================================================
    # 3. PREPARE TRAINING DATA
    # ========================================================================
    logger.info("\n" + "=" * 89)
    logger.info("STEP 3: Preparing training data")
    logger.info("=" * 89)
    
    # Get counts from binned layer
    input_layer_key = "X_binned"
    all_counts = (
        adata_train.layers[input_layer_key].A
        if issparse(adata_train.layers[input_layer_key])
        else adata_train.layers[input_layer_key]
    )
    
    genes = adata_train.var["gene_name"].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)
    
    celltypes_labels = adata_train.obs["celltype_id"].values
    batch_ids = adata_train.obs["batch_id"].values
    
    # Train/validation split
    (
        train_data,
        valid_data,
        train_celltype_labels,
        valid_celltype_labels,
        train_batch_labels,
        valid_batch_labels,
    ) = prepare_train_valid_split(all_counts, celltypes_labels, batch_ids, test_size=0.1)
    
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(valid_data)}")
    
    # Tokenize
    mask_value = -1
    pad_value = -2
    
    tokenized_train = tokenize_and_pad_batch(
        train_data,
        gene_ids,
        max_len=config.MAX_SEQ_LEN,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=config.INCLUDE_ZERO_GENE,
    )
    
    tokenized_valid = tokenize_and_pad_batch(
        valid_data,
        gene_ids,
        max_len=config.MAX_SEQ_LEN,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=config.INCLUDE_ZERO_GENE,
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
    train_loader, valid_loader = prepare_dataloaders(
        train_data_pt, valid_data_pt, config.BATCH_SIZE, config.BATCH_SIZE
    )
    
    # ========================================================================
    # 4. INITIALIZE MODEL
    # ========================================================================
    logger.info("\n" + "=" * 89)
    logger.info("STEP 4: Initializing model")
    logger.info("=" * 89)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Model parameters
    if model_configs:
        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
    else:
        embsize = config.LAYER_SIZE
        nhead = config.N_HEAD
        d_hid = config.LAYER_SIZE
        nlayers = config.N_LAYERS
    
    ntokens = len(vocab)
    num_types = metadata['num_types']
    
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=3,
        n_cls=num_types,
        vocab=vocab,
        dropout=config.DROPOUT,
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        num_batch_labels=1,
        domain_spec_batchnorm=False,
        input_emb_style="continuous",
        n_input_bins=config.N_BINS,
        cell_emb_style="cls",
        mvc_decoder_style="inner product",
        ecs_threshold=0.0,
        explicit_zero_prob=False,
        use_fast_transformer=config.FAST_TRANSFORMER,
        fast_transformer_backend="flash",
        pre_norm=False,
    )
    
    # Load pre-trained weights if available
    if model_file and model_file.exists():
        try:
            model.load_state_dict(torch.load(model_file, map_location=device))
            logger.info(f"Loaded pre-trained weights from {model_file}")
        except:
            logger.warning("Failed to load all weights. Loading compatible weights only.")
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_file, map_location=device)
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            logger.info(f"Loaded {len(pretrained_dict)} compatible weight tensors")
    
    # Freeze encoder if specified
    if config.FREEZE_ENCODER:
        for name, param in model.named_parameters():
            if "encoder" in name and "transformer_encoder" not in name:
                param.requires_grad = False
        logger.info("Froze encoder weights")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    model.to(device)
    
    # ========================================================================
    # 5. SETUP TRAINING
    # ========================================================================
    logger.info("\n" + "=" * 89)
    logger.info("STEP 5: Setup training")
    logger.info("=" * 89)
    
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        eps=1e-4 if config.AMP else 1e-8
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=1, 
        gamma=config.SCHEDULE_RATIO
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP)
    
    # ========================================================================
    # 6. TRAIN MODEL
    # ========================================================================
    logger.info("\n" + "=" * 89)
    logger.info("STEP 6: Training model")
    logger.info("=" * 89)
    
    best_val_loss = float("inf")
    best_model = None
    best_model_epoch = 0
    
    for epoch in range(1, config.EPOCHS + 1):
        epoch_start_time = time.time()
        
        if config.DO_TRAIN:
            train_loss, train_error = train_epoch(
                model, train_loader, criterion_cls, optimizer, scaler,
                device, vocab, pad_token, epoch, config.LOG_INTERVAL, 
                logger, config, scheduler
            )
        
        val_loss, val_error = evaluate(
            model, valid_loader, criterion_cls, device, vocab, pad_token, config
        )
        
        elapsed = time.time() - epoch_start_time
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss {val_loss:5.4f} | valid error {val_error:5.4f}"
        )
        logger.info("-" * 89)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            logger.info(f"Best model at epoch {epoch} with validation loss {best_val_loss:5.4f}")
        
        scheduler.step()
        
        # Log to wandb
        if wandb_run:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss if config.DO_TRAIN else 0,
                "train/error": train_error if config.DO_TRAIN else 0,
                "valid/loss": val_loss,
                "valid/error": val_error,
            })
    
    logger.info(f"\nBest model from epoch {best_model_epoch}")
    
    # ========================================================================
    # 7. EVALUATE ON TEST DATA
    # ========================================================================
    logger.info("\n" + "=" * 89)
    logger.info("STEP 7: Evaluating on test data (opossum)")
    logger.info("=" * 89)
    
    # Prepare test data
    all_counts_test = (
        adata_test.layers[input_layer_key].A
        if issparse(adata_test.layers[input_layer_key])
        else adata_test.layers[input_layer_key]
    )
    
    celltypes_labels_test = adata_test.obs["celltype_id"].values
    batch_ids_test = adata_test.obs["batch_id"].values
    
    tokenized_test = tokenize_and_pad_batch(
        all_counts_test,
        gene_ids,
        max_len=config.MAX_SEQ_LEN,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=config.INCLUDE_ZERO_GENE,
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
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), config.BATCH_SIZE // 2),
        pin_memory=True,
    )
    
    # Get predictions
    predictions = evaluate(
        best_model, test_loader, criterion_cls, device, vocab, pad_token, config, return_raw=True
    )
    
    # Calculate metrics only on cells with valid labels
    labels = celltypes_labels_test
    
    # Check if we have the 'has_valid_label' column
    if 'has_valid_label' in adata_test.obs.columns:
        valid_mask = adata_test.obs['has_valid_label'].values
        n_valid = valid_mask.sum()
        n_total = len(valid_mask)
        
        logger.info(f"\nEvaluating on {n_valid}/{n_total} cells with matching labels")
        
        # Compute metrics only on valid cells
        labels_valid = labels[valid_mask]
        predictions_valid = predictions[valid_mask]
        
        accuracy = accuracy_score(labels_valid, predictions_valid)
        precision = precision_score(labels_valid, predictions_valid, average="macro", zero_division=0)
        recall = recall_score(labels_valid, predictions_valid, average="macro", zero_division=0)
        macro_f1 = f1_score(labels_valid, predictions_valid, average="macro", zero_division=0)
        
        logger.info(
            f"Test Results (on cells with matching labels):\n"
            f"  Accuracy: {accuracy:.4f}\n"
            f"  Precision: {precision:.4f}\n"
            f"  Recall: {recall:.4f}\n"
            f"  Macro F1: {macro_f1:.4f}"
        )
    else:
        # All cells have valid labels (shouldn't happen with cross-species)
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average="macro", zero_division=0)
        recall = recall_score(labels, predictions, average="macro", zero_division=0)
        macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
        
        logger.info(
            f"Test Results:\n"
            f"  Accuracy: {accuracy:.4f}\n"
            f"  Precision: {precision:.4f}\n"
            f"  Recall: {recall:.4f}\n"
            f"  Macro F1: {macro_f1:.4f}"
        )
    
    # Add predictions to test data
    id2type = metadata['id2type']
    adata_test_raw.obs["predictions"] = [id2type[p] for p in predictions]
    
    # Save results
    results = {
        "test/accuracy": accuracy,
        "test/precision": precision,
        "test/recall": recall,
        "test/macro_f1": macro_f1,
    }
    
    save_dict = {
        "predictions": predictions,
        "labels": labels,
        "results": results,
        "id2type": id2type,
        "metadata": metadata,
    }
    
    with open(save_dir / "results.pkl", "wb") as f:
        pickle.dump(save_dict, f)
    
    # ========================================================================
    # 8. GENERATE VISUALIZATIONS
    # ========================================================================
    logger.info("\n" + "=" * 89)
    logger.info("STEP 8: Generating visualizations")
    logger.info("=" * 89)
    
    # Confusion matrix: Opossum true labels (rows) vs Mouse predictions (columns)
    logger.info("Generating confusion matrix with opossum labels (true) vs mouse labels (predicted)")
    
    # Get original opossum cell type labels (before recoding)
    if 'celltype_original' in adata_test.obs.columns:
        true_labels_str = adata_test_raw.obs['celltype_original'].values
    else:
        # Fallback: use current celltype
        true_labels_str = adata_test_raw.obs['celltype'].values
    
    # Get predicted labels as strings (mouse cell types)
    pred_labels_str = adata_test_raw.obs['predictions'].values
    
    # Create confusion matrix with string labels
    from sklearn.preprocessing import LabelEncoder
    
    # Encode both to numeric for confusion_matrix
    all_true_types = np.unique(true_labels_str)
    all_pred_types = np.unique(pred_labels_str)
    
    # Create label encoders
    true_encoder = LabelEncoder()
    true_encoder.fit(all_true_types)
    true_labels_encoded = true_encoder.transform(true_labels_str)
    
    pred_encoder = LabelEncoder()
    pred_encoder.fit(all_pred_types)
    pred_labels_encoded = pred_encoder.transform(pred_labels_str)
    
    # Generate confusion matrix
    cm = confusion_matrix(true_labels_encoded, pred_labels_encoded)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    # Create DataFrame with proper labels
    cm_df = pd.DataFrame(
        cm_normalized,
        index=true_encoder.classes_,  # Opossum cell types (rows = true)
        columns=pred_encoder.classes_  # Mouse cell types (columns = predicted)
    )
    
    # Sort rows and columns for better visualization
    cm_df = cm_df.sort_index()  # Sort opossum types
    cm_df = cm_df[sorted(cm_df.columns)]  # Sort mouse types
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_df, annot=True, fmt=".2f", cmap="Blues", cbar_kws={"label": "Proportion"})
    plt.title("Cross-Species Label Transfer: Opossum (True) → Mouse (Predicted)")
    plt.ylabel("True Label (Opossum)")
    plt.xlabel("Predicted Label (Mouse)")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confusion matrix to {save_dir / 'confusion_matrix.png'}")
    logger.info(f"  Rows (true labels): {len(true_encoder.classes_)} opossum cell types")
    logger.info(f"  Columns (predicted): {len(pred_encoder.classes_)} mouse cell types")
    
    # Save model
    torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    logger.info(f"Saved model weights to {save_dir / 'best_model.pt'}")
    
    # Save test data with predictions
    adata_test_raw.write_h5ad(save_dir / "test_data_with_predictions.h5ad")
    logger.info(f"Saved test data with predictions to {save_dir / 'test_data_with_predictions.h5ad'}")
    
    logger.info("\n" + "=" * 89)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 89)
    
    if wandb_run:
        wandb.log(results)
        wandb.log({"test/confusion_matrix": wandb.Image(str(save_dir / "confusion_matrix.png"))})
        wandb.finish()


if __name__ == "__main__":
    main()
