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
    
    # Store test data for later
    adata_test_raw = adata_test.copy()
    
    # Combine for preprocessing
    adata = prepare_anndata(adata_train, adata_test)
    
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
        vocab, adata, model_configs, model_file = load_pretrained_model(
            config.PRETRAINED_MODEL_DIR, adata, special_tokens, logger
        )
        # Copy vocab to save directory
        shutil.copy(config.PRETRAINED_MODEL_DIR / "vocab.json", save_dir / "vocab.json")
    else:
        logger.info("No pre-trained model specified. Building vocabulary from data.")
        from scgpt.tokenizer.gene_tokenizer import Vocab, VocabPybind
        genes = adata.var["gene_name"].tolist()
        vocab = Vocab(VocabPybind(genes + special_tokens, None))
        model_configs = None
        model_file = None
    
    vocab.set_default_index(vocab[pad_token])
    
    # Preprocessing
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=False,
        result_log1p_key="X_log1p",
        subset_hvg=False,
        binning=config.N_BINS,
        result_binned_key="X_binned",
    )
    
    # Split back into train and test
    adata_test = adata[adata.obs["str_batch"] == "1"]
    adata = adata[adata.obs["str_batch"] == "0"]
    
    # Preprocess
    preprocessor(adata, batch_key=None)
    preprocessor(adata_test, batch_key=None)
    
    logger.info(f"Preprocessed {adata.n_obs} training cells and {adata_test.n_obs} test cells")
    
    # ========================================================================
    # 3. PREPARE TRAINING DATA
    # ========================================================================
    logger.info("\n" + "=" * 89)
    logger.info("STEP 3: Preparing training data")
    logger.info("=" * 89)
    
    # Get counts from binned layer
    input_layer_key = "X_binned"
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    
    genes = adata.var["gene_name"].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)
    
    celltypes_labels = adata.obs["celltype_id"].values
    batch_ids = adata.obs["batch_id"].values
    
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
    
    # Calculate metrics
    labels = celltypes_labels_test
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
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    celltypes = [id2type[i] for i in range(len(id2type))]
    cm_df = pd.DataFrame(
        cm_normalized, 
        index=celltypes[:cm.shape[0]], 
        columns=celltypes[:cm.shape[1]]
    )
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt=".2f", cmap="Blues", cbar_kws={"label": "Proportion"})
    plt.title("Confusion Matrix (Row-Normalized)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confusion matrix to {save_dir / 'confusion_matrix.png'}")
    
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
