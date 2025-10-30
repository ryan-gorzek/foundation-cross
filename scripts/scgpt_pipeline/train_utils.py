"""
Training utilities for scGPT fine-tuning.
"""
import torch
import torch.nn as nn
import numpy as np
import time
import warnings
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split


class SeqDataset(Dataset):
    """PyTorch dataset for gene expression sequences."""
    
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def prepare_train_valid_split(
    all_counts,
    celltypes_labels,
    batch_ids,
    test_size=0.1,
    random_state=0
) -> Tuple:
    """
    Split data into training and validation sets.
    
    Parameters
    ----------
    all_counts : array
        Gene expression counts
    celltypes_labels : array
        Cell type labels
    batch_ids : array
        Batch identifiers
    test_size : float
        Fraction for validation
    random_state : int
        Random seed
        
    Returns
    -------
    Tuple of train/valid data splits
    """
    return train_test_split(
        all_counts, 
        celltypes_labels, 
        batch_ids, 
        test_size=test_size, 
        shuffle=True,
        random_state=random_state
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion_cls: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    vocab: object,
    pad_token: str,
    epoch: int,
    log_interval: int,
    logger: object,
    config: object,
    scheduler=None
) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    
    Returns
    -------
    avg_loss : float
        Average loss over epoch
    avg_error : float
        Average error rate over epoch
    """
    model.train()
    total_loss = 0.0
    total_error = 0.0
    total_num = 0
    start_time = time.time()
    
    num_batches = len(loader)
    
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        celltype_labels = batch_data["celltype_labels"].to(device)
        
        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        
        with torch.cuda.amp.autocast(enabled=config.AMP):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if hasattr(config, 'INPUT_BATCH_LABELS') and config.INPUT_BATCH_LABELS else None,
                CLS=True,  # Cell type classification
                CCE=False,
                MVC=False,
                ECS=False,
                do_sample=False,
            )
            
            # Classification loss
            loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
            loss = loss_cls
            
            # Calculate error rate
            error_rate = 1 - (
                (output_dict["cls_output"].argmax(1) == celltype_labels)
                .sum()
                .item()
            ) / celltype_labels.size(0)
        
        # Backward pass
        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. Scale is {scaler.get_scale()}. "
                    "This warning can be ignored if no longer occurs after autoscaling."
                )
        
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate metrics
        batch_size = len(input_gene_ids)
        total_loss += loss.item() * batch_size
        total_error += error_rate * batch_size
        total_num += batch_size
        
        # Logging
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / total_num
            cur_error = total_error / total_num
            
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.4f} | err {cur_error:5.4f}"
            )
            start_time = time.time()
    
    return total_loss / total_num, total_error / total_num


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion_cls: nn.Module,
    device: torch.device,
    vocab: object,
    pad_token: str,
    config: object,
    return_raw: bool = False
) -> Tuple:
    """
    Evaluate the model on the evaluation data.
    
    Returns
    -------
    If return_raw=False:
        loss : float
            Average loss
        error : float
            Average error rate
    If return_raw=True:
        predictions : array
            Predicted class labels
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_num = 0
    predictions = []
    
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)
            
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            
            with torch.cuda.amp.autocast(enabled=config.AMP):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if hasattr(config, 'INPUT_BATCH_LABELS') and config.INPUT_BATCH_LABELS else None,
                    CLS=True,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=False,
                )
                
                output_values = output_dict["cls_output"]
                loss = criterion_cls(output_values, celltype_labels)
            
            # Accumulate metrics
            batch_size = len(input_gene_ids)
            total_loss += loss.item() * batch_size
            accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
            total_error += (1 - accuracy / batch_size) * batch_size
            total_num += batch_size
            
            # Store predictions
            preds = output_values.argmax(1).cpu().numpy()
            predictions.append(preds)
    
    if return_raw:
        return np.concatenate(predictions, axis=0)
    
    return total_loss / total_num, total_error / total_num
