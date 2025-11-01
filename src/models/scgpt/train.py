"""
Training utilities for scGPT model.
"""
import torch
import torch.nn as nn
import numpy as np
import time
import warnings
from torch.utils.data import DataLoader
from typing import Tuple


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
    amp: bool = True,
    input_batch_labels: bool = False,
    scheduler=None
) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    
    Parameters
    ----------
    model : nn.Module
        Model to train
    loader : DataLoader
        Training data loader
    criterion_cls : nn.Module
        Classification loss criterion
    optimizer : torch.optim.Optimizer
        Optimizer
    scaler : torch.cuda.amp.GradScaler
        Gradient scaler for mixed precision
    device : torch.device
        Device to train on
    vocab : object
        Vocabulary object
    pad_token : str
        Padding token
    epoch : int
        Current epoch number
    log_interval : int
        Logging interval in batches
    logger : object
        Logger instance
    amp : bool
        Whether to use automatic mixed precision
    input_batch_labels : bool
        Whether to input batch labels to model
    scheduler : Optional
        Learning rate scheduler
        
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
        
        with torch.cuda.amp.autocast(enabled=amp):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if input_batch_labels else None,
                CLS=True,
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
    amp: bool = True,
    input_batch_labels: bool = False,
    return_raw: bool = False
) -> Tuple:
    """
    Evaluate the model on the evaluation data.
    
    Parameters
    ----------
    model : nn.Module
        Model to evaluate
    loader : DataLoader
        Evaluation data loader
    criterion_cls : nn.Module
        Classification loss criterion
    device : torch.device
        Device to evaluate on
    vocab : object
        Vocabulary object
    pad_token : str
        Padding token
    amp : bool
        Whether to use automatic mixed precision
    input_batch_labels : bool
        Whether to input batch labels to model
    return_raw : bool
        If True, return raw predictions instead of metrics
        
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
            
            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if input_batch_labels else None,
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