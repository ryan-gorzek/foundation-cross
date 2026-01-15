"""
Training functions for neural network classifiers.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

@torch.no_grad()
def _acc(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
    ):
    """
    Train one epoch for a neural network classifier.
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_num = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        logits = model(inputs).squeeze(-1)
        loss = criterion(logits, labels)
        acc = _acc(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_acc += acc * batch_size
        total_num += batch_size
    
    return total_loss / total_num, total_acc / total_num

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    return_raw=False
    ):
    """
    Evaluate a neural network classifier.
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_num = 0
    predictions = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        logits = model(inputs).squeeze(-1)
        loss = criterion(logits, labels)
        acc = _acc(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_acc += acc * batch_size
        total_num += batch_size
    
        # Store predictions
        preds = logits.argmax(1).cpu().numpy()
        predictions.append(preds)

    if return_raw:
        return np.concatenate(predictions, axis=0)

    return total_loss / total_num, total_acc / total_num