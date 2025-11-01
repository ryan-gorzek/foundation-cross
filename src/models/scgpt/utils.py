"""
Utility functions for scGPT model.
"""
import torch
import numpy as np
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class SeqDataset(Dataset):
    """PyTorch dataset for gene expression sequences."""
    
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def prepare_train_valid_split(
    all_counts: np.ndarray,
    celltypes_labels: np.ndarray,
    batch_ids: np.ndarray,
    test_size: float = 0.1,
    random_state: int = 0
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


def prepare_data_loaders(
    train_data_pt: Dict[str, torch.Tensor],
    valid_data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    num_workers: int = 0
) -> Tuple:
    """
    Create PyTorch dataloaders.
    
    Parameters
    ----------
    train_data_pt : Dict
        Training data tensors
    valid_data_pt : Dict
        Validation data tensors
    batch_size : int
        Batch size
    num_workers : int
        Number of data loading workers
        
    Returns
    -------
    train_loader, valid_loader : DataLoader
        Training and validation data loaders
    """
    from torch.utils.data import DataLoader
    
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
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, valid_loader