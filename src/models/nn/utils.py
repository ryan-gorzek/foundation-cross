"""
Utils for neural network classifier Datasets.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit
import anndata as ad

class SingleCellDataset(Dataset):
    
    def __init__(
        self,
        data: ad.AnnData,
        var_col: str = "highly_variable",
        obs_col: str = "celltype_id",
        make_dense: bool = False
        ):
        
        mask = data.var[var_col].to_numpy()
        X = data.X[:, mask]
        if make_dense:
            X = torch.tensor(X.todense(), dtype=torch.float32)
        y = data.obs[obs_col]
        if hasattr(y, "cat"):
            y = y.cat.codes
        self.X = X
        self.y = torch.tensor(y, dtype=torch.long)
        self.make_dense = make_dense

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        if not self.make_dense:
            if hasattr(x, "toarray"):
                x = x.toarray().ravel()
            else:
                x = np.asarray(x).ravel()
            x = torch.tensor(x, dtype=torch.float32)
        y = self.y[idx]
        return x, y

def make_dataloader(
    data: ad.AnnData,
    var_col: str = "highly_variable",
    obs_col: str = "celltype_id",
    make_dense: bool = False,
    batch_size: int = 512,
    seed: int = 7,
    num_workers: int = 0,
    pin_memory: bool = True
    ):
    """
    Generic dataloader creation for AnnData.
    """

    ds = SingleCellDataset(data, var_col=var_col, obs_col=obs_col, make_dense=make_dense)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        )
    return loader

def make_dataloaders_group_split(
    data: ad.AnnData,
    var_col: str = "highly_variable",
    obs_col: str = "celltype_id",
    make_dense: bool = False,
    batch_col: str = "sample",
    val_frac: float = 0.2,
    batch_size: int = 512,
    use_weighted_sampler: bool = False,
    seed: int = 7,
    num_workers: int = 0,
    pin_memory: bool = True
    ):
    """
    Group split by sample_col (no sample leakage between train/val).
    Optionally use WeightedRandomSampler on the train split to mitigate class imbalance.
    """

    ds = SingleCellDataset(data, var_col=var_col, obs_col=obs_col, make_dense=make_dense)
    X = ds.X
    y = ds.y
    batch_ids = data.obs[batch_col].cat.codes

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_idx, val_idx = next(splitter.split(X, y, groups=batch_ids))
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    # Class weights computed on train labels (only)
    y_train = ds.y[train_idx].cpu().numpy()
    class_counts = np.bincount(y_train)
    class_counts = np.maximum(class_counts, 1) # avoid div by zero

    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
    sample_weights = class_weights[y_train]

    if use_weighted_sampler:
        # Apply weighted sampling to address class imbalance
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False, # must be False when sampler is set
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, train_idx, val_idx, class_weights
