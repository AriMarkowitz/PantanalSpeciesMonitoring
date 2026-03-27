"""PyTorch datasets for training and inference.

FeatureDataset: reads pre-computed features.h5 (Stage 3 output) for
classifier training. Labels, masks, and fold assignments are stored
in the same HDF5 file — no CSV joins at training time.
"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class FeatureDataset(Dataset):
    """Dataset backed by features.h5 from Stage 3.

    Opens the HDF5 file once and keeps the handle alive for the
    process lifetime. Supports read-only concurrent access from
    multiple fold training jobs.
    """

    def __init__(self, h5_path: str, fold: int, split: str = "train",
                 n_folds: int = 5, pseudo_labels: np.ndarray = None,
                 pseudo_weight: float = 0.5):
        """
        Args:
            h5_path: path to features.h5
            fold: which fold to hold out for validation
            split: "train" or "val"
            n_folds: total number of folds
            pseudo_labels: optional (N, C) array of pseudo-label targets
            pseudo_weight: weight for pseudo-labeled samples in the mask
        """
        self.h5 = h5py.File(h5_path, "r")
        self.features = self.h5["features"]
        self.labels = self.h5["labels"][:]    # load fully — small
        self.masks = self.h5["masks"][:].copy()
        self.folds = self.h5["folds"][:]

        # Split by fold
        if split == "val":
            self.indices = np.where(self.folds == fold)[0]
        else:
            self.indices = np.where(
                (self.folds != fold) & (self.folds >= 0)
            )[0]

        # Incorporate pseudo-labels if provided
        if pseudo_labels is not None:
            # Only apply to unlabeled samples (mask is all zeros)
            unlabeled = self.masks.sum(axis=1) == 0
            for i in np.where(unlabeled)[0]:
                pl = pseudo_labels[i]
                confident = pl > 0  # pseudo-label is present
                if confident.any():
                    self.labels[i] = pl
                    self.masks[i, confident] = pseudo_weight

        self.feat_dim = self.features.shape[1]
        self.num_classes = self.labels.shape[1]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        feat = torch.from_numpy(self.features[i].astype(np.float32))
        label = torch.from_numpy(self.labels[i])
        mask = torch.from_numpy(self.masks[i])
        return feat, label, mask

    def close(self):
        self.h5.close()


def get_dataloaders(h5_path: str, fold: int, batch_size: int,
                    n_folds: int = 5, num_workers: int = 0,
                    pseudo_labels: np.ndarray = None,
                    pseudo_weight: float = 0.5):
    """Create train and val DataLoaders for a given fold."""
    train_ds = FeatureDataset(h5_path, fold, "train", n_folds,
                              pseudo_labels, pseudo_weight)
    val_ds = FeatureDataset(h5_path, fold, "val", n_folds)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl
