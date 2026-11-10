"""Unweighted (B2) and training-derived positive-weighted (B3) BCE."""
import numpy as np
import torch


def positive_weights(train_labels: np.ndarray) -> torch.Tensor:
    """w_j = (N_tr - N_j+) / N_j+ computed from training labels only (eq. 6)."""
    n = train_labels.shape[0]
    pos = train_labels.sum(axis=0)
    return torch.tensor((n - pos) / pos, dtype=torch.float32)


def make_criterion(condition: str, train_labels: np.ndarray):
    if condition == "B2":
        return torch.nn.BCEWithLogitsLoss()
    if condition == "B3":
        return torch.nn.BCEWithLogitsLoss(pos_weight=positive_weights(train_labels))
    raise ValueError(condition)
