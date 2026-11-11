"""Dataset loading, verification and training-only preprocessing statistics."""
import hashlib
import json
import os

import numpy as np

EXPECTED_MD5 = "9de6cd0b934ebb5b7426cfba5efbae16"
SPLITS = {"train": 78468, "val": 11219, "test": 22433}


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_archive(path="chestmnist_64.npz", verify=True, splits=None):
    """Load the official MedMNIST+ archive and verify integrity.

    Download chestmnist_64.npz yourself from the official Zenodo record
    (10.5281/zenodo.10519652); it is not redistributed in this repository.
    """
    if verify:
        digest = md5(path)
        if digest != EXPECTED_MD5:
            raise RuntimeError(
                f"MD5 mismatch: got {digest}, expected {EXPECTED_MD5}. "
                "Re-download the archive from the official Zenodo record."
            )
    z = np.load(path)
    data = {}
    for split in (SPLITS if splits is None else splits):
        x = z[f"{split}_images"]
        y = z[f"{split}_labels"].astype(np.int64)
        assert x.shape[0] == SPLITS[split], (split, x.shape)
        assert y.shape == (SPLITS[split], 14)
        assert set(np.unique(y)) <= {0, 1}
        data[split] = (x, y)
    return data


def train_statistics(train_images):
    """Single mean/std over training pixels only (eq. 2 of the manuscript)."""
    x = train_images.astype(np.float32) / 255.0
    return float(x.mean()), float(x.std())


def standardise(images, mean, std):
    return (images.astype(np.float32) / 255.0 - mean) / std


def audit(data, out_path="outputs/audit.json"):
    """Exact-duplicate audit via SHA-256 of decoded pixels."""
    groups = {}
    for split, (x, y) in data.items():
        for i in range(x.shape[0]):
            h = hashlib.sha256(x[i].tobytes()).hexdigest()
            groups.setdefault(h, []).append((split, i, y[i].tolist()))
    dup = {h: v for h, v in groups.items() if len(v) > 1}
    cross = {h: v for h, v in dup.items() if len({s for s, _, _ in v}) > 1}
    conflicting = {h: v for h, v in dup.items()
                   if len({tuple(lab) for _, _, lab in v}) > 1}
    report = {
        "unique_images": len(groups),
        "duplicate_groups": len(dup),
        "duplicate_rows": sum(len(v) for v in dup.values()),
        "cross_split_groups": len(cross),
        "conflicting_label_groups": len(conflicting),
        "cross_split_detail": cross,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return report
