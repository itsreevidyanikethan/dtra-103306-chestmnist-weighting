"""Train B1/B2/B3 exactly as registered; validation-frozen checkpoint selection.

Usage:
  python -I src/train.py --condition B2 --seed 20261
  python -I src/train.py --condition B3 --seed 20262
  python -I src/train.py --condition B1          # linear baseline + weight-decay grid
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data import load_archive, train_statistics, standardise
from losses import make_criterion
from metrics import macro_ap
from model import CompactCNN


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def augment(x, rng):
    """Horizontal flip p=0.5 and cyclic shifts of -2..+2 px (executed protocol)."""
    if rng.random() < 0.5:
        x = torch.flip(x, dims=[-1])
    sh, sw = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
    return torch.roll(x, shifts=(sh, sw), dims=(-2, -1))


class AugmentedDataset(TensorDataset):
    def __init__(self, x, y, seed):
        super().__init__(x, y)
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        return augment(x, self.rng), y


@torch.no_grad()
def predict(model, x, batch=256):
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        out.append(torch.sigmoid(model(x[i:i + batch])).cpu().numpy())
    return np.concatenate(out)


def train_neural(condition, seed, data, out_dir="outputs"):
    set_seed(seed)
    (xt, yt), (xv, yv) = data["train"], data["val"]
    mean, std = train_statistics(xt)
    xt_t = torch.tensor(standardise(xt, mean, std)).unsqueeze(1)
    xv_t = torch.tensor(standardise(xv, mean, std)).unsqueeze(1)
    yt_t = torch.tensor(yt, dtype=torch.float32)

    model = CompactCNN()
    crit = make_criterion(condition, yt)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    loader = DataLoader(AugmentedDataset(xt_t, yt_t, seed), batch_size=64, shuffle=True)

    best_ap, best_epoch, best_state, stale = -1.0, -1, None, 0
    history = []
    for epoch in range(1, 21):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        val_ap = macro_ap(yv, predict(model, xv_t))
        sched.step(val_ap)
        history.append({"epoch": epoch, "val_macro_ap": val_ap})
        print(f"{condition} seed={seed} epoch={epoch} val_macro_ap={val_ap:.6f}")
        if val_ap > best_ap:
            best_ap, best_epoch, stale = val_ap, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale > 4:
                print("early-stopping patience exhausted")
                break

    os.makedirs(out_dir, exist_ok=True)
    torch.save(best_state, f"{out_dir}/{condition}_seed{seed}.pt")
    with open(f"{out_dir}/{condition}_seed{seed}_history.json", "w") as f:
        json.dump({"best_epoch": best_epoch, "best_val_macro_ap": best_ap,
                   "preproc": {"mean": mean, "std": std}, "history": history}, f, indent=2)


def train_linear(data, out_dir="outputs"):
    """B1 linear pixel baseline with validation-selected weight decay."""
    set_seed(103306)
    (xt, yt), (xv, yv) = data["train"], data["val"]
    mean, std = train_statistics(xt)
    xt_t = torch.tensor(standardise(xt, mean, std)).reshape(len(xt), -1)
    xv_t = torch.tensor(standardise(xv, mean, std)).reshape(len(xv), -1)
    yt_t = torch.tensor(yt, dtype=torch.float32)
    results = {}
    for wd in (1e-3, 1e-4, 1e-5):
        torch.manual_seed(103306)
        lin = torch.nn.Linear(4096, 14)
        opt = torch.optim.SGD(lin.parameters(), lr=0.05, momentum=0.9, weight_decay=wd)
        crit = torch.nn.BCEWithLogitsLoss()
        loader = DataLoader(TensorDataset(xt_t, yt_t), batch_size=512, shuffle=True)
        for _ in range(5):
            for xb, yb in loader:
                opt.zero_grad(); crit(lin(xb), yb).backward(); opt.step()
        with torch.no_grad():
            pv = torch.sigmoid(lin(xv_t)).numpy()
        results[wd] = {"val_macro_ap": macro_ap(yv, pv),
                       "state": {k: v.clone() for k, v in lin.state_dict().items()}}
        print(f"B1 wd={wd:g} val_macro_ap={results[wd]['val_macro_ap']:.6f}")
    best = max(results, key=lambda k: results[k]["val_macro_ap"])
    os.makedirs(out_dir, exist_ok=True)
    torch.save(results[best]["state"], f"{out_dir}/B1_wd{best:g}.pt")
    with open(f"{out_dir}/B1_selection.json", "w") as f:
        json.dump({"selected_weight_decay": best,
                   "grid": {str(k): v["val_macro_ap"] for k, v in results.items()}}, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["B1", "B2", "B3"], required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--archive", default="chestmnist_64.npz")
    a = ap.parse_args()
    data = load_archive(a.archive)
    if a.condition == "B1":
        train_linear(data)
    else:
        assert a.seed in (20261, 20262, 20263), "registered seeds only"
        train_neural(a.condition, a.seed, data)
