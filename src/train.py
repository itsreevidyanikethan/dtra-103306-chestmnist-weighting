"""Train B1/B2/B3 for a new reconstruction; validation-frozen checkpoint selection.

Usage:
  python src/train.py --condition B2 --seed 20261
  python src/train.py --condition B3 --seed 20262
  python src/train.py --condition B1          # linear baseline + weight-decay grid
"""
import argparse
import json
import os
import time

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

    model = CompactCNN().to(memory_format=torch.channels_last)
    crit = make_criterion(condition, yt)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    loader = DataLoader(AugmentedDataset(xt_t, yt_t, seed), batch_size=64, shuffle=True)

    os.makedirs(out_dir, exist_ok=True)
    resume_path = f'{out_dir}/{condition}_seed{seed}_resume.pt'
    done_path = f'{out_dir}/{condition}_seed{seed}_complete.json'
    if os.path.exists(done_path):
        print(f'{condition} seed={seed} already complete; preserving existing run', flush=True)
        return
    best_ap, best_epoch, best_state, stale = -1.0, -1, None, 0
    history = []; start_epoch = 1
    if os.path.exists(resume_path):
        saved = torch.load(resume_path, weights_only=False, map_location='cpu')
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        sched.load_state_dict(saved['scheduler'])
        best_ap = saved['best_ap']; best_epoch = saved['best_epoch']
        best_state = saved['best_state']; stale = saved['stale']; history = saved['history']
        np.random.set_state(saved['numpy_rng']); torch.set_rng_state(saved['torch_rng'])
        loader.dataset.rng.bit_generator.state = saved['augmentation_rng']
        start_epoch = saved['epoch'] + 1
        print(f'Resuming {condition} seed={seed} at epoch {start_epoch}', flush=True)
    for epoch in range(start_epoch, 21):
        if stale >= 4: break
        started = time.monotonic()
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb.contiguous(memory_format=torch.channels_last)), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        val_ap = macro_ap(yv, predict(model, xv_t))
        sched.step(val_ap)
        history.append({"epoch": epoch, "val_macro_ap": val_ap, "seconds": time.monotonic() - started})
        print(f"{condition} seed={seed} epoch={epoch} val_macro_ap={val_ap:.6f}")
        if val_ap > best_ap:
            best_ap, best_epoch, stale = val_ap, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 4:
                print("early-stopping patience exhausted")

        snapshot = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': opt.state_dict(),
                    'scheduler': sched.state_dict(), 'best_ap': best_ap, 'best_epoch': best_epoch,
                    'best_state': best_state, 'stale': stale, 'history': history,
                    'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                    'augmentation_rng': loader.dataset.rng.bit_generator.state}
        torch.save(snapshot, resume_path + '.tmp'); os.replace(resume_path + '.tmp', resume_path)
    torch.save(best_state, f"{out_dir}/{condition}_seed{seed}.pt")
    model.load_state_dict(best_state)
    np.save(f'{out_dir}/{condition}_seed{seed}_val_probs.npy', predict(model, xv_t))
    with open(f"{out_dir}/{condition}_seed{seed}_history.json", "w") as f:
        json.dump({"best_epoch": best_epoch, "best_val_macro_ap": best_ap,
                   "preproc": {"mean": mean, "std": std}, "history": history}, f, indent=2)
    with open(done_path, 'w') as f:
        json.dump({'condition': condition, 'seed': seed, 'epochs_run': len(history),
                   'best_epoch': best_epoch, 'run_kind': 'new_reconstruction',
                   'memory_format': 'channels_last', 'precision': 'float32'}, f, indent=2)


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
    lin.load_state_dict(results[best]['state'])
    np.save(f'{out_dir}/B1_val_probs.npy', predict(lin, xv_t))
    with open(f"{out_dir}/B1_selection.json", "w") as f:
        json.dump({"selected_weight_decay": best,
                   "grid": {str(k): v["val_macro_ap"] for k, v in results.items()}}, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["B1", "B2", "B3"], required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--archive", default="chestmnist_64.npz")
    ap.add_argument('--out', default='outputs')
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    data = load_archive(a.archive, splits=['train', 'val'])
    if a.condition == "B1":
        train_linear(data, out_dir=a.out)
    else:
        assert a.seed in (20261, 20262, 20263), "registered seeds only"
        train_neural(a.condition, a.seed, data, out_dir=a.out)
