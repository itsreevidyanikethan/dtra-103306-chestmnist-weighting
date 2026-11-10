"""End-to-end smoke test on a tiny synthetic archive (no dataset download)."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def make_fake_archive(path):
    rng = np.random.default_rng(0)
    arrays = {}
    for split, n in (("train", 120), ("val", 40), ("test", 40)):
        arrays[f"{split}_images"] = rng.integers(0, 256, size=(n, 64, 64), dtype=np.uint8)
        arrays[f"{split}_labels"] = rng.integers(0, 2, size=(n, 14)).astype(np.uint8)
    np.savez(path, **arrays)


def test_pipeline():
    import torch
    from data import load_archive, train_statistics, standardise, audit, SPLITS
    from losses import make_criterion, positive_weights
    from metrics import macro_ap, paired_cluster_bootstrap, select_thresholds
    from model import CompactCNN, count_parameters

    SPLITS.update({"train": 120, "val": 40, "test": 40})
    with tempfile.TemporaryDirectory() as d:
        arc = os.path.join(d, "fake.npz")
        make_fake_archive(arc)
        data = load_archive(arc, verify=False)
        rep = audit(data, out_path=os.path.join(d, "audit.json"))
        assert rep["unique_images"] <= 200

        (xt, yt) = data["train"]
        mean, std = train_statistics(xt)
        x = torch.tensor(standardise(xt, mean, std)).unsqueeze(1)
        y = torch.tensor(yt, dtype=torch.float32)

        assert count_parameters(CompactCNN()) == 99438
        for cond in ("B2", "B3"):
            m = CompactCNN()
            crit = make_criterion(cond, yt)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
            loss = crit(m(x[:16]), y[:16]); loss.backward(); opt.step()
        w = positive_weights(yt)
        assert w.shape == (14,) and (w > 0).all()

        p = np.clip(np.random.default_rng(1).random((40, 14)), 1e-6, 1 - 1e-6)
        ye = data["test"][1]
        assert 0 <= macro_ap(ye, p) <= 1
        ts = select_thresholds(ye, p)
        assert len(ts) == 14
        clusters = np.array([f"c{i//4}" for i in range(40)])
        boot = paired_cluster_bootstrap(ye, p, np.clip(p + 0.01, 0, 1), clusters, n_rep=25, seed=1)
        assert len(boot["ci95"]) == 2
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    test_pipeline()
