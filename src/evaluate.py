"""Single locked-test evaluation: baselines, six checkpoints, ensembles,
paired patient-prefix-cluster bootstrap, and label-level corroborating tests."""
import argparse
import json
import os

import numpy as np
import torch
from scipy import stats

from data import load_archive, train_statistics, standardise
from metrics import (macro_ap, macro_auc, macro_brier, per_label_ap,
                     per_label_auc, paired_cluster_bootstrap, select_thresholds)
from model import CompactCNN
from train import predict


def load_prefix_clusters(mapping_csv, n_test):
    """Patient-prefix cluster ids for the test rows from the official source
    mapping CSV (columns: split,index,source_filename). NIH filenames are
    patientID_followup.png, so the prefix before '_' is the patient proxy."""
    import csv
    clusters = np.empty(n_test, dtype=object)
    with open(mapping_csv) as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                clusters[int(row["index"])] = row["source_filename"].split("_")[0]
    assert not any(c is None for c in clusters)
    return clusters


def main(archive, mapping_csv, out_dir="outputs"):
    data = load_archive(archive)
    (xt, yt), (xv, yv), (xe, ye) = data["train"], data["val"], data["test"]
    mean, std = train_statistics(xt)
    xe_t = torch.tensor(standardise(xe, mean, std)).unsqueeze(1)
    xv_t = torch.tensor(standardise(xv, mean, std)).unsqueeze(1)

    ens = {}
    for cond in ("B2", "B3"):
        ps = []
        for seed in (20261, 20262, 20263):
            m = CompactCNN()
            m.load_state_dict(torch.load(f"{out_dir}/{cond}_seed{seed}.pt"))
            ps.append(predict(m, xe_t))
        ens[cond] = np.mean(ps, axis=0)
        np.save(f"{out_dir}/{cond}_ensemble_test_probs.npy", ens[cond])

    summary = {c: {"macro_ap": macro_ap(ye, p), "macro_auc": macro_auc(ye, p),
                   "macro_brier": macro_brier(ye, p),
                   "per_label_ap": per_label_ap(ye, p),
                   "per_label_auc": per_label_auc(ye, p)} for c, p in ens.items()}

    clusters = load_prefix_clusters(mapping_csv, len(ye))
    boot = paired_cluster_bootstrap(ye, ens["B2"], ens["B3"], clusters)
    summary["primary_contrast_B3_minus_B2"] = {k: boot[k] for k in ("observed", "ci95")}
    np.save(f"{out_dir}/bootstrap_replicates.npy", np.array(boot["replicates"]))

    d_ap = np.array(summary["B2"]["per_label_ap"]) - np.array(summary["B3"]["per_label_ap"])
    summary["corroboration"] = {
        "labels_favoring_B2": int((d_ap > 0).sum()),
        "sign_test_p": float(stats.binomtest(int((d_ap > 0).sum()),
                                             int((d_ap != 0).sum()), 0.5).pvalue),
        "wilcoxon_ap_p": float(stats.wilcoxon(summary["B2"]["per_label_ap"],
                                              summary["B3"]["per_label_ap"]).pvalue),
        "wilcoxon_auc_p": float(stats.wilcoxon(summary["B2"]["per_label_auc"],
                                               summary["B3"]["per_label_auc"]).pvalue),
    }
    with open(f"{out_dir}/locked_test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "B2" and k != "B3"}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="chestmnist_64.npz")
    ap.add_argument("--mapping", required=True,
                    help="official source-mapping CSV (split,index,source_filename)")
    a = ap.parse_args()
    main(a.archive, a.mapping)
