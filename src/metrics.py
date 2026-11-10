"""Evaluation measures exactly as defined in Sections 4.7-4.8 of the manuscript."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score


def macro_ap(y, p):
    return float(np.mean([average_precision_score(y[:, j], p[:, j]) for j in range(y.shape[1])]))


def per_label_ap(y, p):
    return [float(average_precision_score(y[:, j], p[:, j])) for j in range(y.shape[1])]


def macro_auc(y, p):
    return float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1])]))


def per_label_auc(y, p):
    return [float(roc_auc_score(y[:, j], p[:, j])) for j in range(y.shape[1])]


def macro_brier(y, p):
    return float(np.mean((p - y) ** 2))


def select_thresholds(y_val, p_val):
    """Per-label threshold maximising validation F1; highest threshold on ties (eq. 10)."""
    ts = []
    for j in range(y_val.shape[1]):
        cand = np.unique(np.round(p_val[:, j], 6))
        best_f1, best_t = -1.0, 0.5
        for t in cand:
            f1 = f1_score(y_val[:, j], (p_val[:, j] >= t).astype(int), zero_division=0)
            if f1 > best_f1 or (f1 == best_f1 and t > best_t):
                best_f1, best_t = f1, t
        ts.append(float(best_t))
    return ts


def paired_cluster_bootstrap(y, p_a, p_b, clusters, n_rep=2000, seed=77103):
    """Paired patient-prefix-cluster bootstrap of macro-AP difference (B - A).

    Both conditions are evaluated on the same resampled clusters in every
    replicate; returns the observed difference and the 95% percentile interval.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    index = {c: np.where(clusters == c)[0] for c in uniq}
    obs = macro_ap(y, p_b) - macro_ap(y, p_a)
    diffs = np.empty(n_rep)
    for r in range(n_rep):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([index[c] for c in chosen])
        diffs[r] = macro_ap(y[rows], p_b[rows]) - macro_ap(y[rows], p_a[rows])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"observed": obs, "ci95": [float(lo), float(hi)], "replicates": diffs.tolist()}
