# DTRA-103306 — Positive-Class Weighting on ChestMNIST: Reproduction Package

Code for the study *"Does training-derived positive-class weighting improve
multilabel chest-radiograph classification?"* (Digital Technologies Research and
Applications, Article ID 103306). The study compares unweighted (B2) and
inverse-prevalence positive-weighted (B3) binary cross-entropy with a matched
99,438-parameter compact CNN on the official 64 × 64 ChestMNIST split, under a
validation-frozen, three-seed, locked-test protocol.

## Registered protocol
All frozen choices are in `configs/protocol.yaml`: seeds 20261/20262/20263,
AdamW (lr 1e-3, wd 1e-4), batch 64, ≤20 epochs, ReduceLROnPlateau on validation
macro AP, per-label max-F1 thresholds, three-seed probability-mean ensembles,
and a 2,000-replicate paired patient-prefix-cluster bootstrap (seed 77103).

## Dataset (not redistributed)
Download `chestmnist_64.npz` from the official MedMNIST+ v3.0 Zenodo record
(https://zenodo.org/records/10519652, CC BY 4.0) and place it in the repository
root. `src/data.py` refuses to run unless the archive matches the registered
MD5 `9de6cd0b934ebb5b7426cfba5efbae16`.

## Reproduce
```bash
pip install -r requirements.txt
python -I src/model.py                           # architecture check (99,438 params)
python -I src/data.py                            # (optional) import check
python -I src/train.py --condition B1            # linear baseline + wd grid
for s in 20261 20262 20263; do
  python -I src/train.py --condition B2 --seed $s
  python -I src/train.py --condition B3 --seed $s
done
python -I src/evaluate.py --mapping source_mapping.csv   # single locked-test stage
python -I tests/smoke_test.py                    # fast end-to-end check, no dataset
```
`source_mapping.csv` maps each row to its NIH source filename
(`split,index,source_filename`); patient prefixes for the cluster bootstrap are
the part of the filename before the underscore.

## Outputs
`outputs/` receives checkpoints, per-epoch histories, locked-test ensemble
probabilities, the bootstrap replicates, and `locked_test_summary.json`
(macro/per-label AP, ROC-AUC, Brier, sign test and Wilcoxon corroboration).

## Honest-reproduction note
Re-running on different hardware or library versions yields slightly different
numbers than any one archived run; the protocol, not a bit-exact checkpoint, is
what this package freezes. Train on CPU to match the registered environment as
closely as possible.

## License
Code: MIT (see LICENSE). The ChestMNIST data remain under their own CC BY 4.0
licence and must be obtained from the official record above.
