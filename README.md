# DTRA-103306 — Positive-Class Weighting on ChestMNIST: Reproduction Package

## Verification and artifact status

This repository currently contains source code and a protocol configuration.
The original experiment checkpoints, row-level manifests and prediction archives
are not included. Synthetic smoke tests check software execution only; they do
not reproduce or validate the manuscript's numerical results. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the remaining evidence and code gaps.

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
python src/model.py                           # architecture check (99,438 params)
python src/data.py                            # import check only; does not audit data
python src/train.py --condition B1            # linear baseline + wd grid
for s in 20261 20262 20263; do
  python src/train.py --condition B2 --seed $s
  python src/train.py --condition B3 --seed $s
done
python src/evaluate.py --mapping source_mapping.csv   # single locked-test stage
python tests/smoke_test.py                    # fast end-to-end check, no dataset
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

## Check this code release

Run these commands from the repository root, after installing the dependencies:

```bash
python tests/smoke_test.py
python tests/test_checksums.py
python make_checksums.py --verify SOURCE_CHECKSUMS.txt
```

`SOURCE_CHECKSUMS.txt` covers the distributed source and validation records.
`validation/validation.json` records the actual synthetic checks performed for
this code update and their software versions; it is not the original study
environment. The `python` commands intentionally omit isolated mode (`-I`),
because training/evaluation scripts import neighbouring modules under `src/`.

Once authentic run outputs have been recovered and reviewed, verify their bytes
separately:

```bash
python make_checksums.py outputs --output CHECKSUMS.txt
python make_checksums.py --verify CHECKSUMS.txt
```

The generator fails for a missing or empty artifact directory. Do not describe a
source checksum manifest as verification of unavailable experiment outputs.
Keep newly generated results labelled as new runs until their provenance and
agreement with the manuscript have been assessed. Large checkpoints should be
archived separately with an immutable identifier and checksums. The `.gitignore`
excludes local datasets and model outputs from accidental source commits.
