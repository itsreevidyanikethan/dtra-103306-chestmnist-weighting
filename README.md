# DTRA-103306 — Positive-class weighting on ChestMNIST

The original experiment archive is unavailable. This package implements an
explicitly **new reconstruction**, with real-data integrity audits, resumable
training and validation-frozen test evaluation. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for actual execution status.

## Data and source mapping

Obtain `chestmnist_64.npz` from the official
[MedMNIST+ v3.0 record](https://zenodo.org/records/10519652).
The required MD5 is `9de6cd0b934ebb5b7426cfba5efbae16`.
Download `chestmnist_split_info.csv` from the
[official mapping folder](https://drive.google.com/drive/folders/1A_99qH_c-J0p_SatwSiaP_i1CvLUOzVo),
linked by the [MedMNIST repository](https://github.com/MedMNIST/MedMNIST).
The preparation command normalises `image_id` to `source_filename`.

```bash
pip install -r requirements.txt
python src/prepare_data.py --mapping-original chestmnist_split_info.csv
python src/reconstruct.py --stage audit --threads 4
python src/reconstruct.py --stage train --threads 4
python src/reconstruct.py --stage freeze --threads 4
python src/reconstruct.py --stage evaluate --threads 4
python make_checksums.py outputs --output CHECKSUMS.txt
python make_checksums.py --verify CHECKSUMS.txt
```

Each stage must succeed before the next one. `train` resumes epoch snapshots,
including optimizer, scheduler and random-number states. Use the same software,
thread count, data and source for deterministic continuation. CPU training of
six CNNs can take several hours. Interrupted runs are not completed experiments.
The test stage refuses changed model artifacts or protocol files and supports
reanalyzing saved predictions without reloading raw test data.

## Models and evidence

B0 uses training prevalence. B1 selects a linear pixel model's weight decay on
validation macro AP. B2 and B3 use a matched 99,438-parameter CNN with unweighted
and positive-weighted BCE respectively, seeds 20261/20262/20263, AdamW, batch 64,
at most 20 epochs, patience 4, and validation-selected checkpoints.
Per-label thresholds and probability-mean ensembles are frozen before test
performance is calculated. Evaluation saves individual and ensemble predictions,
classification/calibration metrics, duplicate-exclusion sensitivity and 2,000
paired patient-prefix-cluster bootstrap replicates.

The original `evaluate.py` is retained for provenance; use `reconstruct.py` for
the complete new pipeline. Parameters are implemented in the entry points;
editing the documentary YAML alone does not change training.

## Verification

```bash
python src/model.py
python tests/smoke_test.py
python tests/test_reconstruction.py
python tests/test_checksums.py
python make_checksums.py --verify SOURCE_CHECKSUMS.txt
```

Synthetic regression tests cover threshold equivalence, model freezing,
saved-prediction evaluation and bit-identical interrupted/resumed training on a
small fixture. They do not validate any manuscript metric. Dataset: CC BY 4.0;
code: MIT. Raw images are not redistributed here.
