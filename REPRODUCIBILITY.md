# Reproducibility status — 11 November 2026

The original experiment archive is missing. The new implementation and new
outputs are not evidence that the original manuscript values were obtained.

## Real-data audit completed

The official 64 x 64 archive was verified by MD5 and SHA-256. Its SHA-256 is
`e28fbed21400c50040d6a4e51f379dce89bcb24819d3878c252ca78d52d92c3e`.
The official MedMNIST row mapping was obtained and normalised without changing
row order. There are 78,468/11,219/22,433 training/validation/test rows.

The audit found 112,094 unique decoded images, 25 duplicate groups involving
51 rows, 3 cross-split duplicate groups, and 11 groups with conflicting labels.
The mapping yields 6,188 test patient-prefix clusters and 3 prefixes that cross
splits. These overlap findings limit an interpretation of the test data as
fully independent patients. The new evaluation includes exclusion of test rows
belonging to cross-split exact-image duplicate groups as a sensitivity analysis;
it does not remove every possible source of patient leakage.

## Implementation and execution

The B0 prevalence baseline and the B1 validation-selected linear baseline have
been fitted. CNN training was paused after 11 completed epochs of B2 seed 20261. Its full epoch resume snapshot is preserved. No CNN model has completed its planned training.
The current snapshot status is recorded separately in `runs/`; unless
`evaluation_complete.json` exists, no completed six-model locked-test experiment
is claimed. Partial validation AP is not a test result.

The new pipeline implements individual-seed and ensemble output, thresholds
frozen on validation labels, test row identities, duplicate sensitivity,
calibration/classification metrics, paired cluster bootstrap and model hashes.
The optimized threshold selector is tested against a brute-force reference.
A synthetic interrupted/resumed run produces identical model parameters.

Early stopping now means four stale validation epochs; the supplied code used
`stale > 4`. The new run uses channels-last float32 convolution for CPU
execution. These are explicitly new-run choices. The documentary configuration
is not automatically consumed by the original training entry points.

## Remaining evidence and interpretation

Completing a new six-model run cannot recover the missing original source,
checkpoints or random states. Finish all training, freeze thresholds, evaluate
once and archive authentic outputs before asserting new numerical conclusions.
Do not copy manuscript values into an output file or describe synthetic tests
as replication of study results. No new-run archive DOI has been issued.
