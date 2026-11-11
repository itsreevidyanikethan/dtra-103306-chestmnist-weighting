# Reproducibility status for DTRA-103306

The source package has been checked on synthetic data, including the
99,438-parameter network assertion. The original experiment archive and exact
reported results have not been verified. No real-dataset training or locked-test
evaluation was performed for this maintenance update.

## Original evidence required

| Evidence | Status in this repository |
|---|---|
| Executed source and exact original software environment | Original-run provenance not verified; current source and validation environment are available |
| Source mapping (`split,index,source_filename`) and split/audit records | Missing |
| Six selected B2/B3 neural checkpoints and B1 baseline checkpoint | Missing |
| Validation histories, thresholds and preprocessing records | Missing |
| Locked-test labels, row identities and per-model/ensemble predictions | Missing |
| Aggregate/per-label results and duplicate-sensitivity outputs | Missing |
| Original 2,000 paired patient-prefix-cluster bootstrap replicates | Missing |
| Original artifact checksums and persistent archive DOI | Missing |

## Implementation limits identified

`src/evaluate.py` evaluates the B2/B3 ensembles only; it does not evaluate the B0
prevalence or B1 linear baseline, despite its broader module description.
Validation threshold selection is implemented in `metrics.py` but not used or
exported by the evaluation entry point. The duplicate-audit helper is not invoked
by the documented training/evaluation commands, and the duplicate-exclusion
sensitivity analysis is not implemented. The `source_mapping.csv` required by
the evaluation command is absent. Test probabilities are saved without a
companion export of test labels and row/cluster identities. The early-stop
condition uses `stale > 4`, which stops after five non-improving epochs rather
than the four stated in the configuration. Protocol YAML is documentation and
is not loaded by the training scripts.

These scientific implementation choices were not changed in the maintenance
update. Agreement with the executed study must be resolved from the original
source and outputs, not inferred from synthetic tests.

## Completion procedure

Recover the original experiment directory, source mapping and checkpoint files.
Verify the official data archive, mapping coverage and split audits, then check
saved predictions, selected checkpoints, thresholds, baseline results, sensitivity
analysis and bootstrap records against the manuscript. Resolve source/protocol
differences before claiming exact reproduction. Generate artifact checksums and
archive the permitted evidence with a persistent identifier. The present code
release is not the complete reconstruction archive described in the manuscript.
