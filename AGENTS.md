# CornTopo iteration requirements

Read `fm_sia/research/README.md` and `fm_sia/research/iterations.json` before
planning an experiment; do not repeat completed configurations without a stated
new hypothesis. Update the index for every attempt and retain negative results.

The full task requires postoperative image prediction AND BOTH SIA magnitude
and astigmatic axis. Magnitude-only models are diagnostic ablations, not complete
solutions. Use the project's doubled-angle vector convention and periodic axis
errors; predeclare how undefined/near-zero axes are handled.

Default to NO persisted checkpoints, including best, final, last, optimizer,
EMA, or periodic weights. Never enable `--save_checkpoints` unless the user
explicitly requests checkpoint saving. Use the maintained `fm_sia/cddpm/train.py`
and `cgan_baseline/train.py` entry points; old notebooks/archived training code
are historical references and must not be launched without applying this policy.
Memory-only best weights may be used during a live run and must be released on
exit. Keep metrics, predictions, plots, logs, configs, source commits and timing.
Do not save a disguised weight snapshot in an archive or other extension.

Keep the frozen first 3 of 10 patient folds, 200 epochs, no EMA, and the existing
data-quality policy unless the user changes them. Inspect training at 30-minute
intervals. The latest user budget is at most 5 tuning trials per concrete
improvement proposal, not a cap on an entire broad direction. Keep cumulative
direction counts for history. Related hyperparameter variants of the same
proposal share its budget; do not rename a variant to reset its count.
Commit each attempt to GitHub.
Never publish private datasets, individual predictions/images, credentials,
connection details or keys. Public iteration records contain methods only;
quantitative patient-data results remain in the private experiment registry.

Without checkpoints, validation-only trials cannot be tested later by loading
weights. Before training, either predeclare final evaluation in the same live
run or plan a deterministic rerun of a frozen configuration. Do not silently
enable persistence to preserve the previous workflow; report rerun cost and
verify its validation output before using its test result.
