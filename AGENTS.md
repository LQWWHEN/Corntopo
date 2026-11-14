# CornTopo iteration requirements

Read `fm_sia/research/README.md` and `fm_sia/research/iterations.json` before
planning an experiment; do not repeat completed configurations without a stated
new hypothesis. Update the index for every attempt and retain negative results.

The full task requires postoperative image prediction AND BOTH SIA magnitude
and astigmatic axis. Magnitude-only models are diagnostic ablations, not complete
solutions. Use the project's doubled-angle vector convention and periodic axis
errors; predeclare how undefined/near-zero axes are handled.

The user authorizes TEMPORARY checkpoints during each experiment (2026-11-14).
Use `--save_checkpoints` for validation-only training, load the selected weights
for test, verify results, then remove all experiment weights and optimizer states.
Rejected validation trials are cleaned after their decision/results are archived.
Interrupted runs may retain recovery state until recovered or abandoned. Do not
retain or archive checkpoints after a completed round unless explicitly requested.
Use the maintained `fm_sia/cddpm/train.py`
and `cgan_baseline/train.py` entry points; old notebooks/archived training code
are historical references and must not be launched without applying this policy.
Memory-only selection remains available without `--save_checkpoints` for small
engineering checks. Keep metrics, predictions, plots, logs, configs, commits and timing.
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

Use `fm_sia/cddpm/test_saved_run.py` to verify selected validation outputs and
test a frozen saved run without retraining. After audit, use
`fm_sia/cddpm/checkpoint_lifecycle.py` to remove only that run's checkpoints and
record cleanup. Do not launch the old checkpoint-free frozen-rerun workflow.
Every image report includes PSNR (dB), SSIM and RMSE, alongside MAE, on the
common valid region. Aggregate measured per-fold metrics; never derive mean
PSNR from mean RMSE. Keep validation and test tables separate.
Every iteration report identifies the direct parent model and source version,
the changes introduced in this round, and all cumulative changes from basic
FM-SIA. Do not imply that unselected branches were inherited. Use MODEL_LINEAGE.md.
