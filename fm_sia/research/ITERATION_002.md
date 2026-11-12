# Iteration 002: remove relative-training-rate pressure

Direction: multitask balancing, trial 2 of at most 5, not a new budget.

## Literature and design

Reference: Chen et al., GradNorm, ICML 2018,
https://proceedings.mlr.press/v80/chen18a.html . In the GradNorm target
G_mean * relative_rate**alpha, alpha=0 makes the nonzero-rate target simply
G_mean. This is a parameter ablation of an existing method, not a novel method.

Trial 1's private validation/gradient diagnostics showed that the observed
relative-rate targets and task-weight trajectory did not produce the desired
image result. Test whether removing relative-rate pressure helps. This is a
hypothesis; a shared-layer gradient proxy does not establish whole-model
gradient balance or guarantee either task improves.

Change ONLY gradnorm_alpha from 1.5 to 0.0. Keep branch grouping, controller
learning rate=0.001 and all other architecture, input, data, seed, training,
sampling, no-EMA, selection and validation-only settings from iteration 001.
The machine-readable overrides are in configs/iteration002.json.

## Implementation, audit, experiment and analysis

The existing controller implements this ablation without a new algorithm.
Use the audited controller and validation-only training path. Its argument
validation permits alpha=0. Preserve exact baseline fixed-loss behavior.
Runtime assertions must confirm three completed 200-epoch folds, no EMA,
validation reporting, and no test-output directories.

Compare the predeclared validation image MAE criterion with trial 1, fixed
FM-SIA, and the image baseline. SIA remains a secondary diagnostic. Keep
results private; do not inspect test outcomes to decide subsequent trials.
All three 200-epoch validation-only runs completed. The 16-test suite, real-data
smoke and post-run comparison audit passed. The configuration and per-fold
normalization matched the fixed FM-SIA baseline for the declared controls.
Private validation image MAE was lower than trial 1 and fixed FM-SIA, but did
not beat the strongest image baseline. The final SIA multiplier became very
small despite comparable gradient magnitudes at the monitored layer; gradient
equality alone is not a prediction-quality guarantee.

The direction is closed after two trials; unused budget is not a requirement
to continue tuning. Alpha=0 is selected using only the predeclared validation
image MAE. Freeze its existing image-selected checkpoints and source commit
before a single test evaluation. No choice of checkpoint or hyperparameter
will be made from that test outcome. Quantitative results remain private.
