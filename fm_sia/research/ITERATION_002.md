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
Experiment pending at this preregistration commit; no success is claimed.
