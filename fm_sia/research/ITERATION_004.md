# Iteration 004: deterministic SIA magnitude readout

Broad direction: SIA branch and joint learning, trial 3 of at most 5. Count
the two earlier GradNorm trials 001 and 002 against this same broad budget;
do not reset it for a new head or loss weight. The frozen GradNorm selection
remains closed. D4 was a separate geometric augmentation direction and failed
its validation criterion; this attempt does not use augmentation.

## Literature and design rationale

Gneiting, Making and Evaluating Point Forecasts (JASA 2011),
https://arxiv.org/abs/0912.0902 , explains matching a point forecast to its
scoring function. Absolute error targets a conditional median. This motivates
a direct MAE-trained magnitude estimate rather than interpreting one random
joint-flow sample as the point estimate. It does not prove better performance.

Lipman et al., Flow Matching (ICLR 2023), https://arxiv.org/abs/2210.02747 ,
provides the velocity-regression framework retained for the image branch.
The new SIA head is deterministic regression, not a new flow-matching method
or calibrated uncertainty estimator.

## Preregistered intervention

Retain the preoperative condition encoder and clinical FiLM image UNet, paired
image flow target, masked image MSE and Euler sampler. Replace the SIA ODE and
its bidirectional interaction with a preoperative-only magnitude readout:
global-average-pool the encoder's final 512-channel map, concatenate the shared
clinical global embedding, then Linear(640,128), SiLU, Linear(128,1), softplus.
The SIA graph never reads the postoperative image, interpolated image, time,
or random sample. Shared preoperative/clinical encoders receive both losses.

Initialize the final readout weights to zero and its bias to inverse-softplus
of the TRAINING-fold SIA median (clamped at 1e-4 D only for zero medians).
Save that median with training-fold statistics. Compare against the unchanged
training-median control to distinguish learned signal from constant prediction.

Loss = masked image velocity MSE + 1.0 * physical SIA MAE / 1 D. The old
auxiliary SIA term is disabled. No GradNorm/DWA, EMA or augmentation. This is
a bundled head replacement (target representation, objective and interaction
change together); gains cannot be attributed to any one component without
further ablation. Vector/axis prediction is not available for this model.

Keep four baseline clinical inputs, frozen first three folds, 200 epochs,
batch 8, AdamW lr=0.0001 and its existing scheduler, seed 42, evaluation seed
2026, 100 Euler steps and one evaluation draw. All metadata and predictions
must identify deterministic scalar output. Existing flow defaults/checkpoints
must retain their original behavior.

Primary development criterion for this SIA-focused intervention: equal-fold
mean validation SIA MAE at the SAME image-MAE-selected checkpoints used by
the baseline reporting policy. Do not switch to SIA-selected checkpoints.
Image MAE/RMSE/SSIM and SIA RMSE remain separate secondary outcomes. Compare
MCPTP, base FM-SIA and a training-only median control on validation; freeze
the chosen configuration before any planned test evaluation. Earlier baseline
and GradNorm test results make this an exploratory benchmark.

## Implementation and audit

Add an explicit --sia_prediction direct option with backward-compatible flow
default. Acceptance: training-only median, no post/time/noise path to SIA,
finite shared and head gradients, deterministic noise-invariant inference,
nonnegative physical output, loss-weight accounting, saved-checkpoint inference
roundtrip and unchanged original tests. A real-data smoke precedes full training.

## Experiment and analysis

The 35-test audit and one-fold real-data smoke passed. All three 200-epoch
validation-only folds completed; the post-run audit verified unchanged image/
clinical normalization and training-only scalar statistics/median.

Validation SIA MAE at image-selected checkpoints improved over the two learned
SIA baselines, but did not improve over the training-median constant control.
This limits the evidence for useful individualized predictions. Image MAE
also improved over base FM-SIA but remained above the strongest image baseline.

Freeze this configuration and its existing best_image checkpoints before one
test evaluation; do not tune more head/loss variants from the test outcome.
The broad SIA/joint-learning budget stops at 3/5 trials. Retain all private
metrics, predictions and timing, including the constant-control comparison.
