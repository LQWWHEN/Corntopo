# Iterations 001–020: completed campaign and pause

All twenty planned attempts have completed training, result analysis and artifact
audit. Eight attempts reached frozen test evaluation; twelve stopped after
validation. Trial 004 remains a magnitude-only diagnostic ablation. Trial 002
can produce a vector but its historical report did not formally evaluate axis
error. Neither is evidence of a fully audited joint-task improvement.

The campaign is paused at the user's request after iteration020 and the private
summary report. No iteration021 has been started. A later explicit user request
to resume is required before further experimentation.

## What the experiments established

The main development path is 005 → 009 → 012: a direct preoperative axial-moment
SIA readout, conditional Fourier image velocity, and magnitude/periodic-axis
parameterization. Trial 010 separately adds laterality and a reversible mirror
reference frame to 009. Parents identify recipes trained from scratch, not
inherited pretrained weights; see [MODEL_LINEAGE.md](MODEL_LINEAGE.md).

Within the tested complete joint models, 010 is the strongest image reference
and 012 the strongest magnitude/axis reference. They are distinct models; their
best metrics must not be combined into a fictitious single-model result. The
independent FNO image baseline and training-only SIA constant controls remain
competitive. Lower average SIA errors have not resolved individual high-magnitude
underprediction. No candidate dominates every comparison.

The final two attempts used 012 as their direct parent. Trial 019 added signed
valid-neighbour image-gradient supervision and failed the SSIM and magnitude
validation guards. Trial 020 added a mean-free local convolutional velocity
branch alongside the Fourier network. Its SSIM improved across folds, but its
PSNR guard failed; the very small difference is not a statistical inferiority
claim. Both outcomes are retained as mixed/negative validation evidence, without
test evaluation or a post-result change to their gates.

## Lessons by direction

- Augmentation: synchronizing images and doubled-angle labels is necessary;
  D4 and paired spherical offsets did not establish an overall benefit.
- Clinical inputs: extra preoperative vector inputs did not help. The laterality
  studies support a joint coordinate/condition design, not arbitrary feature
  expansion; removal of the original clinical variables has not been studied.
- Fusion: preoperative reference-frame features and predicted SIA spatial
  conditioning did not establish a complete improvement.
- SIA and joint learning: the direct readout and polar objective were more useful
  than GradNorm in this campaign. Parameterization, losses and initialization
  changed together in 012, so their effects are not individually identified.
- Architecture: conditional Fourier velocity helped image prediction; capacity
  and conditioning also changed, preventing attribution solely to Fourier mixing.
- Image objectives: physical-unit, angular-shape and gradient losses did not
  yield a fully accepted joint improvement.
- Contrastive/autoregressive approaches: the contrastive validation benefit did
  not persist on test. Cross-task prediction coupling is not evidence of a fully
  developed autoregressive model; strict autoregressive modeling was not run.

## Audit and retained records

The protocol uses the same frozen first three patient folds, 200 epochs and no
EMA. Image reports include PSNR, SSIM, RMSE and MAE. SIA magnitude, periodic axis
and doubled-angle vector errors are reported separately, with predefined axis
coverage. Validation and test are separate tables. Measured training costs include
historical repeat runs explicitly.

The private seventeen-page summary contains all twenty attempts, eight original
baselines, controls, actual per-fold results, plots, timing and source hashes.
Numeric patient-data results, individual predictions, images, deployment records
and credentials are not included in this public repository.

Experiment checkpoint and optimizer-state residuals are zero after completed
rounds. The pre-existing Git-tracked LPIPS dependency asset is unchanged and is
not an experiment checkpoint. Metrics, predictions, plots, logs, configurations
and source records remain in the private archive. Proposal budgets are retained
without being reset; no proposal exceeded its five-trial limit.

## Interpretation notes

Repeated exploration of this cohort does not provide an untouched external
validation set. Fold standard deviations are descriptive, not confidence
intervals. Different model inputs, capacity and training costs remain relevant
despite a common evaluation protocol. These findings do not establish clinical
utility or uniform superiority over the strongest controls.
