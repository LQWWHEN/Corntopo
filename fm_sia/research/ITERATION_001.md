# Iteration 001: shared-gradient task balancing

## Literature and hypothesis

GradNorm (Chen et al., ICML 2018) balances task gradient magnitudes using relative
training rates: https://proceedings.mlr.press/v80/chen18a.html .
Flow Matching (Lipman et al., ICLR 2023) supplies the underlying velocity
regression objective: https://arxiv.org/abs/2210.02747 .
DWA in Liu et al. (CVPR 2019) is a possible later comparison, not part of the
first trial: https://arxiv.org/abs/1803.10704 .

Hypothesis: balancing gradients at the common bottleneck can reduce competition
between image velocity and the combined SIA objective. Different scalar loss
sizes alone do not establish gradient imbalance. Record measured gradient norms,
relative training rates and task weights; interpret them alongside validation.
This is an application of existing methods, not a claim of methodological novelty.

## Design fixed before the first training run

Direction: multitask balancing. Budget: at most FIVE full three-fold parameter
trials for this entire direction, including unsuccessful configurations. A smoke
run validates execution only and cannot be used to choose hyperparameters.

Trial 1 changes only fixed weighting to GradNorm: branch grouping, alpha=1.5,
weight optimizer learning rate=0.001. Existing image/SIA base coefficients remain
unchanged. Use the existing vector FM-SIA architecture, FiLM, four preoperative
clinical inputs, no augmentation, no EMA, 200 epochs, batch 8, learning rate
0.0001, seed 42, evaluation seed 2026, 100 Euler steps, one evaluation draw.
Reuse the first three frozen patient-grouped folds and original data-quality
policy. All preprocessing statistics remain training-only.

Primary development criterion: equal-fold mean validation image MAE at the
image-selected checkpoints. SIA MAE at those same checkpoints is a secondary
diagnostic. Compare with validation scores of the existing fixed-weight FM-SIA
and strongest image baseline using their saved histories; do not tune against
their test numbers. A lower primary value is a descriptive improvement, not a
significance test. Stronger external or nested validation is required for a
confirmatory generalization claim.

Development runs use --validation_only: construct no test DataLoader and write
no test predictions or metrics. After the direction's candidate selection is
frozen, evaluate that selected candidate once on the existing test partitions.
Previously observed baseline test results make this an exploratory benchmark;
do not label those partitions untouched holdouts. Repeated CV training subsets
overlap other folds' evaluation subsets; these are per-fold models, not one
globally held-out test cohort.

## Implementation and audit acceptance

Reuse the repository's existing GradNorm controller, rather than claiming its
implementation is newly invented. Shared parameters are the bottleneck conv
before SIA interaction. Gradient targets are detached; controller gradients do
not accumulate in model parameters. Positive controller weights sum to the task
count and controller optimizer/reference losses survive checkpoint recovery.

Acceptance: fixed-loss numerical equivalence; isolated controller gradient;
finite connected image/SIA gradients; weight normalization; checkpoint round
trip; validation-only runs have zero test evaluations; baseline defaults are
unchanged; evaluation partition explicitly recorded. No credentials, private
tables, sample predictions, patient identifiers, checkpoints, or server details
belong in public Git commits. Keep private experiment registry and results
outside the repository. Archive the precise source commit with each run.

## Experiment and analysis status

Pending actual run; no improvement or experimental result is claimed here.
Record failures and negative results privately with the same care as successes.
Future trial decisions must include the validation evidence and remaining budget.
