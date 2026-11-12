# Iteration 003: vector-consistent D4 augmentation

Direction: geometric data augmentation, trial 1 of at most 5 for this direction.

## Literature and hypothesis

Thibos, Wheeler and Horner (1997) describe the Fourier/power-vector representation
of astigmatism: https://doi.org/10.1002/j.1538-9235.1997.tb04707.x . Retain this
project's existing physical X/Y convention and scale; do not replace it with
another power-vector convention. For doubled-angle coordinates, a 90-degree
image rotation negates both components, a 180-degree rotation leaves them
unchanged, and one image-axis reflection negates Y only. Norm is preserved.

Cohen and Welling (ICML 2016) motivate exploiting discrete geometric symmetries:
https://proceedings.mlr.press/v48/cohenc16.html . Their group-equivariant network
is distinct from the random augmentation tested here. No exact architectural
equivariance or clinical symmetry is claimed. Surgical location and anatomy
may make rotational augmentation harmful; that is an empirical limitation.

Hypothesis: additional paired geometric views reduce overfitting with limited
training pairs. Test this independently against BASE FM-SIA, with FIXED task
weights. Do not combine it with the previous GradNorm candidate.

## Design

Change only training augmentation from none to the existing d4 implementation.
Transform preoperative image, postoperative target and both masks together,
then transform the physical SIA vector before its model-side standardization.
The four baseline clinical inputs are nondirectional scalar measurements and
remain unchanged. No interpolation, rescaling, translation or new data is used.
Sixteen equally likely flip/rotation parameter combinations represent the eight
D4 transformations uniformly (each occurs twice).

Keep first three frozen patient folds, 200 epochs, batch 8, lr=0.0001, seed 42,
evaluation seed 2026, no EMA, FiLM, 100 Euler steps, one draw, original training
normalization, data-quality policy, and image-MAE checkpoint selection. Training
alone is augmented; validation and eventual test images stay in their original
orientation. Primary selection is mean validation image MAE; SIA is secondary.
Development is validation-only. Freeze a direction candidate before one planned
test evaluation, and retain unsuccessful attempts in the private ledger.

## Implementation and audit

Reuse DataTransform and the existing paired dataset implementation. Add an
independent synthetic quadratic-field oracle for all 16 transformations to
verify the spin-two label law, and a dataset integration check covering masks,
paired maps, scalar clinical input and SIA magnitude. The existing numerical,
checkpoint and leakage tests remain applicable. Audit a real-data smoke before
full training. Source, configuration and protocol must be committed first.

## Experiment and analysis

The 33-test audit and one-fold, one-epoch real-data smoke passed. All three
200-epoch validation-only folds completed and the comparison audit passed.
Validation image MAE and SIA MAE were both worse than base FM-SIA. Retain this
negative result; do not advance this candidate to test evaluation. The direction
is closed after one full trial without exhausting its five-trial maximum.
Report validation and test separately, training time, and limitations of three
overlapping CV training sets. Keep private data and results outside public Git.
