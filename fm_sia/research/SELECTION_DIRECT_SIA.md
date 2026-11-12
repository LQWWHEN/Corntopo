# Frozen selection: direct SIA magnitude head

Select iteration 004, source 0149bb9c7b96351f6deb98fadb44d651e47bef60, without
further tuning. Use each fold's existing image-MAE-selected best_image.pth;
do not switch to best_sia.pth. Selection follows the preregistered validation
SIA criterion at the baseline-compatible image checkpoints. The validation
training-median control remains stronger, so do not claim individualized
predictive skill from learned-baseline superiority alone.

Before test execution, retain checkpoint SHA256 hashes and this decision.
Run the same frozen three test folds once, no EMA, original orientation,
100 Euler steps, saved training-only statistics and one evaluation draw.
Report scalar SIA magnitude metrics separately from image metrics. Vector
metrics are unavailable. This head produces no predictive uncertainty.

The exploratory comparison includes the unchanged eight-model baseline,
the prior frozen GradNorm candidate and the training-only median control.
Keep positive and negative outcomes. Do not revisit the frozen selection
after observing this test result. Close the current SIA/joint-learning
direction at three total trials, including iterations 001 and 002.
