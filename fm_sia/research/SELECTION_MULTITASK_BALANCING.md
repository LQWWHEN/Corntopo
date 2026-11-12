# Frozen selection: multitask balancing

Two full three-fold trials were completed against the five-trial maximum.
Select iteration 002 (GradNorm alpha=0, branches, controller lr=0.001), with
training source d0921814e226884d80f12765c2240dbeb92ded48. Selection used equal-fold
mean validation image MAE, as declared in iteration 001. Use each fold's
existing best_image checkpoint; do not select a SIA-specific checkpoint.

Close this direction without further tuning. The one planned test evaluation
uses the frozen folds, saved training-only statistics, no EMA, 100 Euler steps,
one draw, and seed 2026. Report image and SIA results separately, alongside
the unchanged eight-model baseline and training time. Keep the test result
whether positive or negative; do not revisit this selection after seeing it.

This is exploratory benchmarking because earlier baseline test results have
already been observed. Three folds with overlapping training sets do not
support claims of independent replicates or confirmed statistical superiority.
FNO also differs in input modality (preoperative image only). Preserve that
distinction when interpreting the comparison.

## Completed test status

The frozen alpha=0 candidate was evaluated once on all three test partitions.
The image MAE improved modestly relative to base FM-SIA while SIA MAE worsened;
the strongest corresponding baseline was not surpassed. Retain this mixed
outcome without revising the frozen selection. Metrics, native-size turbo
visuals, prediction arrays, source hashes and timing are archived privately.
