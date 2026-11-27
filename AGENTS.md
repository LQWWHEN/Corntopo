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
data-quality policy unless the user changes them. Inspect training at 60-minute
intervals. The latest user budget is at most 5 tuning trials per concrete
improvement proposal, not a cap on an entire broad direction. Keep cumulative
direction counts for history. Related hyperparameter variants of the same
proposal share its budget; do not rename a variant to reset its count.
Commit each attempt to GitHub.
Latest user stop condition (2026-11-14): finish iteration020 and its result audit,
prepare a private summary report covering iterations001-020, then pause the goal.
2026-11-15 update: the user authorized ONE new iteration021 after directory
reorganization, including research, implementation, audit, 3 folds x 200 epochs
and reporting. Stop after021; do not start022. The older pause remains historical.
Completion record (2026-11-14): iterations001-020 and the private seventeen-page
summary have passed final audit. The campaign is paused; there are no active
experiment jobs or retained experiment checkpoints. See the qualitative public
record in `fm_sia/research/SUMMARY_001_020.md` before any later resumption.
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

2026-11-15 completion: iteration021 finished all three 200-epoch validation folds
and result audit. Its SIA gates failed, so no test was run; temporary weights
were cleaned. Its private report is complete and the task is paused.
No iteration022 is authorized.

2026-11-15新指令：用户明确要求“从022开始继续迭代任务”。长期迭代已恢复；此前021后暂停记录为历史状态。022已完成，后续各轮状态见公开索引和私有动态上下文。继续遵守既有协议、最多5轮方案预算及checkpoint清理规则。

2026-11-16用户更新：研究方向尝试次数不限；每个具体方案（实验组）初始实验单列，最多5次超参数实验，同组学习率/损失权重变体共享预算。历史预算仅留档。历史过门槛且取得提升的方案也安排调参；PSNR、SSIM纳入正式筛选。026前回顾和计划见repo/fm_sia/research/TUNING_POLICY_026.md及私有tuning_review_pre026。当前不启动训练。

2026-11-16新授权：用户要求开始补充超参数实验，完成后继续长期迭代。前述“当前不启动训练”已被覆盖。按TUNING_POLICY_026执行，每组先3次、最多5次；组内验证选优后才测试最终候选。

2026-11-16命名更新：超参数实验采用基础方案编号加组内序号，如016_1、016_2；与新方案迭代编号分开。当前授权开始补充超参数实验，完成后继续长期迭代；026保留给下一新方案。每实验组最多5次调参，方向不限。

2026-11-18最新用户指令：汇总本批补充超参数实验，先不要开始026。九组共41次调参已完成，所有组关闭、无活动训练且实验权重清零；当前暂停于026之前。此前“继续026”的计划暂不执行，须用户新指令恢复。

2026-11-19：用户恢复长期迭代，从026继续；每60分钟检查训练。每轮指标表逐指标最优加粗、次优加下划线。

2026-11-24 用户最新范围：完成 050 全部适用训练、组内调参、冻结测试选择及清理；暂停后续迭代并提交 001–050 阶段报告。后续实验仅提出方向，恢复训练需用户新指令。 已预登记的 051 保留源代码和审计记录，原服务器启动等待器已撤下。

2026-11-26 恢复长期研究。先对 010、012、030 作模块消融，并从基础 FM-SIA 开独立分支。051 保留停派。验证保护仅一至两项不通过可列入组末冻结测试候选。 每个具体方案初始项单列，近门槛及达标方案调参 3–5 次，每 60 分钟检查训练。

2026-11-27 用户更新：065 为最后一个新方案。完成截至 065 已适用的调参、组末冻结测试和清理后结束迭代。交付单一最终模型、MICCAI 风格论文初稿及必要的消融补充。论文和定量结果保留私有。
