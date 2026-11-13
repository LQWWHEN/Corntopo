# FRONT 角膜图与 SIA 联合 Flow Matching：第一版

> 2026-11-13 起：完整任务同时要求 SIA 幅值和轴位。标量模式只作为历史基线/消融，不构成完整任务解法。实验索引及预算见 `fm_sia/research/README.md`（仓库根目录相对路径）。
>
> 训练默认不写 checkpoint；仅用户明确要求时使用 `--save_checkpoints`。默认模式在内存中选择最佳权重并完成当次评估，保留指标、图像、预测、日志及选择epoch元数据。下文基于 `.pth` 的恢复训练/独立评估命令仅适用于用户授权保存的运行；历史实验权重已按要求删除。


本版本复用现有 ConditionEncoder、ConditionalUNet、临床参数融合、患者级 CV 和 EMA，新增二维散光状态，与 FRONT 图像共同积分。代码已通过合成数据和数学接口测试，尚未在用户真实数据上训练，默认超参数不是经过验证的最优配置。

## 安装位置

将压缩包内的 `cddpm/` 合并到项目根目录的同名文件夹。替换前保存原工程版本，以便复现旧实验。包内保留 CDDPM、BBDM、FM、FNO、UNO 的代码入口；FM-SIA 不要求安装 neuraloperator、flow_matching、torchvision 或 OpenCV。

```text
CornTopo/
  cddpm/
    train.py
    predict.py
    evaluate.py
    dataloader.py
    transforms.py
    metrics.py
    patient_cv.py
    model/
      unet.py
      encoder.py
      sia_branch.py
      fm_sia.py
      fm_common.py
      fm.py
      ...
    diffusion/scheduler.py
  data/dataset_3_SIA/
    paired_dataset.xlsx
    raw/FRONT/1_pre.npy
    raw/FRONT/1_post.npy
    ...
```

在现有 `CDEReg` 环境中运行；缺少依赖时可执行 `python -m pip install -r requirements_fm_sia.txt`。请根据 GPU/CUDA 安装适当的 PyTorch，而不是盲目替换现有可用版本。

## 与用户现有命令对应的新命令

在 CornTopo 项目根目录运行以下一整行（可直接粘贴到 Windows CMD 或 PowerShell）：

```bash
python cddpm/train.py --data_root data/dataset_3_SIA --excel_path data/dataset_3_SIA/paired_dataset.xlsx --image_size 128 --batch_size 8 --epochs 200 --lr 1e-4 --save_dir cddpm/results/FRONT/fm_sia_fusion/CA --model fm_sia --k_folds 3 --fusion_method cross_attention --conditions "pre_age_years" "pre_GENERAL DATA_Cornea Front Astig" "pre_GENERAL DATA_K Max (Front)" "pre_GENERAL DATA_Pachy Min" --input_maps FRONT --output_maps FRONT --sample_steps 100 --ode_solver euler --sia_embed_dim 32 --sia_num_heads 4 --lambda_vector 1.0 --lambda_sia 0.1 --use_ema
```

与 CDDPM 命令的区别：模型改为 `fm_sia`，结果写入新目录；删除 `--sampler ddim` 和 `--time_steps 2000`，改用 `--ode_solver euler`。FM 时间为连续的 [0,1]；`sample_steps=100` 表示 100 次 Euler 更新，包含终点更新。`heun` 和 `rk4` 分别每步调用网络 2、4 次。上例显式启用 EMA（原命令未启用），去掉 `--use_ema` 可关闭；基线比较时保持一致。

建议先在相同命令后添加 `--preflight_only`，核对文件、列、SIA 一致性和患者划分。预检会写报告，并在不存在时建立 CV JSON，但不训练模型。代码也接受 CSV 作为 `--excel_path`，主要便于自动化测试。

正式运行可以先将 `--epochs 200 --k_folds 3` 临时改成 `--epochs 1 --k_folds 1`，使用另一个 smoke 输出目录。`--num_workers 0` 适合排查 Windows 多进程加载问题。

## 数据要求和标签定义

必须有唯一 `label_index`，并存在与它对应的 `raw/FRONT/{label}_pre.npy`、`{label}_post.npy`。沿用现有 `patient_cv.py` 的患者身份字段（术前/术后 PatientID、LastName、FirstName、DateOfBirth），同一患者双眼保持同一划分。图像为至少 128×128 的二维原始矩阵，先做中心裁剪，不做 resize。

默认 SIA 标签列：

- `delta_front_X`
- `delta_front_Y`
- 可选校验列 `front_SIA_D`
- 可选质量列 `sia_valid`

这里 X/Y 是完整柱镜尺度（X=2J0、Y=2J45），SIA = hypot(delta_X, delta_Y)，单位 D。不要把半柱镜尺度的 J0/J45 列直接传入 `--sia_columns`；若标签列名不同，可用 `--sia_columns 列X 列Y`，但物理定义必须相同。模型不重新解释仪器轴位，不自动改正符号或将 SIA 幅值拆成方向向量。

若已有 `front_SIA_D` 与向量范数不一致，或 `sia_valid` 存在 false/空值，会明确报错，不静默过滤、不填零。没有 `front_SIA_D` 时从 X/Y 计算；没有 `sia_valid` 时仅能完成数值校验，报告会注明。缺失 SIA 的样本不能直接作为这版联合训练样本，需要先修复标签或另行定义缺失监督方案。

四个临床参数完全按命令给定的顺序读取。临床输入严禁真实术后信息、SIA 标签、差值列和样本编号。随访窗由数据集定义；本版本不假设一个未经确认的统一术后月份。预检报告中的术前/术后检查间隔也不等于术后随访时长。

## 路径、网络、损失

令 q=(delta_X,delta_Y)，z*=(q-mu)/sigma，epsilon~N(0,I2)，t~Uniform(0,1)：

```text
x_t = (1-t) * x_pre + t * x_post
z_t = (1-t) * epsilon + t * z*
v_image_target = x_post - x_pre
v_sia_target = z* - epsilon
```

U-Net bottleneck 先执行原有临床融合，再执行 SIA interaction：

1. 二维 z_t 和时间 embedding 投影为 32 维 token。
2. token 通过 cross-attention 查询含位置编码的图像空间特征。
3. 更新后的 token 通过零初始化的 FiLM 调节 bottleneck 图像特征。
4. U-Net 输出图像速度；小型 MLP 输出二维散光速度。

32 维 embedding 只是内部计算特征，真正被积分的状态仍然是二维；没有高维目标编码器，也没有另一个独立 SIA 标量回归头。

```text
z_clean = z_t + (1-t) * predicted_sia_velocity
q_clean = mu + sigma * z_clean
s_clean = norm(q_clean)
loss = image_masked_MSE
     + lambda_vector * MSE(predicted_sia_velocity, z* - epsilon)
     + lambda_sia * Huber((s_clean - norm(q)) / sia_ref, 0)
```

默认 `lambda_vector=1.0, lambda_sia=0.1, sia_ref=1.0 D, huber_delta=1.0`，只是起始配置。图像 MSE 先对每个样本有效像素平均，再对 batch 平均；向量 MSE 对两个分量和 batch 平均。`--lambda_sia 0` 是必须保留的消融。一步终点估计只用于训练辅助损失；验证/测试 SIA 全部来自完整 ODE 积分。

## 归一化、mask 和增强

每折仅在训练样本的术前/术后有效像素上拟合统一 min/max，两个时相共用同一变换。验证、测试不参与拟合，不单独拉伸，也不强制截断到 [0,1]。折内临床均值/标准差与 SIA 向量均值/标准差也仅由训练集拟合。

有效像素由归一化前的有限数值确定。若原始数据明确用 0 表示无效区域，显式增加 `--invalid_zero`；默认把有限的原始 0 当作真实数值。否则归一化后的最小值 0 会被错误排除。填充区域设为归一化 0。

FM 图像训练和图像评估使用术前/术后有效区域交集。目标 mask 只用于训练/评估，不传给采样器；真实推理只使用术前有效区域保存输出。SIA 分支不利用真实术后图或标签采样。

默认 `--augmentation none`。可选 `--augmentation d4` 会同步变换术前/术后图、mask 和物理 X/Y 向量；90/270 度旋转使两个分量变号，单次水平/垂直镜像使 Y 变号。该版本不处理方向性/眼别临床输入的坐标变换，常见轴位、角度、眼别字段会被拒绝；新增方向条件时必须保持增强关闭。随机增强由 worker RNG 提供，修复了原代码每次重置相同 seed 的问题。

FRONT 的原始单位取决于数据导出方式（例如曲率半径或屈光力），代码不会擅自当作 D。图像 `_raw` 指标及 `.npy` 输出保留输入矩阵原始单位。SIA 标签始终要求 D。第一版没有图像—SIA 物理一致性损失。

## 验证、checkpoint 与结果

默认第一次 epoch、每 10 个 epoch、最后一次 epoch 验证，`--eval_every` 可调整。

- `best_image.pth`：验证集完整采样的图像 MAE 最低。
- `best_sia.pth`：验证集完整采样的 SIA MAE 最低。
- `best.pth`：训练结束后复制预先选定的主要 checkpoint，默认 `--best_metric MAE`；若研究以 SIA 为主，可在训练前指定 `--best_metric SIA_MAE`。
- `last.pth`：可恢复的最近 checkpoint，包含 optimizer、scheduler、EMA、RNG、训练加载器 RNG、划分和全部归一化参数。
- `final.pth`：最后一个训练 epoch 的 checkpoint。

同一次实验的自动测试只评价 `best` 与 `final`，不会根据测试结果重新选择最优模型。每折的另一种 best checkpoint 可单独评估，但不能看过测试集后更改主结果的选择规则。

固定 `eval_seed` 与样本 label 生成散光噪声，不因 batch 大小或样本排列改变。默认 `--eval_samples 1`；多样本时图像和二维向量取样本均值，SIA 对每个样本先求范数再取均值。`SIA_from_mean_vector_D` 另列，避免把二者混淆。多样本离散程度并不等于已经校准的不确定性。

图像指标：MAE/RMSE/PSNR 对全部有效像素累计；SSIM 对每张图都计算，并只平均整个 SSIM 窗口都有效的中心位置。SSIM 无可用窗口时返回 null，并报告计数。PSNR 使用固定归一化 data_range=1，零误差时上限 120 dB。`MAE_raw/RMSE_raw` 为原始图像单位。SIA 指标包括 MAE、RMSE、Bias、向量 L2 误差及 X/Y 分量误差（均为 D）。

```text
results/.../
  config.json
  preflight.json
  cv_summary.json
  fold_01/
    normalization.json
    history.json
    runs/                 # TensorBoard
    best_image.pth
    best_sia.pth
    best.pth
    final.pth
    last.pth
    test_results/
      best.json
      final.json
      best/
        metrics.json
        evaluation.json
        predictions.csv
        000000.png
        ...
```

`--save_predictions` 可额外保存所有原始单位预测图和每个样本的 SIA 采样数组；默认仅保存表格和前几个可视化，避免大量文件。`predictions.csv` 的行号对应 `000000_*` 文件名，并保留 label_index。`cv_summary.json` 保留 `aggregate.best/final` 的 mean/std 结构，包含新 SIA 指标；只跑 3 折时结果仍是 3 折汇总，不是完整 10 折性能。

旧版图像归一化、SSIM 和 checkpoint 选择方式已改变，因此不能将旧实验数值直接用于证明新分支增益。比较图像单任务基线时用相同命令改为 `--model fm`、换一个输出目录，其余条件、EMA、数据划分与训练设置保持一致。

## 恢复训练

同一命令增加 `--resume`。每折从自己的 `last.pth` 恢复。默认每 10 个 epoch、每次验证后和最终 epoch 保存；突然中断最多丢失距离最近保存点的训练，`--checkpoint_every` 可调整。

为避免悄悄改变余弦学习率计划，恢复时 epochs、lr、模型、损失、采样和数据内容必须与原运行一致（设备、worker 数等运行设置可改变）。如果要把原本 200 epochs 的计划延长为另一训练计划，应单独实现明确的 finetune 流程，不能把它伪装成相同实验的恢复。

## 独立重评估

```bash
python cddpm/evaluate.py --checkpoint cddpm/results/FRONT/fm_sia_fusion/CA/fold_01/best.pth --data_root data/dataset_3_SIA --excel_path data/dataset_3_SIA/paired_dataset.xlsx --partition test --output_dir cddpm/results/FRONT/fm_sia_fusion/CA/fold_01/reevaluation --save_predictions
```

加载 checkpoint 原有划分和归一化参数，不重新拟合。表格内容必须与训练时相同。仅加载可信 checkpoint，模型文件带有 Python 序列化的训练状态。

## 只给术前图做新样本预测

```bash
python cddpm/predict.py --checkpoint cddpm/results/FRONT/fm_sia_fusion/CA/fold_01/best.pth --pre data/dataset_3_SIA/raw/FRONT/1_pre.npy --clinical 65 1.2 44.5 530 --sample_id 1 --output_dir prediction_case_1
```

`--clinical` 的顺序必须与训练命令相同；这里的四个数仅是演示值，需替换为该病例真实术前数据。该命令不读取 Excel、真实术后图或 SIA 标签。输出 `post_prediction_raw.npy`、`prediction.npz` 和 `prediction.json`。`--samples` 与 `--seed` 可指定采样次数和种子。

## 验证范围

```bash
python -m pytest tests -q
```

随包测试覆盖精确路径/标签、最终更新次数、联合 ODE 更新、辅助损失梯度、零幅值梯度、随机初始化的 batch 不变性、所有图像的 masked metrics、D4 标签变换、仅训练集归一化、真实 UNet 前后向及输入泄漏检查。

此外已在 CPU 上进行合成 FRONT 数据的一折训练与 EMA 保存、独立重评估、无术后输入的 checkpoint 推理、128×128 采样，以及原 FM/CDDPM-DDIM/BBDM 的前向与采样兼容检查。FNO/UNO 入口保留，未安装其可选依赖进行运行验证。GPU、Windows 多 worker 和真实临床数据性能尚未验证。

另外，已模拟“两轮计划在第一轮保存后中断、恢复到第二轮”，并与不中断训练对照：在 CPU、D4 增强开启的相同设置下，最终模型参数逐项完全一致。该测试不保证跨设备、跨依赖版本逐位复现。
