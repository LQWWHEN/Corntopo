# CPTP / mCPTP 与 FM_SIA 公平对比版本（v2：标量主实验 + 向量消融）

本版本以项目当前 `/Corntopo/train.py`、`dataloader.py`、`metrics.py`、`patient_cv.py`、`transforms.py`、`fm_sia.py`、`ema.py` 和 `visualization.py` 为参考，重构上传的 cGAN 训练与测试入口。不是仅将一次随机划分改成循环训练。

请完整解压并使用整个 `cgan_fair` 目录，或将目录内容整体放入项目的 `cGAN` 子目录。不要只替换两个模型文件后继续使用旧的 `models/__init__.py`、options 或 util 调用方式。新版本不依赖旧 util/Visdom/DDP，支持单进程 CPU / 单 GPU；旧权重不能直接当作新框架 checkpoint 使用。

## 本次修改：直接标量回归作为主实验

`--model mCPTP` 现在默认解析为 `--sia_mode scalar`；`--sia_mode vector` 保留为显式消融；CPTP 默认仍为 off。三个模式共用相同 U-Net、FiLM、多尺度池化/attention 主体及数据协议，SIA 头最后一层输出维数分别为 1、2 或不存在。

| 实验 | 参数 | 输出与监督 |
|---|---|---|
| 直接 SIA 标量回归（主实验） | --sia_mode scalar | 一个非负物理幅值；默认 L1 监督 |
| ΔX/ΔY 向量回归（消融） | --sia_mode vector | 两个标准化分量；向量 MSE + 同一幅值损失 |
| 无 SIA 辅助任务 | --sia_mode off | 仅预测图像，无 SIA 指标 |

标量默认 `SIA_pred = sia_ref * Softplus(head(features))`。没有向量预测、没有向量损失、没有 SIA z-score 标准化。`--scalar_activation identity` 可恢复不施加非负约束的线性回归输出；sia_ref=1 时和原代码的数值单位相同。

两个模式共享 `--sia_loss l1|huber`，本版默认 l1；`--lambda_sia` 仍默认 0.1，`--sia_ref` 默认 1 D。若要恢复原代码回归项实际权重 1，应显式使用 `--lambda_sia 1`。vector 模式仍额外使用 `--lambda_vector`（默认 1），该参数在 scalar 模式不参与损失。对比时保持幅值损失、lambda_sia、sia_ref 一致。要复现上一版向量分支的损失，请显式设置 `--sia_mode vector --sia_loss huber --lambda_sia 0.1`。

两种模式只在 SIA 参数化和对应监督上变化；所有训练模块均正确注册、优化、保存和进入 EMA。scalar 只报告 SIA_MAE、SIA_RMSE、SIA_Bias，不伪造 Vector_MAE、ΔX/ΔY 或向量样本；标量 NPZ 只有 s，向量 NPZ 有 s/q。best_sia checkpoint 同时支持两种分支。

本版 pipeline_version 已更新为 cgan_fm_sia_comparison_v2，不能使用 --resume 将上一版向量 checkpoint 接到标量模式，也不能跨模式续训；应给两个实验不同 save_dir。

## 原代码的问题与修改

| 原实现 | 影响 | 本版处理 |
|---|---|---|
| train_test_split 按样本做一次 8/1/1 划分，并保存位置索引 | 双眼可能跨集合；与 FM_SIA 固定划分不同 | 只读既有 10 折 JSON，默认依次运行第 1、2、3 折；用 label_index 对应样本 |
| 术前、术后各自做单图 min/max | 目标尺度不同，无法公平评价数值变化 | 复用 FM 训练折 global 归一化；local 模式仅取该样本术前范围并共享给术后/预测 |
| mask = normalized_image > 0 | 真实最小值被误判为缺失 | 从 raw 的 finite 状态生成 mask；仅显式 --invalid_zero 才把原始 0 当缺失 |
| clinical 直接读取未标准化 | 尺度不一致 | 仅训练折拟合 mean/std，保存并复用于验证/测试 |
| mCPTP 在 forward 中创建 proj_layers；optimizer_G 只有 netG.parameters | 回归头、attention 和投影未被 optimizer 更新；梯度仍可通过随机头流到 G | 所有 SIA 模块在初始化时注册，纳入 G 优化器、EMA 和 state_dict |
| model_names 只有 G/D | SIA 模块不保存、不随 eval 切换 | BaseModel 改为 nn.Module，整个模型统一保存、切换模式 |
| 定义 lambda_reg 但没有乘到回归损失 | 参数不生效 | 改为生效的 lambda_sia；向量模式另有 lambda_vector |
| CPTP 把 netG 返回的 (image, feats) 当作单个 Tensor | forward 失败 | 统一解包；mCPTP 继续使用多尺度特征与 attention 回归 |
| 最深 1×1 特征仍做 BatchNorm/InstanceNorm | batch=1 等场景报错 | 最内层 downnorm 改为 Identity，验证了单样本训练 |
| 训练循环验证后未明确恢复 train 模式 | BN/Dropout 行为可能错误 | 每个训练 epoch 显式 model.train()，评估显式 eval() |
| 默认 Tanh，而目标为 [0,1]；global/local 也可能产生范围外目标 | 输出范围与任务不合 | 默认 identity，无推理裁剪；sigmoid/tanh 仅保留显式实验选项 |
| metrics 只算 full，按逐图/批次平均；多通道 SSIM 不完整 | 与 FM 的 pooled MSE 口径不一致 | metrics.py 直接复用 FM 当前版本；单图指标另存 CSV |
| test.py 缺 opt、把 *_all 列表写入只有标量键的字典；可视化文件反复覆盖 | 测试报错或丢图 | 新独立测试入口从 checkpoint 读取配置和统计量；逐样本唯一文件名 |

原 mCPTP 的 weighted L1（背景 1、交集 2）和梯度除以全图像素，不等于仅在交集监督。本版默认 `--reconstruction_domain valid`，L1 在交集上按每张图有效像素归一化，gradient 仅使用两个端点都有效的相邻像素；判别器 real/fake 的候选术后图也乘同一训练 mask。**这是明确的训练损失修改，不是原论文模型的逐字复现。** `full` 可用于全图损失消融，但也不是恢复原 mCPTP 的 1/2 权重。

## 固定 CV 与实验选择

`--split_file` 必填；文件不存在就报错，不生成、不改写、不回退随机划分。JSON 必须是 FM 的 version/seed/n_splits/dataset_fingerprint/folds 格式，folds 中 train/val/test 为 **label_index，不是 Excel 行号**。

验证包括：10 个折完整、折号顺序、样本覆盖、分区互斥、每个样本全 10 折恰好进入 test 一次、患者跨 train/val/test 隔离、患者身份指纹。患者身份取术前/术后 ID 以及姓名+生日并集关系，沿用 FM 实现。

随包附上项目已有 `paired_dataset_10fold_splits.json`。该文件包含 275 个样本，前三折为：

| Fold | train | val | test |
|---|---:|---:|---:|
| 1 | 220 | 27 | 28 |
| 2 | 219 | 28 | 28 |
| 3 | 219 | 28 | 28 |

必须使用 FM 实际实验使用的文件；附带文件若与当前表不匹配，程序会拒绝运行。`--k_folds 3` 是运行上述 10 折中的前三折，不是重新做三折交叉验证。样本比例允许因患者双眼分组轻微偏离 8/1/1。

与 FM 一样：每折 seed+fold；训练到指定 epochs；第一轮、eval_every 和最后一轮做验证；分别保存 best_image / best_sia。`--best_metric MAE` 默认取 valid MAE 最优 checkpoint 作为 `best.pth`；SIA_MAE 可以显式指定。训练结束只评估预先指定的 best 和 final，不用 test 选择模型。不同方法须预先固定同一 best_metric。

G/D 均用 AdamW + cosine，默认 lr=1e-4、weight_decay=1e-4、betas=(0.9,0.999)，梯度裁剪 1.0；G/D 各更新一次。保留 GAN 所需的双优化器。`--lr_D` 可另设，实验配置会记录。EMA 是可选项，必须与 FM 实验保持一致。断点包含 G、D、SIA、两个优化器/调度器、EMA、随机状态及 DataLoader generator；--resume 不允许更改会影响实验的配置。默认不开 D4；开启时同时变换图、mask 和双角度 SIA 向量。方向性 clinical 未实现同步变换，因此 D4 会拒绝这类列名。

## condition、clinical 与 SIA

- **图像 condition**：术前 FRONT（默认），一直输入 U-Net；`--fusion_method none` 仅关闭 clinical，不会关闭术前图像条件。输入和输出必须是同一个单通道 map，image_size 与 unet_128/unet_256 对应。
- **clinical**：`--conditions` 指定的有序列，训练折标准化后送入 ConditionEncoder，并在 U-Net 多尺度下采样特征中做 FiLM。`--fusion_method film` 才启用；本版保留 cGAN 的 FiLM，不伪装成 FM 的 cross-attention/concat 等融合。判别器沿用图像对输入 `[pre, candidate_post]`，没有额外接 clinical；G 的生成和 SIA 头可以通过 FiLM 特征使用 clinical。
- **禁止泄漏**：拒绝 label_index、post_*、delta_*、名称含 sia 的 clinical 条件及 SIA 目标列。仍需人工确认自定义别名是否为术后结果，以及术中参数是否符合所声明的预测时间点。
- **SIA 标签**：为对齐 FM，若存在完整 `[delta_front_X, delta_front_Y]`，两种模式都使用其模长作为同一个幅值标签，并校验已有 `front_SIA_D`。scalar 仅监督这个标量，不把向量送入模型；vector 才额外监督两个分量。若只有 `front_SIA_D`，scalar 也可独立训练；vector 明确报错。只有标量的数据不能用于与 vector 的同样本消融，正式比较应使用完整配对表。
- **标签检查**：缺失、非有限值、负幅值会报错；向量模式检查训练折 mean/std 和 X/Y 定义。不要把半尺度 J0/J45 差分直接当成 X/Y。`--skip_sia_checks` 仅跳过质量标记/幅值一致性检查，不能跳过数值有效性。D4 变换保持标量幅值不变，向量仍使用双角度变换。

设幅值残差为 `r = (SIA_pred - SIA_true) / sia_ref`，默认 `L_SIA = mean(abs(r))`；选择 huber 时为 `Huber(r, 0, delta=huber_delta)`。

标量模式：

`L_G = lambda_gan * L_GAN + lambda_L1 * L1 + lambda_grad * L_gradient + lambda_sia * L_SIA`

向量模式：

`q_pred = mean_train + std_train * z_pred`

`SIA_pred = sqrt(q_pred_X**2 + q_pred_Y**2)`

`L_vector = MSE(z_pred, (q_true - mean_train) / std_train)`

`L_G = lambda_gan * L_GAN + lambda_L1 * L1 + lambda_grad * L_gradient + lambda_vector * L_vector + lambda_sia * L_SIA`

这里保留 cGAN 的直接回归性质，不使用 FM 的速度回归。向量消融同时改变了输出参数化与辅助监督，不能把其差异全部归因于方向信息；标量/向量都共享特征也不自动保证图像预测与 SIA 一致。

推理 API 为 `model.sample(pre, clinical)`，不接收 post、mask_post 或真实 SIA。输出为确定性结果；SIA_sample_std=0 仅为兼容结果列，不代表校准后的不确定性；不会伪造多次随机采样。

| 设置 | 默认 SIA | 使用含义 |
|---|---|---|
| --model CPTP | off | 纯图像 cGAN 基线，SIA 指标不输出，不能记为 0 |
| --model mCPTP | scalar | 图像 + 直接幅值回归主实验 |
| --model mCPTP --sia_mode vector | vector | 图像 + 向量回归消融 |
| --model CPTP --sia_mode scalar | scalar | 显式加入相同标量头 |
| --model CPTP --sia_mode vector | vector | 显式加入同一回归头的 CPTP+SIA |
| --model mCPTP --sia_mode off | off | 去掉辅助任务的消融 |

clinical 均需显式启用。**两个名称在相同 sia_mode、fusion 和其余参数下使用同一套实现，没有额外隐藏的结构差别；不要把相同配置当作两个独立方法。** 若要比较同等临床输入和辅助监督下的模型能力，使用 mCPTP + SIA + 同一 clinical 列集对比 FM_SIA；纯 CPTP 作为缺少辅助任务的基线单列报告。

## 指标口径

`metrics.py` 与读取到的 FM 参考代码逐字节相同；新增 `sia_metrics.py` 适配标量分支，其图像和向量模式结果与原 MetricAccumulator 一致。所有图像指标先将模型输出反归一化为物理值，再重编码到该训练折 global 范围；不按测试图的 GT min/max 重新拉伸、不裁剪预测。

| 指标 | 定义 |
|---|---|
| MAE_full / RMSE_full / PSNR_full / SSIM_full | 全图，含 GT 归一化后补 0 的无效区域 |
| MAE_valid / RMSE_valid / PSNR_valid / SSIM_valid | mask_pre ∩ mask_post；不是先乘 mask 再对整图平均 |
| 无后缀 MAE / RMSE / PSNR / SSIM | 保留 FM 的 valid 别名，best_metric=MAE 据此选择 |
| MAE_raw_full / RMSE_raw_full / MAE_raw_valid / RMSE_raw_valid | 对应像素域，转换为原始 map 单位（FRONT 为 D 时即 D） |
| SIA_MAE / SIA_RMSE / SIA_Bias | 物理 SIA 幅值误差，单位 D |
| Vector_MAE / Vector_RMSE | 二维向量误差的 L2 范数均值 / 均方根，单位 D |
| DeltaX_MAE / DeltaY_MAE | 分量物理误差 |

PSNR 用整个评估集合该像素域 pooled MSE，data_range=1，MSE 下限 1e-12；不是平均每张图 PSNR。SSIM 用最大 7×7 的奇数窗口；valid SSIM 只取完全落在有效区的窗口中心；无合格窗口则为 null。折间 aggregate 是三折指标的算术 mean / population std（ddof=0），不是把三折测试像素再合并计算，也不代表完整 10 折结果。

full 的 raw 指标仍包含无效 GT 补 0 对应的参考基准，不能解释为“真实角膜区临床误差”；临床解释以 valid/raw_valid 为主，full 用于描述整个输出域。逐样本 CSV 的 PSNR 不应自行平均替代 metrics.json 的 pooled PSNR。

## 运行示例（Windows）

进入解压后的目录。安装依赖时保留你已有的、适合本机 CUDA 的 PyTorch；无需更换 FM 所在环境。缺依赖可用 `python -m pip install -r requirements.txt`。

下面路径是示例，需替换为 FM 实际使用的同一表格、raw 目录、split 路径。data_root 下应有 `raw/FRONT/<label>_pre.npy` 和 `<label>_post.npy`。本版只读取物理 raw .npy，不使用 PNG 热图作为训练输入。

先预检查（不训练）：

```powershell
python train.py --model mCPTP --excel_path "D:/VS Workspace/CornTopo/data/paired_dataset.xlsx" --data_root "D:/VS Workspace/CornTopo/data/img512" --split_file "D:/VS Workspace/CornTopo/data/paired_dataset_10fold_splits.json" --save_dir "results/mcptp_sia" --k_folds 3 --preflight_only
```

图像 + 直接 SIA 标量回归主实验（未启用 clinical）：

```powershell
python train.py --model mCPTP --excel_path "D:/VS Workspace/CornTopo/data/paired_dataset.xlsx" --data_root "D:/VS Workspace/CornTopo/data/img512" --split_file "D:/VS Workspace/CornTopo/data/paired_dataset_10fold_splits.json" --save_dir "results/mcptp_sia_scalar" --sia_mode scalar --sia_loss l1 --lambda_sia 1 --k_folds 3 --epochs 200 --batch_size 8 --image_normalization global --best_metric MAE --save_predictions
```

向量消融：将上面命令的 `--sia_mode scalar` 改为 `--sia_mode vector --lambda_vector 1`，并把输出目录改为 `results/mcptp_sia_vector`。其余所有设置保持相同，包括 `--sia_loss l1 --lambda_sia 1`。分别从 `cv_summary.json` 的 aggregate.best 对比共同图像/SIA 指标；Vector_* 只对向量模式有效。

需要与含 clinical 的 FM_SIA 对比时，在上面命令添加：

```text
--conditions <与FM完全一致且顺序相同的列名列表> --fusion_method film --clinical_embed_dim 128
```

尖括号是说明占位符，不能原样执行。比如表中确有 `pre_age` 且 FM 也使用它，才可写 `--conditions pre_age`。不要根据示例自行改变临床变量集合。若 FM 使用 EMA、local、D4、自定义 epochs/batch/lr/eval_every/lambda_vector/lambda_sia，须显式给出对应设置；本包没有拿到你最近一次真实训练的 config.json，默认值不能证明与那次运行完全相同。

纯图像基线：将 `--model mCPTP` 改为 `--model CPTP`，使用新的 `--save_dir`，其余数据协议一致。断点续训：原完整训练命令加 `--resume`；不能通过增加 epochs 改变既定 cosine 计划。独立重评估：

```powershell
python test.py --save_dir "results/mcptp_sia_scalar" --device cuda --checkpoint both
```

默认从保存的配置重建同一前三折，加载每折统计量；不重新拟合、不截断 test 数量。目录迁移后可显式覆盖 --excel_path / --data_root / --split_file，表格和 split 字节哈希必须与训练一致。raw 文件内容应自行保持冻结，当前与 FM 一样没有为所有 .npy 建立内容哈希。

## 结果与检查点

`save_dir/cv_summary.json` 包含 `aggregate.best.<metric>.mean/std/n`、`aggregate.final`、fold_results 和完整 config；兼容按 best aggregate 汇总的脚本。

每折目录包含 normalization.json、history.json、runs/、last/final/best/best_image.pth，以及启用 SIA 时的 best_sia.pth。`visual/` 每轮固定 1 个验证样本；`val_results/` 每次验证前 5 个；`test_results/best/` 和 `final/` 保存全部测试样本，各有 pre/gt/pred/error 四张 PNG、物理预测（--save_predictions）、predictions.csv、metrics.json、evaluation.json 和 checkpoint.json。独立 test.py 写入 reevaluation/。

绘图直接复用 FM visualization.py：固定 global 或术前 local 色域、误差上限一致，显示裁剪不影响指标。每个样本唯一编号，不会反复覆盖同名图片。

## 已完成验证与限制

本次共 18 项测试通过，覆盖原有图像指标、固定划分、归一化合同，以及：

- CPTP off/scalar/vector、mCPTP scalar/vector 单样本训练和保存/加载，SIA 投影、attention、回归头参数确实更新。
- scalar 非负输出、独立于向量损失权重的 L1/Huber 损失公式、默认模式、跨模式权重加载拒绝。
- scalar 支持仅幅值标签；D4 后幅值不变；vector 缺少 X/Y 时拒绝。
- 同一预测幅值下，scalar 与原 FM 向量评估的幅值指标完全相同；全部图像指标不变，scalar 不产生向量指标。
- scalar/vector 分别完整执行合成数据前三折、EMA、物理结果保存、独立重评估、SIA 最优模型保存。
- scalar/vector 分别模拟中断，在 D4、Dropout、EMA 下恢复后的权重与连续训练逐元素一致（CPU、num_workers=0）。

没有真实原始角膜数组，未执行真实数据 3 折训练，也未验证 CUDA、多进程 DataLoader 或模型效果。所有真实比较仍需保持同一 split、表格、clinical、训练轮数、选模规则及调参预算。

可在本目录重跑：`python -m pytest tests -q`。
