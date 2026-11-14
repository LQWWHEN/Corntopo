# 018 — 预测SIA驱动的二阶方向场条件

状态：144项单元与回归检查通过，待真实GPU冒烟。
与014共享`preoperative_velocity_sia_coupling`跨任务预测耦合预算，第2/5轮；本轮为SIA到图像的反向条件变体，主要方向为SIA分支与联合学习。
完整任务保持术后图像、SIA幅值和180°周期轴位。

## 直接来源与任务构思

直接父模型为012，源码`a8bd0232cf64b6f1f12cdf7f8e05e7e4f0d221f8`。
从头训练，不加载父模型权重；本轮设计在017正式结果之前冻结。
累计保留005的术前方向矩直接SIA路径、009条件FNO、012非负幅值与双角度
相位读出、径向L1与周期cosine目标及训练折初始化。原四项临床输入不变。

012的两项输出主要通过术前编码器共享信息；本轮检验更明确的条件关系：
先由术前信息预测SIA，再将其幅值/轴位编码为图像中的方向场，组织FNO速度。
014曾从图像端点速度向SIA传递描述符，本轮采用相反方向，不继承其端点
监督或额外FNO前向。没有眼别镜像、排序对比、GradNorm、EMA或数据增强。

## 文献调研

- Xu等，PAD-Net，CVPR2018：[原文摘要](https://openaccess.thecvf.com/content_cvpr_2018/html/Xu_PAD-Net_Multi-Tasks_Guided_CVPR_2018_paper.html)。借鉴将预测任务信息传递给后续任务的设计。
- Keller与van Saarloos，1997：[Fourier transformation of corneal topography data](https://pubmed.ncbi.nlm.nih.gov/9267626/)。角膜功率的二周期分量与规则散光有关，为二阶方向表示提供任务依据。
- Thibos与Horner，2001：[Power vector analysis](https://pubmed.ncbi.nlm.nih.gov/11165859/)。双分量散光向量可表达手术前后的向量变化，并与光学多项式表示联系。
- Worrall等，Harmonic Networks，CVPR2017：[原文](https://openaccess.thecvf.com/content_cvpr_2017/html/Worrall_Harmonic_Networks_Deep_CVPR_2017_paper.html)。借鉴保留角向信息的表示思路。

## 设计与实现

012先输出`q = softplus(u) * (cos(phi), sin(phi))`，`phi`为双角度相位。
在原始图像像素中心建立单位圆坐标，x向右，数组行增加方向为y正向，与
`origin=lower`显示约定一致。构造单通道条件：

`H(q,x,y) = pre_mask * circle * [qx*(x²-y²) + qy*2xy] / 1D`。

这一平滑多项式将幅值和周期方向共同转为空间编码，零幅值对应零场。
将`gain * H`加到FNO已有64通道条件，其中64个gain全零初始化；不改FNO
输入宽度，也不额外消耗初始化随机数，父模型参数与初始输出保持一致。
只新增64个实数参数。图像目标可以通过非零gain反向传播至SIA头。

训练和推理都只使用**预测q**。GT向量只用于SIA监督，术后图只用于FM插值
和图像目标；术后mask不能参与条件生成。原始图像有效域MSE、幅值L1和
周期cosine目标及权重保持012设置，没有辅助目标权重需要搜索。
采样时只计算一次q、方向场和准备好的条件；100步Euler仍调用100次FNO。
入口开关为`--sia_spatial_conditioning`，旧配置默认关闭。

## 审计与冻结实验

144项检查覆盖手算方向场、幅值缩放/零向量、90°旋转/反射关系、原始坐标
45°方向符号、术前有效域、无GT/术后/时间/噪声依赖、图像梯度到SIA头、
零初始化时父模型随机状态/参数/损失/预测一致、推理缓存、回载和旧配置兼容。
GPU冒烟验收之前不启动正式实验；冒烟后清除临时权重。

冻结原10fold前三fold，每折200epoch，batch8、lr1e-4、原AdamW/Cosine，
seed42/eval2026、严格确定性、100步Euler；无EMA，全部授权样本，图像MAE
选优。配置见`configs/iteration018.json`。以下六项验证门槛均通过才测试：

1. 图像MAE低于012，PSNR、SSIM均不低于012。
2. 幅值MAE与GT>=0.1D稳定组轴位MAE各不超过012的1.01倍。
3. 向量RMSE不超过012的1.05倍。

同时报告010/FNO、术前图保持对照以及仅训练折拟合的SIA常量，不切换门槛。
临时保存训练checkpoint，选择提交后直接回载核对各折验证输出再测试，
审计后清除全部权重；拒绝方案保存验证结论后清理。报告包含PSNR、SSIM、
RMSE、MAE，SIA幅值/向量/周期轴位及覆盖率，训练和评估耗时。

## 结果分析

尚无正式实验结果。

## 备注

方向场是可学习条件，不是物理校准后的角膜功率变化，也不强制生成图的
某个谐波系数等于临床SIA。r²包络用于连续、无中心奇点的方向编码。
不声称复现PAD-Net/H-Nets、整个网络旋转等变、通用算法首创或完整自回归
联合似然。历史005特征矩保持原样，新的方向场采用原始图显示坐标。
具体数值、逐例图像及数据来源只存私有档案；同方案变体共享最多5轮预算。
