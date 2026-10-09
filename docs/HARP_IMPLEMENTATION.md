# HARP 实现说明：前向、fix2 反向与训练调度

本文解释本仓库 **HARP 路由实际执行的计算**，服务于代码阅读、复现和技术写作，不替代实验结果分析。公式、参数和伪代码对应冻结源码；数值收益必须另查真实结果，不能由结构或初始化直接推定。

HARP 与 LARP 是两条独立路线。HARP 在 PromptKD 学生视觉编码器中增加瓶颈残差，保留提示学习、特征投影器和教师蒸馏结构；这里没有同时打开 LARP，也没有打开表示约束分支。

## 1. 阅读顺序与源码入口

| 要核对的内容 | 实际入口 |
| --- | --- |
| 残差接入位置、提示 token 替换、视觉序列变换 | [`clip/model.py`](../routes/harp/vendor/project/clip/model.py)：`ResidualAttentionBlock_IVLP.forward`、`VisionTransformer.forward` |
| 瓶颈模块、初始化、两个自定义 VJP | [`efficient_adaptation.py`](../routes/harp/vendor/project/trainers/efficient_adaptation.py)：`HighLevelAdapter`、`_InputScaledLinearFunction`、`_ScaledLinearFunction` |
| 学生/教师构建、训练范围、投影器、KD 损失 | [`promptkd.py`](../routes/harp/vendor/project/trainers/promptkd.py)：`CustomCLIP`、`CustomCLIP_teacher`、`PromptKD` |
| 公共组与新增组划分 | [`adaptation_optim.py`](../routes/harp/vendor/project/trainers/adaptation_optim.py)：`adaptation_param_groups` |
| 第 1 轮新增组保持、第 2 轮新增组渐升 | [`policy.py`](../routes/harp/vendor/project/policy.py)：`effective_lr`、`before_update` |
| 第 2 轮公共投影器渐升、临时参数分组 | [`projector_schedule.py`](../routes/harp/vendor/project/projector_schedule.py)：`effective_lr`、`split_step` |
| 组合两套策略的真实 SGD 包装 | [`frozen_install.py`](../routes/harp/runtime/frozen_install.py)：`install`；该函数由冻结源逐字保留 |
| 常数 warmup 与后续 cosine | [`lr_scheduler.py`](../routes/harp/vendor/project/Dassl.pytorch/dassl/optim/lr_scheduler.py) |
| 路径参数化、配置/初态核对及安装原调度 | [`portable.py`](../routes/harp/portable.py) |
| 全部实际配置，而非不完整命令行默认值 | [`configs/index.json`](../routes/harp/configs/index.json)、[`configs/portable/`](../routes/harp/configs/portable/) |

如何准备物料和启动完整流程见 [运行说明](RUN_REPRODUCTION.md) 与 [实验协议](EXPERIMENT_PROTOCOL.md)。不要单独执行 `vendor/project/train.py` 来代替本仓库入口：裸入口没有自动包含这里的冻结双渐升包装与分阶段核验流程。

## 2. 主方案及结构消融

代码中的视觉层编号从 **0** 开始，论文自然语言中的“第几层”通常从 **1** 开始，两种编号必须区分。

| 路由变体 | `ADAPTATION.TYPE` | 代码层集合 | 自然语言层号 | 瓶颈宽度 d | 新增参数 |
| --- | --- | --- | --- | --- | --- |
| `r1` | `none` | 无模块；配置中的层列表不生效 | 无 | 不生效 | 0 |
| `harp`（主方案） | `harp` | `[8,9,10,11]` | 第 9–12 层 | 32 | 199,808 |
| `mid4_d32` | `harp` | `[4,5,6,7]` | 第 5–8 层 | 32 | 199,808 |
| `last4_d16` | `harp` | `[8,9,10,11]` | 第 9–12 层 | 16 | 101,440 |
| `last4_d64` | `harp` | `[8,9,10,11]` | 第 9–12 层 | 64 | 396,544 |

共同约束：学生 `ViT-B/16`、教师 `ViT-L/14`、输入 `224×224`、视觉提示深度 9、每个提示位置 4 个 token、`HARP_SCALE=0.001`、新增组 `LR_MULT=0.1`、`REP_WEIGHT=0`、训练 20 轮、seed 为 1/2/3。`r1` 没有新增组，其 `LR_MULT=1`。

每层参数数目为

\[
P_{\text{layer}}=Dd+d+Dd+D=2Dd+d+D=1537d+768,\qquad D=768.
\]

四层总数为 `4 × (1537d + 768)`。统计包含 down/up 的权重与偏置，不包含冻结主干，也不把原提示和投影器算作“新增 HARP 参数”。这三个消融只对应表中已登记的结构变化，不是自动超参数搜索接口。

## 3. 从图像到分类分数：实际形状

设本批图像数为 `N`，类别数为 `C`。训练 batch 为 8，测试 batch 为 100，测试末批可以小于 100。

| 阶段 | 学生端形状 | 说明 |
| --- | --- | --- |
| 图像输入 | `[N,3,224,224]` | 输入转换与增强由固定数据配置决定 |
| patch 卷积 | `[N,768,14,14]` | `kernel_size=stride=16` |
| 展平 patch | `[N,196,768]` | 14×14 个 patch，不是语义分割 |
| 加 CLS 与位置编码 | `[N,197,768]` | 1 个 CLS＋196 个图像 token |
| 追加 4 个视觉提示 | `[N,201,768]` | 提示追加在序列尾部 |
| `ln_pre` 后转置 | `[201,N,768]` | Transformer 内部采用序列优先 `LND` |
| HARP 的 down/ReLU/up | `[201,N,768] → [201,N,d] → [201,N,768]` | 模块逐 token 作用，**不是只处理 CLS** |
| 最终 CLS 与 CLIP 视觉投影 | `[N,768] → [N,512]` | 取序列第 0 个 token；沿用原 `ln_post` 和 `proj` |
| `VPT_image_trans` | `[N,512] → [N,768]` | 对齐教师文本向量的维度 |
| L2 归一化与文本点积 | `[N,768] × [768,C] → [N,C]` | 使用冻结教师类别文本向量 |

学生浅层提示在 `VisionTransformer` 入口追加；代码层 1–8 在本层计算前用各自的 `VPT_shallow` **替换**序列尾部的 4 个提示，不累计追加。代码层 9–11 没有新的深层提示参数，但输入序列仍保留之前的 4 个提示输出。因此当前固定配置下，HARP 所见序列长度为 201。

主方案代码层 8 仍先执行该层的提示替换，再计算 HARP；代码层 9–11 没有这次替换。中四层变体也遵循完全相同的原提示逻辑。

`VPT_image_trans` 的真实结构为：`1×1 Conv(512→768) → BatchNorm2d(768) → ReLU → 1×1 Conv(768→768)`。代码把 `[N,512]` 临时扩为 `[N,512,1,1]`，最后再去掉两个空间维；它不是额外的 token 注意力模块。

## 4. HARP 放在块中的什么位置

令 `X` 表示已经完成本层提示替换的输入，`A_l` 表示本层 HARP。所选视觉块按下式计算：

\[
U=\operatorname{LN}_1(X),\qquad
R=A_l(U),
\]
\[
Z=X+\operatorname{MHA}(U,U,U),\qquad
Y=Z+\operatorname{MLP}(\operatorname{LN}_2(Z))+R.
\]

未选中的块取 `R=0`。原 MLP 为 `Linear(D,4D) → QuickGELU → Linear(4D,D)`；HARP 内部用的是 **ReLU**，二者不能混写。

这里必须同时保留两个位置事实：

- HARP **读取注意力前**、经 `LN1` 归一化的输入，与注意力共享同一 `U`。
- 它的残差在原注意力更新及 MLP 计算后相加，**不进入本层 MLP 的输入**。

因此它不是“把 adapter 串接在 MLP 后面”，也不是“读取 `LN2(Z)` 的并行 MLP”。固定注意力权重仍参与前向与梯度传播，只是不由优化器更新。

## 5. `HighLevelAdapter`：前向及初始化

将序列和批次维展平为 `M=L×N` 行，仅为书写公式方便。设

\[
U\in\mathbb R^{M\times D},\quad
W_d\in\mathbb R^{d\times D},\ b_d\in\mathbb R^d,\quad
W_u\in\mathbb R^{D\times d},\ b_u\in\mathbb R^D,\quad s=10^{-3}.
\]

实际前向为

\[
Q=UW_d^\top+b_d,\qquad H=\operatorname{ReLU}(Q),
\]
\[
R=\operatorname{linear}(H,sW_u,sb_u).
\]

在实数算术下最后一式等于 `s × (H W_uᵀ + b_u)`；**FP16 中代码先缩放权重/偏置再执行 linear，不是先执行未缩放 GEMM 再缩放输出**，两者的舍入及中间值范围可能不同。

初始化严格使用：

- `down.weight`：`kaiming_normal_(mode="fan_out", nonlinearity="relu")`，不是默认 `fan_in`。
- `down.bias`、`up.weight`、`up.bias`：全部为 0。
- 模块转为原注意力输入投影权重的 dtype；当前 HARP 新增权重为 FP16。
- `s` 为固定数值，不是可学习门控参数。

因此新增残差在初始化时为 0，结构上保持原学生的前向函数。这个事实不等于已经证明任意 GPU 上两次独立前向逐位相同；实际运行仍须通过既定工程门槛。

`CustomCLIP` 在 `torch.random.fork_rng(devices=[])` 内完成新增模块初始化，以免新增 CPU 随机抽样改变后续原投影器初始化和数据 RNG 流。中四层宽 32 的自然初态按层序与后四层宽 32 核对：代码层 4/5/6/7 分别对应 8/9/10/11；不是从已训练模型迁移。宽 16/64 保留各自尺寸下的自然初始化，不截取宽 32 参数。

## 6. fix2 自定义 VJP：不能用普通导数替代

VJP 指向量–雅可比积，即反向传播中给定上游梯度后计算下游梯度的规则。当前两个 linear 都由自定义 `torch.autograd.Function` 实现。**仅按第 5 节前向公式调用普通 autograd，会得到不同的参数梯度，不是当前实验的复现。**

### 6.1 up：前向缩放，反向不带该缩放

对于上游梯度 `G=∂L/∂R`，`_ScaledLinearFunction.backward` 返回：

\[
\bar H=GW_u,\qquad
\bar W_u=G^\top H,\qquad
\bar b_u=\sum_{m=1}^{M}G_m.
\]

这些式子**没有乘 `s`**。普通的 `linear(H,sW_u,sb_u)` 导数会让三者都带 `s`；这里保留的是项目登记的参数侧梯度规则。代码也不会显式制造 `G/s` 这一半精度中间量。

### 6.2 down：只在输入梯度侧应用缩放

令 `K=\bar H ⊙ 1[Q>0]`，ReLU 在 0 处按 PyTorch 规则取 0。`_InputScaledLinearFunction.backward` 返回：

\[
\bar U=K(sW_d),\qquad
\bar W_d=K^\top U,\qquad
\bar b_d=\sum_{m=1}^{M}K_m.
\]

其中 `K(sW_d)` 在矩阵乘之前缩放权重，而不是先计算 `KW_d` 后再把结果乘 `s`。参数侧梯度仍不额外乘 `s`。

### 6.3 应如何解释这组规则

在实数算术下，HARP 分支传回输入 `U` 的梯度保留了一个 `s` 因子；up/down 参数侧梯度则省略了普通缩放前向的 `s` 因子。这里体现的是项目采用的 **MMA-inspired 残差/梯度尺度配套规则**，不能把它描述为普通小残差的标准链式求导，也不能把新增组学习率 0.1 与残差尺度 0.001 合并成一个“有效学习率”。

fix2 将两个危险中间量对应的缩放提前到 GEMM 操作数侧，并保持上述登记 VJP：up 前向用缩放权重，down 反向输入项用缩放权重。这是有实际代码差别的有限精度实现调整，不是宣称 FP16 与实数/FP32 严格等价，也不保证任意输入下不会溢出。原有限值检查仍然必须保留。

文件内还保留 `_GradientScale`、`_ResidualScale` 等基础类；当前 `HighLevelAdapter.forward` 真正调用的是 `InputScaledLinear → ReLU → ScaledResidualLinear`。不能在这条路径外再加一次 reciprocal scaling，否则会重复改变反向。

## 7. 哪些部分训练，哪些部分冻结

| 部分 | 当前状态 | 说明 |
| --- | --- | --- |
| 学生视觉主干 patch/attention/MLP/LN/原视觉投影、`logit_scale` | 权重冻结 | 保留前向和向提示/新增模块传播的梯度路径 |
| 学生视觉提示 `VPT` 与各层 `VPT_shallow` | 训练 | 9 个提示位置×4×768，共 27,648 个参数 |
| `VPT_image_trans` | 训练 | 两个卷积的权重/偏置与 BN 仿射参数，共 986,112 个参数；6 个可训练张量 |
| 所选层 `harp_adapter` | 训练，但第 1 轮不更新 | 数量见第 2 节 |
| 教师视觉/文本编码器、教师提示、教师 `logit_scale` | 全部冻结、`eval()` | 由指定教师 checkpoint 加载；不是和学生共同训练 |
| 教师类别文本向量 | 缓存、detach | 首次构建后复用；不创建额外学生文本分支 |
| 表示约束参考编码器 | 不创建 | `REP_WEIGHT=0`，所以不执行该辅助分支 |

公共可训练参数为 `27,648 + 986,112 = 1,013,760`；主 HARP 总可训练参数为 `1,213,568`。`BatchNorm` 的 running mean/variance 等是状态缓冲区，不计入参数量，但会随原训练/评估模式更新或使用，保存与重载也必须包含它们。

配置 `PREC=fp16` 表示沿用原 CLIP 半精度路径，**不是把所有参数和缓冲区统一强转为 half**。例如原 CLIP LayerNorm、原提示参数及投影器 BN 保留原构造/转换规则；HARP 的 down/up 参数匹配注意力权重的 FP16。当前路径不是 AMP，没有 GradScaler，没有 FP32 master 副本，没有梯度裁剪，也没有改用 AdamW。

## 8. 损失与分组 SGD

教师和学生接收同一批经过数据变换的图像。学生的 768 维归一化视觉特征与教师缓存的 `C×768` 归一化文本向量点积得到 `Z_s`；教师自己的视觉特征得到 `Z_t`，二者分别使用各自冻结的 `exp(logit_scale)`。

令 `P_t=softmax(Z_t/T)`，`P_s=softmax(Z_s/T)`，当前损失为

\[
\mathcal L=\lambda_{KD}\frac{T^2}{NC}
\sum_{n=1}^{N}\sum_{c=1}^{C}
P_{t,nc}\bigl(\log P_{t,nc}-\log P_{s,nc}\bigr),\qquad T=1.
\]

这是代码中 `kl_div(log_softmax(student/T), softmax(teacher/T), reduction="sum") / stu_logits.numel()` 的方向和归一化方式，不是除以 batch 数的 `batchmean`。`λKD=200` 用于 Aircraft、Flowers、DTD，其余七集为 1000。当前总目标只有这一项 KD；`CE_WEIGHT=0`、`REP_WEIGHT=0`，不存在额外 HARP 分类监督或表示损失。

`forward_backward` 虽然读取标签，但此损失不使用训练标签。当前 `base2novel + NUM_SHOTS=0` 的原数据读器提供全类别的训练图像，不能写成“只用 Base 类训练图像的监督微调”。这不等于把 Base/Novel 测试图像加入训练；具体 membership 由固定划分与身份核验限定。

优化器沿用 SGD：公共初始基准 LR `0.005`、momentum `0.9`、weight decay `0.0005`、dampening `0`、Nesterov `False`。HARP 有公共组和新增组两组，后者名义 LR 为前者的 0.1 倍；R1 仅有公共组。

## 9. hold/ramp 的精确时序

本节用 **零基** `e∈{0,…,19}` 表示 epoch，用 `b∈{0,…,Q−1}` 表示本轮 batch；`Q=len(train_loader_x)>1`。人读日志的 epoch/batch 则分别为 `e+1`、`b+1`。原训练读器以 batch 8 丢弃不完整训练尾批，当前 `Q=floor(N_train/8)`。

### 9.1 公共名义调度

原 scheduler 是一轮 constant warmup，之后 `CosineAnnealingLR(T_max=20)`，`WARMUP_RECOUNT=True`。实数参考式为

\[
\eta(e)=
\begin{cases}
10^{-5},& e=0,\\
\frac{0.005}{2}\left[1+\cos\!\left(\frac{\pi(e-1)}{20}\right)\right],&1\le e\le19.
\end{cases}
\]

实现以原 scheduler 状态与实际 LR 记录为准，不能用重写闭式公式替代浮点状态更新。`forward_backward` 在每轮**最后一个 batch 更新完成后**调用 `update_lr()`；warmup 占一轮，所以不能笼统声称“第 20 轮训练 LR 已衰减到 0”。

### 9.2 三类参数的实际 LR

| 参数类别 | 第 1 轮 `e=0` | 第 2 轮 `e=1` | 第 3–20 轮 `e≥2` |
| --- | --- | --- | --- |
| 原视觉提示 | `1e-5` | `0.005`，本轮不做 batch 渐升 | `η(e)` |
| 原 `VPT_image_trans` 六个张量 | `1e-5` | `1e-5 + (0.005−1e-5) × b/(Q−1)` | `η(e)` |
| 新增 HARP 参数 | 保持；实际 LR 0，且梯度设为 `None` | `1e-6 + (0.0005−1e-6) × b/(Q−1)` | `0.1 × η(e)` |

第二轮两个渐升均包含首尾端点：第一个 batch 分别为 `1e-5`/`1e-6`，最后一个 batch 为 `0.005`/`0.0005`。两个尺度不要混淆：`HARP_SCALE=0.001` 是第 5–6 节的残差/VJP 尺度；`LR_MULT=0.1` 是这里的 SGD 分组倍率。

### 9.3 不是“设 LR=0 就算保持”

第一轮仍执行原前向和反向、检查原始梯度有限；紧邻 SGD 更新之前，`policy.before_update` 把新增参数梯度置为 `None`，并断言它们没有已有优化器状态。这样该组既不更新参数，也不积累 momentum 或应用 weight decay。`frozen_install.install` 还逐步核对新增参数保持不变、没有生成状态。

第二轮为了让公共投影器与提示使用不同 LR，`split_step` **临时**把原组拆开，但仍调用同一个原 SGD 的一次 `step()`，保留参数对象与 optimizer state 身份。调用结束恢复原组列表和名义 LR，不重建优化器、不清空动量、不把同一梯度更新两遍。

公共投影器渐升也用于当前 HARP 路由的 `r1`，否则 HARP/R1 会同时混入公共训练策略差别。新增组 hold/ramp 只用于 HARP 及其结构消融。这里的 R1 是本路由对照，不应直接改称“未修改官方 R0”。

## 10. 与实现对应的科学伪代码

以下是说明性伪代码，**不是另一份可替换训练器**。`HARP_UP_VJP` 与 `HARP_DOWN_VJP` 必须按第 6 节实现，不能省略后当作普通 linear。

```text
给定：冻结原 CLIP/教师权重、固定数据划分、完整已解析配置、seed
配置：选定层 S，瓶颈 d，残差尺度 s=0.001，新增 LR 倍率 m=0.1

按原构造顺序加载学生 ViT-B/16 与教师 ViT-L/14，保留原视觉提示
在隔离的 CPU RNG 上下文内：
    对 l ∈ S：
        down_l ← InputScaledLinear(768, d, s)
        up_l   ← ScaledResidualLinear(d, 768, s)
        down_l.weight ← Kaiming normal(fan_out, ReLU)
        down_l.bias, up_l.weight, up_l.bias ← 0
构建学生原 VPT_image_trans 投影器
构建并加载教师；冻结全部教师参数，eval；缓存归一化类别文本向量
只开放原视觉提示、原投影器及新增 HARP 参数
创建原 SGD 与原 warmup/cosine scheduler
安装原 frozen_install.install（R1 也安装公共投影器渐升）

对 epoch e = 0..19：
    设置学生 train 模式
    对本轮 batch b = 0..Q−1：
        读取同一批增强图像 I
        teacher_logits ← 冻结教师(I)，no_grad
        student_image_features ← 学生视觉编码器(I)：
            原 patch/CLS/位置编码/提示处理
            对每个视觉块 l：
                X ← 原提示替换逻辑
                U ← LN1_l(X)
                R ← 0
                若 l ∈ S：
                    H ← ReLU(HARP_DOWN_FORWARD(U, down_l))
                    R ← HARP_UP_FORWARD(H, up_l, s)
                Z ← X + Attention_l(U,U,U)
                X ← Z + MLP_l(LN2_l(Z)) + R
            取 CLS，经原 ln_post 与视觉投影输出 512 维
        F ← 原 VPT_image_trans(student_image_features)，输出 768 维并归一化
        student_logits ← exp(冻结学生 logit_scale) × F × 教师文本向量转置
        loss ← λKD × T²/(N×C) × KL(teacher || student)，T=1
        原 optim.zero_grad()
        loss.backward()，HARP 必须使用登记自定义 VJP
        检查 loss/原始梯度有限；异常直接停止，不裁剪或自动降精度
        原 SGD step 的冻结包装：
            求当前名义 LR 与两条实际渐升 LR
            若 e=0：新增参数 grad=None，核实无 momentum state
            若 e=1：临时拆分公共投影器组
            执行一次原 SGD step，保留状态
            恢复原组/名义 LR，检查参数有限与 hold 条件
        若为本轮最后一个 batch：调用原 scheduler.step()
    依原 best_val 规则评估并保存选中模型
```

工程流程还包括新进程初态核验、四步门槛、全部训练输入/LR 配对、保存后原进程评估与首次独立重载。这些步骤不通过改变损失或梯度计算代替验证；运行入口和审计文件说明见 [运行说明](RUN_REPRODUCTION.md)。

## 11. 复现与结论边界

- **后端是协议的一部分。** 当前 HARP 路由保留 `cudnn.benchmark=True`、`cudnn.deterministic=False`、确定性算法关闭、Flash/math SDP 开启、memory-efficient SDP 关闭。不能只看 YAML 的 `REPRODUCIBLE=False` 就漏掉运行包装里的 `efficient=False`；也不能把另一条路线的后端规则套过来。
- **FP16 的舍入是实现事实。** 前向等价的实数公式不能证明 GPU 上的数值逐位等价；fix2 的算子顺序与自定义 VJP 都属于复现要求。保持硬失败记录，不用重试取高分证明等价。
- **身份保持初始化不证明训练必定提高。** 零输出初始化、第一轮 hold 与第二轮渐升解释了模型和优化器如何启动；是否改善迁移、精度或稳定性仍由匹配协议的结果决定，不在本文件预设。
- **容量与层位实验支持有限范围判断。** 中四层与后四层宽 32 对比控制了新增参数量；宽 16/32/64 对比同时改变表示容量与参数规模。不能据此宣称已经穷尽所有结构、证明所有高层都优于所有中层，或把特定选择的收益归因于唯一原因。
- **不把通用 Adapter 或其梯度尺度思想说成本项目首创。** 源码标注为 MMA-inspired；本文记录本项目接入 PromptKD 的具体位置、fix2 数值实现和配套训练规则。正式论文的来源归属应依据相应原文与引用登记，不能用本实现说明代替文献证据。
- **模型选择口径需披露。** 当前 `best_val` 按原流程使用 `val_loader` 的 Base 指标，更新条件为严格 `>`，相同最高值保留最早 epoch；当前 base-to-novel 读器中的 `val_loader` 对应 Base 测试划分，不能将其写成额外独立验证集。Novel 不用于选择最佳 epoch，HM 也不是选择依据。
- **发布状态与运行证据分开。** 本仓库有冻结源码与本地工程测试，不因此宣称这个便携版本已经完成新的 RTX 4090 端到端复现。新运行应产生自己的初态、训练和首次独立重载证据，不能沿用历史成绩充当新运行结果。

如需修改上述层位、宽度、VJP、损失、后端或启动策略，应将其作为新的、明确登记的协议，而不是在原方案名称下静默更换。
