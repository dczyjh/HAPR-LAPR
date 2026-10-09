# LARP：实际计算与复现入口

本文对应本仓库冻结的 **O-only LARP**，主方案是 `o4-r2`：保留 PromptKD 提示蒸馏，在学生 ViT-B/16 的最后四个视觉 Transformer 块中，仅给注意力输出投影 O 加低秩权重增量。它与 HARP 是独立路线，不是 HARP 与 LoRA 叠加。这里解释实际实现，不宣称某个变体必然涨分，也不把本地 CPU 检查称为 GPU 复现通过。

先读[运行步骤](RUN_REPRODUCTION.md)准备环境和外部物料；本页回答“模型到底算什么、哪些参数被训练、各消融改变什么”。完整数值配置以[冻结配置模板](../routes/larp/protocol/resolved_templates.json)、[固定协议](../routes/larp/protocol/protocol.json)和[变体表](../routes/larp/protocol/variants.json)为准。

## 1. 从图像到蒸馏损失

学生保留原 ViT-B/16 的 patch、CLS token、位置编码、视觉提示、注意力、MLP、残差和 LayerNorm。224×224 输入对应 14×14 个图像 patch；加入 CLS 和 4 个视觉提示后，序列长为 201。实际 Transformer 张量布局为 `[L, N, d]`，其中 `L=201`、训练批大小 `N=8`、学生隐层宽度 `d=768`。提示深度为 9：入口放入 4 个提示，后续指定块替换上一层提示，而不是不断追加新 token。[视觉序列与提示替换源码](../routes/larp/archive/project/clip/model.py#L191)

学生视觉编码器输出 512 维图像特征，经过原 `VPT_image_trans` 投影器变为 768 维，再做 L2 归一化。投影器是 `1×1 Conv(512→768) → BatchNorm → ReLU → 1×1 Conv(768→768)`，并非新增的 LARP 模块。教师为加载固定教师 checkpoint 的 ViT-L/14 PromptKD；教师图像特征和缓存的教师文本类别向量同为 768 维。学生和教师各自保留其原有的冻结 `logit_scale`。学生使用教师的同一组文本向量计算 logits，不另外训练学生文本分类器。[投影器、学生和教师前向源码](../routes/larp/archive/project/trainers/promptkd.py#L31)

设归一化后的学生/教师图像特征为 $f_s,f_t\in\mathbb{R}^{N\times768}$，教师文本类别向量为 $T\in\mathbb{R}^{C\times768}$，则：

$$
z_s=\exp(a_s)f_sT^\top,\qquad
z_t=\exp(a_t)f_tT^\top.
$$

训练时 `C` 是该数据集完整类别集合的大小，而不是评估时单独的 Base 或 Novel 类别数。标签仍由数据读取器提供并用于输入审计，但本方案训练损失没有标签交叉熵项。评估时才按原 Base/Novel 划分选取对应文本类别向量。[训练 logits 与损失](../routes/larp/archive/project/trainers/promptkd.py#L518)、[评估实现](../routes/larp/archive/project/trainers/promptkd.py)

## 2. O-only 参数化：改权重，不重写注意力

在选定块中，原 `nn.MultiheadAttention` 仍执行 Q/K/V 投影、多头注意力和 O 投影；`attention()` 仍调用 `self.attn(x, x, x, need_weights=False, ...)`。LARP 通过 PyTorch `parametrize.register_parametrization(attention.out_proj, "weight", ...)` 改变读取到的 O 权重，没有替换 MHA 前向，也没有加入 HARP 并行残差。[原注意力调用](../routes/larp/archive/project/clip/model.py#L229)、[O 参数化及注入](../routes/larp/archive/project/trainers/efficient_adaptation.py#L131)

令 $X=\mathrm{LN}(x)$ 是 MHA 的输入，$H=\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_{12})$ 是 O 投影之前的注意力输出。采用 PyTorch 线性层的权重存储约定：

| 对象 | 实际维度 | 本方案是否训练 |
| --- | --- | --- |
| $X,H$ | $L\times N\times768$ | 激活，不是独立参数 |
| 原 packed Q/K/V 权重 | $2304\times768$ | 冻结 |
| 原 O 权重 $W_O$ | $768\times768$ | 冻结 |
| 原 O bias $b_O$ | $768$ | 冻结 |
| $A$ / `lora_A_o` | $r\times768$ | 训练 |
| $B$ / `lora_B_o` | $768\times r$ | 训练 |

$$
s_r=\frac{\alpha}{\sqrt r},\qquad
\Delta W_O=s_rBA,\qquad
\widetilde W_O=W_O+\Delta W_O,
$$

$$
Y=H\widetilde W_O^\top+b_O
 =HW_O^\top+s_r(HA^\top)B^\top+b_O.
$$

最后一式只是代数展开。**实际代码先形成 `weight + (B @ A) * scaling`，再让原 MHA 使用该权重**；不能据展开式声称实现了两段低秩前向、合并推理权重或必然降低注意力 FLOPs。MHA 中的 softmax、Q/K/V、头数和残差结构均保留；`LORA_DROPOUT=0`，非零 dropout 在本实现中直接报错。[精确权重表达式](../routes/larp/archive/project/trainers/efficient_adaptation.py#L147)、[参数化配置约束](../routes/larp/archive/project/trainers/efficient_adaptation.py#L250)

初始化时 $A$ 使用 `kaiming_uniform_(a=sqrt(5))`，$B=0$，所以初始有效权重为原 $W_O$。这是数学上的零增量初态；不能据此保证不同 CUDA 路径、机器或运行上下文的全部输出逐位相同，发布运行器仍执行真实首步门槛。

## 3. 主方案与消融

所有变体均固定 `alpha=1`，因此 rank 变化同时改变 $1/\sqrt r$ 缩放，而不是保持同一个数值缩放。源码层号从 0 开始。

| 角色 | 零基层号 | 按自然数计数的层 | rank | 缩放 | 新增参数 |
| --- | --- | --- | --- | --- | --- |
| `r1` | 无注入 | 无 | 不适用 | 不适用 | 0 |
| `o4-r2` 主方案 | 8,9,10,11 | 第 9–12 层 | 2 | $1/\sqrt2$ | 12,288 |
| `o4-r1` | 8,9,10,11 | 第 9–12 层 | 1 | 1 | 6,144 |
| `o4-r4` | 8,9,10,11 | 第 9–12 层 | 4 | $1/2$ | 24,576 |
| `o3-r2` | 9,10,11 | 第 10–12 层 | 2 | $1/\sqrt2$ | 9,216 |

每层新增数为 $768r+768r=1536r$。rank 消融有容量与缩放耦合；后三层对后四层比较同时改变层覆盖和参数量，不是“只改变层位置”的等容量因果实验。`o3-r2` 也不是历史的第 9–11 层方案。以上边界不因结果较好或较差而改变。[变体定义](../routes/larp/protocol/variants.json)

`r1` 是 LARP 路线的匹配控制：不注入 A/B，但保留下面的共同投影器渐升；它不是另一路的官方 `r0`。配置中 R1 仍保留失活的旧 `LORA_TARGETS=["v"]` 等字段，因为 `ADAPTATION.TYPE="none"`，这些字段不会创建 V 或 O 参数。不可直接采用源码目录里的历史默认配置来替代发布入口解析后的 O-only 配置。

## 4. 实际训练与冻结范围

公共可训练参数为视觉提示和原投影器：9 组 `4×768` 提示共 27,648 参数；两层投影器含卷积 bias 与 BN affine，共 986,112 参数；合计 1,013,760。主方案再加 12,288 个 O 的 A/B 参数，总计 1,026,048。适配器会按变体核对实际可训练参数数目。[训练参数选择](../routes/larp/archive/project/trainers/promptkd.py#L411)、[发布构建检查](../routes/larp/portable.py)

冻结范围包括原学生 CLIP 图像主干权重、原 Q/K/V/O 权重及 bias、LayerNorm/MLP、原视觉输出投影和 `logit_scale`；整个教师包含其视觉/文本编码器及教师提示均冻结、处于 eval 模式。学生的梯度仍需经过冻结主干传播，不能把整个学生编码器包进 `no_grad()`。投影器 BN 的 running mean/variance 是训练态会更新的 buffer，不是“冻结参数被改坏”。

`PREC="fp16"` 不意味着每个张量都为 FP16。源码按 CLIP 原规则将 Conv/Linear/MHA 等适用权重转半精度，保留 LayerNorm 等原有精度；视觉提示及 BN affine 不应被一律强制 `.half()`。新增 A/B 明确与注意力权重同 dtype，即 FP16。不能为便携运行偷偷改成 AMP、全 FP32 或新增 FP32 master 参数。[原精度转换](../routes/larp/archive/project/clip/model.py#L657)、[A/B dtype 来源](../routes/larp/archive/project/trainers/efficient_adaptation.py#L235)

LARP 保留 `cudnn.benchmark=True`、`deterministic=False`、deterministic algorithms 关闭，以及 SDP 的 `flash=True / efficient=True / math=True`。`efficient=True` 与 HARP 路线的设置不同，不能跨 route 复用已导入的进程或自行统一后端。固定后端仍不是逐位确定性承诺。[冻结后端字段](../routes/larp/protocol/protocol.json)

## 5. 原 KD：`sum / (N × C)`，不是 `batchmean`

源实现为：

```python
raw_kd = F.kl_div(
    F.log_softmax(student_logits / temperature, dim=1),
    F.softmax(teacher_logits / temperature, dim=1),
    reduction="sum",
) * temperature**2 / student_logits.numel()
loss = KD_WEIGHT * raw_kd
```

等价地，令 $p_t=\mathrm{softmax}(z_t/\tau)$、$p_s=\mathrm{softmax}(z_s/\tau)$：

$$
\mathcal L=\lambda_{KD}\frac{\tau^2}{NC}
\sum_{i=1}^{N}\sum_{c=1}^{C}
p_{t,ic}\log\frac{p_{t,ic}}{p_{s,ic}}.
$$

固定 $\tau=1$；DTD、FGVC Aircraft、Oxford Flowers 的 `KD_WEIGHT=200`，其余七集为 1000。`CE_WEIGHT=0`、`REP_WEIGHT=0`，没有标签 CE、额外 R1-reference KL 或表征损失。虽然源码保留其他实验分支，此路线不启用；不能把 `reduction="sum" / numel` 换成 PyTorch `batchmean` 或另加一个所谓“稳定 KL”。[实际损失函数](../routes/larp/archive/project/trainers/promptkd.py#L518)、[各集固定配置](../routes/larp/protocol/resolved_templates.json)

## 6. 普通 SGD 与公共投影器第 2 轮渐升

固定普通 `torch.optim.SGD`：momentum 0.9、weight decay 0.0005、dampening 0、Nesterov 关闭。A 与 B 使用相同实际学习率，均为公共组学习率的 0.1 倍；从首个 batch 起参与更新，没有 HARP 的新增组停更，也没有 A/B 16 倍异速。[参数分组](../routes/larp/archive/project/trainers/adaptation_optim.py)、[原 SGD 构造](../routes/larp/archive/project/Dassl.pytorch/dassl/optim/optimizer.py#L105)、[保留的实际 step 安装器](../routes/larp/protocol/original_sgd.py)

用数学式概括每个有梯度的参数 $\theta$：

$$
g_t=\nabla_\theta\mathcal L_t+0.0005\theta_t,\qquad
v_t=0.9v_{t-1}+g_t,\qquad
\theta_{t+1}=\theta_t-\eta_{\theta,t}v_t.
$$

实际运算与舍入由固定版本 PyTorch SGD 按各参数 dtype 执行；新增 FP16 A/B 的动量也为普通 FP16 SGD 状态，并无额外 FP32 master 副本。零 B 会使第一步 A 的**数据损失梯度**为零，但 A 未被代码冻结，SGD 仍包含 weight decay 项；实际 FP16 更新是否可见还受舍入影响。源码在更新前检查 loss/梯度、更新后检查参数有限性，失败就停，不自动裁剪梯度或改精度。[原训练步](../routes/larp/archive/project/trainers/promptkd.py#L569)

公共组基准 LR 为 0.005，首轮使用常数 warmup 0.00001，之后沿原 cosine scheduler。A/B 在 warmup 也保持 0.1 倍，即首轮 0.000001。scheduler 在每轮最后一个 batch 内由 `forward_backward()` 调用一次，外层不能再额外 `step()`。[原 warmup/cosine 实现](../routes/larp/archive/project/Dassl.pytorch/dassl/optim/lr_scheduler.py)

只有共同投影器在**第 2 轮**使用逐 batch 线性渐升。设本轮有 $K>1$ 个 batch、零基 batch 下标 $b=0,\ldots,K-1$：

$$
\eta_{proj}(2,b)=10^{-5}+(0.005-10^{-5})\frac{b}{K-1}.
$$

该轮视觉提示 LR 仍为 0.005，A/B LR 仍为 0.0005，不随投影器渐升；其他轮投影器使用公共名义 LR。安装器仅在这轮暂时拆分 optimizer 参数组、执行**一次**原 SGD step，再恢复组结构，不重置参数或动量。R1 和全部 LARP 候选使用相同投影器调度。[实际渐升及临时分组源码](../routes/larp/protocol/projector_schedule.py)

## 7. 新运行的初始化配对

源码把新增模块初始化放在 `torch.random.fork_rng(devices=[])` 内；在当前 CPU 构建流程中，它恢复新增 A/B 所消耗的 CPU RNG，避免扰动之后的公共投影器初始化和数据流。发布运行器还检查自然构建出来的公共参数 SHA 与 CPU/CUDA RNG SHA，而不是事后把不同的公共模型覆盖成一样。[源码初始化位置](../routes/larp/archive/project/trainers/promptkd.py#L241)、[新运行初态记录与核对](../routes/larp/portable.py)

同一个 dataset/seed 的 `o3-r2` 首次初始化，用本次新建、尚未训练的 `o4-r2` 作为锚点，按实际层名复制交集层 9、10、11 的 A/B；不按“第几个新增模块”顺序错位复制。之后 gate、正式训练、独立评估分别核对本角色自己的已登记初态。rank1/rank4 保留各自自然初始化，不截断或填充 rank2 的 A/B。

不同阶段使用新进程，不拿四步 gate 更新后的模型继续正式训练；公共参数 hash 不代替新增张量，初态记录中 A/B 保留实际 CPU tensor。全训练 batch 的增强图像、标签、路径和公共 LR 继续与本次 R1 配对。配对减少特定混杂，但不能取消数据划分、rank 缩放、非确定性 CUDA 或跨机差异的解释边界。

## 8. 与真实入口对应的伪代码

下面是流程说明，不是替代被冻结函数的简化训练程序：

```text
解析 dataset/seed/variant 的完整冻结配置，仅替换外部物料与输出路径
检查源码/配置/权重/划分元数据身份，读取实际数据后再核类别与有序划分
构建新初态；共同参数和 RNG 自然配对，必要时做同 seed O4→O3 物理层映射
在独立新进程做每角色 4 步 gate；不把 gate 状态交给正式训练

先正式训练本次 R1，首次独立重载通过后再处理候选：
  在新进程从登记初态重新构建真实 PromptKD trainer
  安装原 SGD 包装器与共同投影器第 2 轮渐升
  for epoch in 1..20:
    for 每个完整训练 batch:
      teacher.eval + no_grad：计算/复用教师文本特征，计算教师图像 logits
      student：提示 + 原视觉块（选定层 O 权重参数化）+ 原投影器
      计算唯一原 KD，zero_grad，backward，有限性检查
      按该 epoch/batch 的实际 LR 执行一次原 SGD step
      记录全部输入身份、实际 LR、loss；轮末在原 forward_backward 内更新 scheduler
    原 after_epoch 在 Base 上评估，严格变好才保存 best，平分保留较早轮
    第 20 轮另外保存 last
  原 after_train 加载同一 best，记录 Base 和 Novel 的首次 native 证据
  新进程首次独立重载该 best，按既定硬限核对；通过后才给出 audited_metrics
  任何硬失败：保留第一份现场，停止后继，不自动重试/改阈值/择高结果
```

best 由 Base 选择，不按 Novel 或 HM 挑点；Base 和 Novel 必须来自同一已保存 best。独立重载的硬核验不是只看预测类别相同，更不是因为 HM 接近就放行。[训练/终评源码](../routes/larp/archive/project/trainers/promptkd.py)、[原证据比较函数](../routes/larp/archive/helpers/larp_two_dataset_ramp_20260929/evaluation_evidence.py.txt)、[发布阶段运行器](../runtime/execution.py)

## 9. 怎样调用

从仓库根目录先查看一个数据集、一个 seed 的计划；`plan` 不启动 GPU：

```bash
python reproduce.py plan --route larp --dataset dtd --seed 1 \
  --data-root /data/datasets --clip-root /data/pretrained/clip \
  --teacher-root /data/pretrained/teacher_model --output-root /runs/larp_dtd_s1
```

准备好固定环境/物料、确认预算后，把 `plan` 改为 `run` 才会真正启动。默认是 R1 + `o4-r2`；加 `--variant o3-r2` 请求末三层候选时仍会加入匹配 R1，并建立所需 O4 初始化锚，不因此自动多训练一个未请求的主方案。`--all-variants` 包含表中五个角色；不指定 dataset/seed 会扩为十集×三 seed，不能误当成一次小测试。[完整操作与资源要求](RUN_REPRODUCTION.md)

不得直接运行 `archive/` 中带历史服务器路径的 worker，也不要把旧 V-only、额外 KL、FP32 master、A/B 异速或层位探索结果混入这五个固定角色。`archive/` 是保留的源码/来源，执行入口是根 `reproduce.py` 经 [LARP portable adapter](../routes/larp/portable.py) 接线。已完成历史结果与本次新输出分开保存；是否重现历史分数须等待目标 GPU 上实际门槛、训练及首次重载的证据，本文不预先承诺。
