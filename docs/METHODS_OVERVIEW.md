# 方法原理：提示蒸馏、HARP与LARP

本页说明本项目实际实现，不保证任何变体必然涨分。HARP与LARP是两条独立路线，细节见[HARP公式与伪代码](HARP_IMPLEMENTATION.md)和[LARP公式与伪代码](LARP_IMPLEMENTATION.md)。固定参数见[实验协议](EXPERIMENT_PROTOCOL.md)，如何运行见[运行指南](RUN_REPRODUCTION.md)。

## 1. 师生结构与任务

任务为Base-to-Novel分类，分别评估Base与Novel类别。学生训练接触两侧的**无标签图像**，模仿教师分布；不是只用Base图像的标签交叉熵训练。标签由原读取器提供，用于输入配对/评估，不进入最终KD分类损失。

教师为加载对应数据集提示检查点的CLIP ViT-L/14，提供图像表示、全部类别文本表示和logits，不参与学生优化。学生从原CLIP ViT-B/16初始化，训练视觉提示与语义投影器，再按路线加入HARP或LARP；原学生骨干权重不做完整微调。学生使用教师文本类别向量，不另训练学生文本编码器。

实际代码：[HARP PromptKD](../routes/harp/vendor/project/trainers/promptkd.py)、[LARP PromptKD](../routes/larp/archive/project/trainers/promptkd.py)；R0由隔离的[官方实现](../routes/larp/archive/official_reference/trainers/promptkd.py)执行。

## 2. 特征对齐与分类

设batch大小B、完整类别数C。教师归一化类别文本特征`E`为`C×768`，教师归一化图像特征`v_t`为`B×768`。学生视觉块内部宽度768，经其原CLIP视觉投影后输出`u_s`为`B×512`。原`VPT_image_trans`将512映射至768，再L2归一化为`v_s`。

实际投影器是`Conv1×1(512→768) → BN → ReLU → Conv1×1(768→768)`，不是另加一层任意Linear，也不是新增HARP/LARP模块。视觉块内部768维与编码器输出512维是不同位置。

```text
z_t = exp(logit_scale_t) · v_t · E^T   # B×C
z_s = exp(logit_scale_s) · v_s · E^T   # B×C
```

师生保留各自原logit_scale，不强行改成共同新参数。真实dtype、归一化和算子顺序以冻结源为准。

## 3. 唯一训练目标：原KD

最终不启用RP、额外R1-KL或标签CE。温度T=1，KD系数λ在DTD/Aircraft/Flowers为200，其他七集为1000。源码计算关系：

```python
p_teacher = softmax(z_t / T, dim=1)
log_p_student = log_softmax(z_s / T, dim=1)
loss = lambda_kd * kl_div(log_p_student, p_teacher, reduction="sum")
loss = loss * T * T / z_s.numel()
```

分母是`B×C`，不是仅除B的`batchmean`。数学上为`λT²/(BC) · Σ_b KL(p_t(b)||p_s(b))`，方向是教师分布到学生分布。上面是计算关系伪码，不授权重写实际FP16算子；教师前向在no_grad内，不更新其提示/编码器。

## 4. 两条方法在基础结构上改什么

| 路线 | 实际改变 | 要检验的问题 |
| --- | --- | --- |
| HARP | 学生后四层非线性瓶颈分支，宽32、零输出起点、scale0.001、fix2自定义VJP、新增组停更/渐升 | 高层非线性适配能否改善匹配提示蒸馏控制 |
| LARP | 只参数化学生后四层注意力O权重，`W_eff=W+alpha/sqrt(r)·B·A`，主rank2，保留原MHA | 小型低秩O适配能否改善匹配控制 |

HARP不是串接LARP，LARP不启用HARP。新增模块初始化不扰动公共投影器/数据RNG流，并核共同初态。结构变体的物理层映射来自新建且未训练的锚，不从已训练主方法取权重。

## 5. R0与R1为什么都保留

R0保留原框架单组SGD/原调度，提供原框架总体参照。各路线R1关闭新增模块，保留同路线共同投影器第2轮渐升和其他共同训练设置，隔离新增模块贡献。主方法优于R0不能代替优于匹配R1的证据；R1不是第三个创新方法，两路线不共享一个R1。

工程监督器记录初始化、输入/LR配对、保存/重载与失败现场，是执行保障，不是第三个模型或新增损失。完整用途见[工程总览](PROJECT_MAP.md)。

## 6. 比较结果怎样产生

每角色从原初始化训练20轮；每轮Base严格提高才存best，同值保留较早轮。同一个best用于Base/Novel，实际是**Base-test选点**，不能称独立验证集选点。首次独立重载先保存成绩，通过原硬检查后才写正式audited_metrics。

`HM=2×Base×Novel/(Base+Novel)`。每seed分别算HM，再对1/2/3的HM求均值/样本标准差（ddof=1）；不是先平均两侧再算HM。增幅是百分点pp，即方法百分数减参考百分数，不是相对百分比增长。

单集单seed可先验收执行，但不能称完整十集三seed复现。工程通过不证明模块有效；历史results和新运行正式记录分开保存，不把两者混成同一批新实验。
