# 各项目与工程的用途

这是一个统一复现仓库，包含原始基线和两条独立研究路线，并非需要把所有目录依次执行的多个模型。HARP、LARP均保留提示蒸馏基础结构，但各自添加不同模块；两者不互相加载、不串联，也不叠加。正式运行优先使用根目录`reproduce.py`，其他目录的角色如下。

## 1. 科研角色：各实验用来回答什么问题

| 角色/工程 | 用途与研究问题 | 不是用来做什么 | 执行选择 |
| --- | --- | --- | --- |
| 原始基线R0 | 按原提示蒸馏框架的实际基线配置从头训练，为研究方法提供原框架参照 | 不是直接抄论文网站分数，也不是加载已训练学生继续训练 | `--route r0`，角色`r0` |
| HARP匹配控制R1 | 关闭HARP，保留本路线共同训练结构/投影器渐升和后端，隔离新增模块相对匹配控制的贡献 | 不是另一个创新方法；不能与R0合并 | `--route harp --variant r1` |
| HARP主工程 | 检验视觉高层非线性瓶颈适配是否改善提示蒸馏的Base/Novel表现；主结构后四层、宽32 | 不自动保证提升，不按数据集换成最优消融 | `--route harp --variant harp` |
| HARP结构消融 | 比较位置（中四层/后四层）及宽度16/32/64，检查主结构选择及容量作用 | 不是三种新的主方法；不把位置/容量比较夸大为严格单一因果证明 | `mid4_d32`、`last4_d16`、`last4_d64` |
| LARP匹配控制R1 | 关闭LoRA，保留本路线共同训练结构/投影器渐升和后端，提供新增低秩模块的匹配控制 | 不借用HARP的R1代替本路线控制 | `--route larp --variant r1` |
| LARP主工程 | 检验注意力输出投影O的低秩适配是否改善提示蒸馏；主结构后四层、rank2 | 不是Q/K/V全部加LoRA，不与HARP组合 | `--route larp --variant o4-r2` |
| LARP结构消融 | 比较rank1/2/4与末三层/后四层，检查低秩容量和作用层数 | rank同时影响缩放，层数也改变参数量，不能宣称只改变单一独立因素 | `o4-r1`、`o4-r4`、`o3-r2` |

R1是**角色**，并没有第三个独立`routes/r1`工程；它分别存在于HARP和LARP的配置及运行目录。R0保留官方单组SGD和原调度，不加入研究路线的投影器专用第2轮渐升。两条R1则保留这项共同规则，因此R0与R1是不同对照层级。

优先看“主方法−本路线R1”，判断新增模块相对匹配训练结构的效果；同时列“主方法−R0”，展示相对原框架的总体差异。只高于R0不能自动证明新增模块有效；消融也不必高于主方法，重点是解释模块/容量/位置选择。是否有效以完整配对真实结果为准，不能由工程名称或设计动机直接下结论。

主实验与消融均按十集、seed1/2/3、每项20轮登记；不含ImageNet。具体参数以[固定协议](EXPERIMENT_PROTOCOL.md)与完整resolved配置为准，不以早期v3默认值为准。

## 2. 启动与公共执行工程

| 路径 | 实际职责 | 是否会训练 |
| --- | --- | --- |
| `project.py` | 统一命令分发，转到各路线检查或根复现入口；自身不定义模型 | 只有转发到`train/run`才会 |
| `reproduce.py` | 正式复现入口；选择route/数据集/seed/变体，打印计划或启动冻结任务 | `plan`不训练；`run`实际训练；`evaluate`实际评估 |
| `runtime/execution.py` | 公共监督器：新初始化、各四步检查、原20轮训练循环、R1输入/LR配对、best/last核验、首次独立重载、失败留存 | 被`run/evaluate`调用时会执行 |
| `routes/harp/portable.py` | 从HARP完整模板构建原模型/读取器；接原fix2和双渐升，核新增/共同初态 | 由监督器调用，不直接跑旧队列 |
| `routes/larp/portable.py` | 构建原LARP模型/读取器；接普通FP16 SGD与共同投影器渐升，映射共享物理层初态 | 同上 |
| `routes/r0/portable.py` | 构建官方R0模型；建立相对权重路径视图，透明记录原单组SGD，不添加研究模块/渐升 | 同上 |
| `routes/harp/cli.py`、`routes/larp/cli.py` | 路线专用配置/源码检查入口；`train`转发正式复现入口 | `check/plan/preflight`不训练；`train`会 |

公共监督器是工程保障，不是论文中新提出的第三个模型或额外损失。它记录执行是否满足规则，不因检测失败自动改变科学参数。`evaluate`是第一次独立重载入口，不是反复测分择优的功能。

## 3. 科学代码、配置、归档各放哪里

| 目录 | 用途 | 使用边界 |
| --- | --- | --- |
| `routes/harp/vendor/project/` | HARP实际科学实现，包括学生/教师、适配器、自定义VJP、原读取器与Dassl | 正式入口加载这里的实现；不直接调用底层`train.py`代替最终策略 |
| `routes/harp/runtime/frozen_install.py` | 原HARP新增组停更/渐升及共同投影器更新规则，保留科学函数AST | 由HARP adapter安装；不是独立训练脚本 |
| `routes/harp/configs/portable/` | 十集三seed五角色共150份完整配置，外部路径模板化 | 正式配置来源，不自行改KD/LR/轮数 |
| `routes/harp/archive/recorded_configs/` | 150份实际resolved YAML快照 | 原配置溯源，历史绝对路径不是新机运行路径 |
| `routes/harp/historical/` | 旧控制/恢复运行器及环境记录 | 只追溯，不启动；新运行不要求旧日志存在 |
| `routes/larp/archive/project/` | 最终LARP研究实现，含O参数化、原模型、读取器/优化器与Dassl | 正式LARP adapter加载这里；文件中未启用的通用分支不属于本轮方法 |
| `routes/larp/archive/official_reference/` | 隔离的官方参考实现，也是新R0训练的实际源码来源 | 仅通过R0 adapter与R0实际配置使用；上游默认值不能替代研究协议 |
| `routes/larp/archive/helpers/`、`final_worker/`、`main_workers/` | 最终辅助源及旧队列文本证据 | 只追溯，不直接执行历史worker |
| `routes/larp/protocol/` | 完整模板、数据集身份、五角色声明、原SGD及共同投影器调度 | LARP正式配置/函数来源 |
| `routes/r0/templates.json`、`provenance.json` | 实际R0完整配置模板及原快照来源 | R0使用，不替代两条R1 |
| 各route的`environment/` | 对应真实环境版本与可迁移依赖说明 | 不是已经安装好的环境或容器；下载可用性/新Linux安装未完整验证 |
| 各route的`provenance/`、`dataset_reference.json` | 原源码SHA、科学函数、配置、数据成员/类别/计数依据 | 用于核对，不是模型权重或图像文件 |

第三方PromptKD、CLIP、Dassl是基础依赖；MMA与CLIP-LoRA是设计参考，不是要在本仓库另外依次训练的第三/第四条方法。CLIP-LoRA参考仓库没有作为独立工程复制进来。来源、归属与许可见[第三方说明](../THIRD_PARTY_NOTICES.md)，不要将所有上游代码称为本项目原创。

## 4. 工具、结果及元数据

| 路径 | 用途 | 不会做什么 |
| --- | --- | --- |
| `tools/prepare_assets.py` | 列出来源/目标/指纹；显式`--download`仅获取两个固定官方CLIP文件 | 不自动下载十教师、不训练教师、不安装环境 |
| `tools/check_assets.py` | 只读核12份权重及13份固定元数据的大小/SHA | 不核全部图片字节，不执行模型 |
| `tools/verify_release.py` | 核本发布文件SHA、语法、清单与敏感信息模式 | 不验收GPU或论文准确率 |
| `tools/summarize_run.py` | 汇总新运行passed正式成绩、三seed均值/样本SD及R1/R0配对增幅 | 不重评模型、不补缺seed、不择优、不重复计数 |
| `tests/`及各route的`tests/` | 配置/科学函数/工具及小型CPU执行链测试 | 合成测试不是论文数据，不能当完整FP16 CLIP精度验收 |
| `results/harp/`、`results/larp/` | 已完成实验的轻量成绩、轨迹和来源副本，便于查询及写作对照 | 不是新训练输出，不含全量检查点，不覆盖原始记录 |
| `metadata/external_assets.json` | 精确外部权重/固定划分身份约束 | 不包含资产本身，不允许改SHA放行不同文件 |
| `metadata/publication_sources.json` | 历史结果副本的来源/转换/SHA | 不证明本次发布重新训练成功 |
| `metadata/release_manifest.json` | 本发布版本逐文件SHA | 不等于Git远端地址或环境镜像 |
| `metadata/validation_report.json` | 本轮真实测试通过/跳过/未验证范围 | 不把本地CPU测试称成真实GPU成绩复现 |
| `docs/`、各route的`README.md` | 物料、参数、执行、目录及验证范围说明 | 不自动启动实验 |

## 5. 模型物料的用途

- `ViT-B-16.pt`：学生的原始CLIP初始化。不是训练完成的HARP/LARP学生成绩。
- `ViT-L-14.pt`：教师CLIP基础权重；还需对应数据集的预训练教师提示检查点。
- `teacher_model/<DatasetClass>/VLPromptLearner/model-best.pth.tar`：对应数据集教师，提供类别文本表示/蒸馏指导，不参与学生优化；不同数据集教师不能混用。
- BPE词表：文本分词运行资源，已随源码提供；不是可随意删除的下载缓存。
- 图像与固定JSON/TXT：原读取器的成员、类别与Base/Novel划分；训练取Base/Novel无标签图像，不能缺文件后随机重新划分。
- 新运行的学生`model-best.pth.tar`：由每轮Base严格最高选出，用同一文件评Base/Novel；`model.pth.tar-20`用于末轮状态核验。它们不是教师或下一个方法的初始化。

图片、模型大权重、环境二进制不随Git提供，按[物料说明](ASSET_PREPARATION.md)单独准备。新运行的日志/初态/检查点/首次重载证据保存在用户指定的新`output-root`，不写入历史`results/`。全部角色从原初始化开始，不把训练过的R1、HARP或LARP检查点作为另一个主方法的起点。

## 6. 使用者最短路径

1. 查本页，确定要跑R0、某条主路线或其消融。
2. 按[物料准备](ASSET_PREPARATION.md)与[运行指南](RUN_REPRODUCTION.md)准备固定环境/数据/权重，并执行只读检查。
3. `reproduce.py plan --route ...`看范围；确认后`run`使用全新输出根。主方法默认包含匹配R1，消融用`--all-variants`或重复`--variant`。
4. 检查该运行`status.json`及正式`audited_metrics.json`，只汇总核验通过项。
5. 用`summarize_run.py`生成单seed与完整三seed对照，不以已有`results/`冒充本次新运行成绩。

本页说明的是当前交付用途，不新增训练或服务器操作。是否已在新环境重现历史分数，以真实GPU运行与审计回执为准。
