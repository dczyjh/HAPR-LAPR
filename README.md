# HARP / LARP：无监督提示蒸馏复现项目

本项目整合论文对应的原始 R0 基线、匹配 R1 控制、HARP、LARP 和既定结构消融。HARP 与 LARP 是两条独立路线，不是叠加模型。科学源码及完整实验配置来自实际运行快照，新增便携运行器负责路径、初始化、配对核验和结果管理，不重写原损失、优化器或训练循环。

首次接触本仓库，请先读[各项目与工程用途总览](docs/PROJECT_MAP.md)：说明R0/R1的不同对照作用、两条主方法与消融要检验的问题，以及正式运行器、工具、科学源码、历史归档和结果目录的职责。不是每个目录都要运行，历史worker不作为新训练入口。

## 我们的研究贡献

- **HARP**：在提示蒸馏学生的高层视觉块中引入非线性瓶颈残差，配套明确的尺度处理和更新规则，检验高层表示适配的识别收益。
- **LARP**：仅为高层注意力输出投影O增加低秩权重增量，保留原注意力计算，研究较小新增参数预算下的另一条蒸馏适配路线。
- **实验分析**：通过十集三seed主比较及位置、宽度、rank和层集合对照，分析实际收益、Base/Novel取舍与适用条件，不将参考方法或工程核验本身算作算法创新。

具体改进与参考工作的区别、实验证据和贡献边界见[本项目的研究贡献](docs/OUR_CONTRIBUTIONS.md)。两条路线分别有任务收益，不意味着所有任务或整体平均都提高。

## 从哪里开始

获取源码（数据与权重仍按下文单独准备）：

```bash
git clone https://github.com/dczyjh/HAPR-LAPR.git
cd HAPR-LAPR
```

使用者文档按阅读顺序组织：

1. [工程用途](docs/PROJECT_MAP.md)：各项目做什么、正式科学源与历史记录的分工。
2. [方法原理](docs/METHODS_OVERVIEW.md)：师生结构、特征对齐、KD与R0/R1；[HARP细节](docs/HARP_IMPLEMENTATION.md)、[LARP细节](docs/LARP_IMPLEMENTATION.md)给公式、训练范围和伪代码。
3. [物料准备](docs/ASSET_PREPARATION.md)：来源、目录、固定划分和权重指纹。
4. [运行指南](docs/RUN_REPRODUCTION.md)：环境、训练、主实验/消融、首次重载与汇总命令。
5. [参数与排错](docs/CLI_AND_TROUBLESHOOTING.md)：默认值、最小例子、结果字段和失败检查。
6. [验证范围](docs/VALIDATION.md)：通过、跳过及未执行项，不把本地测试当作论文精度复现。
7. [实验发现与结论](docs/RESULTS_AND_LIMITATIONS.md)：项目问题、最终修正、完整主比较、结构消融及效果边界。
8. [三篇参考论文与致谢](docs/REFERENCES_AND_ACKNOWLEDGEMENTS.md)：论文/代码地址、采用范围及可用BibTeX。

先按[物料准备](docs/ASSET_PREPARATION.md)取得图片、固定划分、CLIP 和教师权重，再按[运行指南](docs/RUN_REPRODUCTION.md)准备固定环境并执行。权重和图片不放进 Git；已有服务器日志不是新训练的前提。

```bash
python project.py verify
python reproduce.py --help
python reproduce.py plan --route harp --dataset dtd --seed 1 \
  --data-root /data/datasets --clip-root /data/pretrained/clip \
  --teacher-root /data/pretrained/teacher_model --output-root /runs/harp_dtd_s1
```

把 `plan` 改为 `run` 才会实际执行。未指定数据集或 seed 时默认十集、seed 1/2/3；HARP/LARP 默认分别跑 R1 与主方法，`--all-variants` 增加各自三个消融。R0 使用 `--route r0`。每项从原 CLIP/教师初始化开始，不加载训练过的学生。

| 内容 | 入口 |
| --- | --- |
| 各工程用途、实验角色与目录边界 | [工程用途总览](docs/PROJECT_MAP.md) |
| 从头训练、主实验、消融、首次独立重载 | [reproduce.py](reproduce.py) / [运行指南](docs/RUN_REPRODUCTION.md) |
| HARP fix2、完整 150 配置与双渐升 | [HARP](routes/harp/README.md) |
| LARP 后四层 O/rank2 与三种消融 | [LARP](routes/larp/README.md) |
| 官方 R0 原始配置及原 SGD | [R0](routes/r0/README.md) |
| 数据/权重来源、目录与 SHA | [物料说明](docs/ASSET_PREPARATION.md) |
| 十集三 seed、训练与选点规则 | [实验协议](docs/EXPERIMENT_PROTOCOL.md) |
| 新运行的三 seed 均值、标准差和增幅 | [summarize_run.py](tools/summarize_run.py) |
| 已完成实验的轻量结果 | [results](results/README.md) |
| 实际提升、负结果、消融及发现的问题 | [实验结论](docs/RESULTS_AND_LIMITATIONS.md) / [主实验汇总表](results/findings/main_comparison.md) |
| 我们提出什么、与参考工作有什么区别 | [研究贡献](docs/OUR_CONTRIBUTIONS.md) |
| 主要参考论文、采用范围和致谢 | [引用与致谢](docs/REFERENCES_AND_ACKNOWLEDGEMENTS.md) / [references.bib](references.bib) |
| 实际测试范围与验证报告 | [验证说明](docs/VALIDATION.md) / [报告](metadata/validation_report.json) |

新运行自动建立同 seed 的共同初始化、每角色四步工程检查、R1/候选全 batch 输入和公共学习率配对；保留实际 best/last SHA，首次独立重载同一个 best 评估 Base/Novel。遇到硬检查失败保留现场并停队列，不覆盖已有输出、不自动重试、不放宽阈值。

发布代码已做离线完整性、科学函数、小型 CPU 完整执行链测试；本次整理未重新执行真实 CLIP 的 CUDA 训练，因此不把工程测试称为论文分数重现。实际数据和固定 Linux/CUDA 环境仍须在启动时验收；[外部文件指纹](metadata/external_assets.json)是物料身份约束，公开教师包若不同必须取得对应文件，不能修改期望 SHA 来放行。

这是独立 Git 发布项目，不包含上层论文工作区、密码、私钥、数据、权重和运行环境。第三方许可见[来源与许可](THIRD_PARTY_NOTICES.md)；自有新增代码的开放许可尚待作者指定。代码托管不等于已在新环境重现历史分数。

## 参考与致谢

感谢 [PromptKD](https://github.com/zhengli97/PromptKD)、[MMA](https://github.com/ZjjConan/VLM-MMA) 和 [CLIP-LoRA](https://github.com/MaxZanella/CLIP-LoRA) 作者公开论文与代码。PromptKD提供基础蒸馏框架，MMA和CLIP-LoRA分别提供高层瓶颈与低秩更新的设计参考；本项目不声称完整复现或叠加三篇方法。完整论文著录、采用边界及英文致谢见[引用与致谢](docs/REFERENCES_AND_ACKNOWLEDGEMENTS.md)。
