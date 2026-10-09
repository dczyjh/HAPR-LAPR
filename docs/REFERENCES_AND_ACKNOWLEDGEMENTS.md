# 三篇主要参考论文、项目来源与致谢

本项目的基础框架来自无监督提示蒸馏方法 **PromptKD**；**MMA** 和 **CLIP-LoRA** 分别为高层瓶颈适配和视觉语言模型低秩适配提供设计参考。HARP 与 LARP 是在共同蒸馏框架下实现的两条独立路线，不是将三篇论文的完整方法串联或叠加。本页列出来源及实际采用范围，便于使用者引用与核查。

## 1. PromptKD：基础蒸馏框架

Zheng Li, Xiang Li, Xinyi Fu, Xin Zhang, Weiqiang Wang, Shuo Chen, Jian Yang. **PromptKD: Unsupervised Prompt Distillation for Vision-Language Models.** CVPR 2024, pp. 26617–26626。

- [正式论文及会议著录](https://openaccess.thecvf.com/content/CVPR2024/html/Li_PromptKD_Unsupervised_Prompt_Distillation_for_Vision-Language_Models_CVPR_2024_paper.html)
- [论文 PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_PromptKD_Unsupervised_Prompt_Distillation_for_Vision-Language_Models_CVPR_2024_paper.pdf)
- [作者 GitHub 项目](https://github.com/zhengli97/PromptKD)

本项目保留其固定教师、教师类别文本缓存、学生视觉提示、语义投影器和输出分布蒸馏基础结构。原框架复现结果是 R0；各研究路线另有匹配公共训练条件的 R1。学生使用 Base 与 Novel 的无标签训练图像，实验采用发布实现的 Base 测试选 best 方式。本项目十集结果不能无条件等同原论文十一集平均，也不把本文新增模块或公共调度归为原作者提出。[实际实验协议](EXPERIMENT_PROTOCOL.md)

## 2. MMA：HARP 的高层适配设计参考

Lingxiao Yang, Ru-Yuan Zhang, Yanchen Wang, Xiaohua Xie. **MMA: Multi-Modal Adapter for Vision-Language Models.** CVPR 2024, pp. 23826–23837。

- [正式论文 PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_MMA_Multi-Modal_Adapter_for_Vision-Language_Models_CVPR_2024_paper.pdf)
- [作者 GitHub 项目及著录](https://github.com/ZjjConan/VLM-MMA)
- [原项目地址](https://github.com/ZjjConan/Multi-Modal-Adapter)（当前重定向至上面的 VLM-MMA）

MMA 为高层适配位置、瓶颈结构及梯度尺度处理提供研究线索；本项目 HARP 源码也保留 MMA-inspired 归属说明。HARP 实际只在学生视觉编码块中接入独立瓶颈残差，采用本项目的零输出初始化、fix2 指定反向和配套更新规则。没有移植完整 MMA 的文本适配器或共享跨模态投影，也没有重新运行原 MMA 的整套少样本实验。其原论文表现不能作为 HARP 已有效的证据；效果应看本项目匹配对照和结构消融。[HARP 实现](HARP_IMPLEMENTATION.md)、[HARP 实验发现](HARP_FINDINGS.md)

## 3. CLIP-LoRA：LARP 的低秩适配设计参考

Maxime Zanella, Ismail Ben Ayed. **Low-Rank Few-Shot Adaptation of Vision-Language Models.** CVPR Workshops 2024, pp. 1593–1603。

- [正式论文及会议著录](https://openaccess.thecvf.com/content/CVPR2024W/PV/html/Zanella_Low-Rank_Few-Shot_Adaptation_of_Vision-Language_Models_CVPRW_2024_paper.html)
- [论文 PDF](https://openaccess.thecvf.com/content/CVPR2024W/PV/papers/Zanella_Low-Rank_Few-Shot_Adaptation_of_Vision-Language_Models_CVPRW_2024_paper.pdf)
- [作者 GitHub 项目](https://github.com/MaxZanella/CLIP-LoRA)

该工作为在 CLIP 注意力映射中使用低秩更新提供应用参考。LARP 在原提示蒸馏学生的最后四个视觉块中仅参数化注意力输出投影 O，保留原多头注意力算子，采用 rank2、alpha/sqrt(rank) 缩放及原 KD 监督。它不是完整 CLIP-LoRA 少样本分类实验，不同时更新全部 Q/K/V/O 或文本编码器。本仓库未将 CLIP-LoRA 参考项目整体复制进来，也未验证永久合并权重或零额外推理时延。[LARP 实现](LARP_IMPLEMENTATION.md)、[LARP 实验发现](LARP_FINDINGS.md)

## 4. 引用与致谢

感谢 PromptKD、MMA 和 CLIP-LoRA 的作者公开论文与研究代码，为本项目提供可核查的基础框架和设计参考。感谢 OpenAI CLIP、Dassl.pytorch，以及原 PromptKD 所依赖的 PromptSRC、MaPLe、CoOp/Co-CoOp 等项目的开放贡献。第三方成果、原作者署名和许可归属保持不变；本项目的新增结构、训练规则及实验结论另按实际实现和记录说明。

We thank the authors of PromptKD, MMA, and CLIP-LoRA for making their papers and research code available. We also acknowledge OpenAI CLIP, Dassl.pytorch, and the upstream projects acknowledged by PromptKD. HARP and LARP are independent adaptations within the prompt-distillation framework; this repository does not claim authorship of the upstream methods or their results.

引用上述基础或参考工作时，可使用仓库根目录的 [references.bib](../references.bib)。这些条目不替代自有研究的发表信息，也不虚构本项目已有会议论文。论文正文的全篇引用编号由论文自身维护；此处 BibTeX 键不改动工作区论文的现行编号。

本页于 2026-10-10 核对 CVF 正式著录、作者仓库和本地参考快照；核对当前网页不代表本项目冻结源码等于上游当前 main。具体源码身份以两路线 provenance 为准。来源声明不是新的许可授权，分发边界见[第三方说明](../THIRD_PARTY_NOTICES.md)。
