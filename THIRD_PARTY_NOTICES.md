# 第三方来源与修改说明

本仓库是研究代码整理，不主张所有代码均为原创。原源码被隔离保存，科学函数未因本次打包而修改；本次新增的是目录组织、路径化配置、便携运行/核验工具、原理与操作说明及来源清单。实际研究修改相对原框架涉及HARP/LARP适配模块、优化/渐升策略和审计运行器，具体以两路线源码及SHA为准。

| 组件/参考 | 来源 | 随附许可 |
| --- | --- | --- |
| PromptKD | https://github.com/zhengli97/PromptKD | [Apache-2.0](licenses/PromptKD-Apache-2.0.txt) |
| Dassl.pytorch | https://github.com/KaiyangZhou/Dassl.pytorch | [MIT](licenses/Dassl-MIT.txt) |
| OpenAI CLIP | https://github.com/openai/CLIP | [MIT](licenses/OpenAI-CLIP-MIT.txt) |
| MMA的适配/梯度缩放设计参考 | 本地MMA参考仓库及HARP源码注释 | [MIT原文](licenses/MMA-MIT.txt) |

OpenAI CLIP许可文本于2026-10-08从其[官方LICENSE](https://github.com/openai/CLIP/blob/main/LICENSE)核对补入；不因此声称本项目CLIP源码等于当日上游main。各实际版本由来源清单定位。

保留PromptKD原有致谢：代码建立在[PromptSRC](https://github.com/muzairkhattak/PromptSRC)、[MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning)、[CoOp/Co-CoOp](https://github.com/KaiyangZhou/CoOp)工作之上。相关原文件中的归属与许可声明保留，不改称本项目提出。

[CLIP-LoRA](https://github.com/MaxZanella/CLIP-LoRA)是低秩设计参考，其参考仓库自身使用AGPLv3；本次未将该参考仓库复制入发布包。可见最终LARP源码采用PyTorch参数化保留原MHA，未发现直接vendor/导入CLIP-LoRA的相关模块。这是源码范围说明，不是全量版权认定。后续若复制其实现，需要另行核查分发要求，不能套用PromptKD许可。

HARP的缩放函数标注MMA设计关联，随附本地MMA的完整许可，不将短函数改名视为消除原归属。数据、预训练权重和第三方Python包的条款分别适用；未纳入包的资产不受本仓库自有说明替代。

自有新增部分尚未指定开放许可，见[LICENSE.md](LICENSE.md)。Git托管不构成对自有部分的新开放许可，也不替代第三方条款。
