# 已有实验结果查询

想先了解“提升了什么、哪些任务退化、为什么做结构消融”，请读[实验发现与结论](../docs/RESULTS_AND_LIMITATIONS.md)；精确全十集表、差值图和可重算摘要见[主实验汇总表](findings/main_comparison.md)、[差值图](findings/main_hm_deltas.svg)、[summary.json](findings/summary.json)。这些是下列原JSON的派生视图，不是新增训练结果。

这里是既有完成记录的轻量副本，不是本次整理重新训练产生的分数。数值、记录条数、原始/选用标记及跨机信息保持不变；仅将原个人电脑目录和服务器工作目录前缀改为 `workspace://` 与 `server-workspace://`。两种前缀是溯源标识，不是可打开的网络地址。

| 路线 | 文件 | 范围 |
| --- | --- | --- |
| HARP | [complete_results.json](harp/complete_results.json) | 180条选定结果：R0/R1/主方案各30，加三种结构消融90；含3000轮轨迹及原/选用记录字段 |
| LARP | [flat_results.json](larp/flat_results.json) | R1、O4-r1/r2/r4、O3-r2，共150角色 |
| LARP | [analysis.json](larp/analysis.json) | 原有三seed统计与配对差；本轮未重新定义统计 |
| LARP | [epochs_index.json](larp/epochs_index.json) | 以run_id查询3000轮真实Base/KD；没有模拟逐轮Novel |
| LARP | [r0_external_reference.json](larp/r0_external_reference.json) | 独立R0外部参照，不混作同机R1 |
| LARP | [main_and_replay_summary.json](larp/main_and_replay_summary.json) | 原主实验和Pets诊断复放的来源；复放不计新seed |

查询键：HARP 使用 `records` 中的 `dataset + seed + method`；LARP 使用 `dataset + seed + variant` 或 `run_id`。不同来源中的角色名不必相同，不能仅凭末级目录 `o4` 推断层数和rank。

HARP 的 `source_policy`、`revision_selections`、`original_eurosat_seed2`、`original_main_macro` 等字段保留既有记录选用和敏感性信息。LARP 保留负结果与跨机参照字段。不能删除这些来源差异后，将当前主表视为全部首次运行或严格同机控制。

原始来源SHA、发布副本SHA及仅路径转换的说明见 [publication_sources.json](../metadata/publication_sources.json)。原始逐batch日志、所有节点取证、模型检查点并未全部放入本仓库；JSON中的源指针只用于回查历史资料，不宣称可以仅凭此包重新验证全部训练现场。原工作区的完整图表/论文材料仍保留，本仓库没有覆盖它们。
