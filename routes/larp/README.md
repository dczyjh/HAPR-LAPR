# LARP O4/rank2 复现路线

本工程用于复现LARP主实验、匹配R1及三种rank/层数消融，检验O投影低秩适配的作用，不与HARP串联。完整工程分工见[用途总览](../../docs/PROJECT_MAP.md)。

主方法固定为视觉编码器后四层注意力输出投影O（零基层8–11）、rank2，新增12,288参数。R1关闭LoRA；O4/rank1、O4/rank4、O3/rank2（层9–11）为消融。HARP在另一独立route，不把两者叠加。

341份最终科学源、官方参考和辅助源的SHA保存在`provenance/`；完整实际模板在`protocol/`。`portable.py`通过原科学实现构建模型/读取器与普通SGD；根`runtime/execution.py`自行建立R1配对、新初态、四步门槛、20轮和首次独立重载，不要求旧服务器日志。

```bash
python routes/larp/cli.py check
python -B -m unittest discover -s routes/larp/tests -p 'test_*.py' -v
python reproduce.py run --route larp --dataset dtd --seed 1 \
  --data-root /data/datasets --clip-root /data/pretrained/clip \
  --teacher-root /data/pretrained/teacher_model --output-root /runs/larp_dtd_s1
```

默认十集/三seed/R1+主方法60次；`--all-variants`五角色150次，`--variant`可重复。`cli.py train`转入同一入口；`plan/check`只读，不把通过离线检查当作真实训练通过。完整准备见[运行指南](../../docs/RUN_REPRODUCTION.md)。

## 固定科学规则

| 项目 | 固定设置 |
| --- | --- |
| 主结构 | 后四层O/rank2；不改为V/Q/K |
| 参数化 | alpha=1、alpha/sqrt(rank)、Kaiming A、全零B、无dropout |
| 优化 | 普通FP16 SGD及动量；A/B公共LR×0.1，从第一batch更新 |
| 共同投影器 | 第2轮按batch从1e-5升至0.005，不重置动量 |
| 损失 | 原KD唯一目标、T=1、KL sum除以学生logits元素数 |
| KD/轮数 | DTD/Aircraft/Flowers200，其他1000；每项20轮 |
| 数据/精度 | 十集三seed，不含ImageNet；FP16，batch8/100，workers8 |
| 后端 | benchmark=True、deterministic=False、Flash/efficient/Math=True |
| 选点 | Base-test严格提高才存best，同值取较早轮；同checkpoint评Base/Novel |

无FP32 master、额外KL、AB16、HARP停更或其他历史候选。O3/rank2共享物理层A/B按新O4/rank2锚精确映射；rank1/4自然初始化，不能从rank2强行切片。rank变化也改变缩放，层数变化也改变参数量，不冒称纯因果独立消融。

`archive`中的worker仅用于溯源，不直接执行。便携入口不改原源码、原损失或原重载硬限；硬失败保留首份证据并停队列，无自动重试。当前离线/CPU测试不是完整CUDA CLIP验收，真实执行需逐项通过检查。权重/图片不随Git提供，BPE词表是已保留的运行资源，来源许可随源保留。
