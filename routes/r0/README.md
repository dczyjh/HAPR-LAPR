# R0：原始提示蒸馏基线

本工程用于产生原框架方式训练的基线参照，不直接采用网站分数，也不替代两条研究路线的匹配R1。对照分工见[用途总览](../../docs/PROJECT_MAP.md)。

R0通过 `routes/larp/archive/official_reference/` 的原官方源码运行，十集完整配置来自实际基线resolved快照；seed1/2/3可从头执行。`templates.json`仅将外部数据/输出路径模板化，`provenance.json`记录原配置来源和转换，不以HARP/LARP默认配置替代官方基线。

```bash
python reproduce.py run --route r0 --dataset dtd --seed 1 \
  --data-root /data/datasets --clip-root /data/pretrained/clip \
  --teacher-root /data/pretrained/teacher_model --output-root /runs/r0_dtd_s1
```

无HARP/LoRA，无新增组、投影器第2轮专用渐升；保留官方单组SGD与原学习率调度。R1则是研究路线的匹配控制，二者不能合并成一个基线。每项20轮、按Base选best，第一次独立重载同一best评Base/Novel，额外保存核验证据不改变训练更新。

官方代码使用相对的 `clip/*.pt` 和 `teacher_model/...`；adapter在该次运行目录建立独立源码视图及外部权重只读引用，原发布源码不改。原框架的所有依赖（含wilds）需先按版本锁准备；本地完整导入因缺wilds未验证，真实defaults/extend_cfg、30配置合并与单组SGD透明性已做CPU检查。详见[运行指南](../../docs/RUN_REPRODUCTION.md)与[验证范围](../../docs/VALIDATION.md)。
