# HARP fix2 复现路线

本工程用于复现HARP主实验、匹配R1及三种位置/宽度消融，检验非线性适配模块的作用，不负责LARP训练。完整工程分工见[用途总览](../../docs/PROJECT_MAP.md)。

主方法固定为视觉编码器后四层（零基层8–11）、宽32、scale=0.001，新增199,808参数。消融是中四层宽32、后四层宽16/64；R1 关闭新增模块但保留共同投影器渐升。两路线独立，不与 LARP 堆叠。

实际源在 `vendor/project/`，150份原配置在 `archive/recorded_configs/`，便携模板仅改四项外部路径。`portable.py` 构建原模型/读取器，`runtime/frozen_install.py` 保留原 installer AST；根 `runtime/execution.py` 接入全新初始化、四步门槛、20轮和首次独立重载，不依赖历史日志。

## 执行

```bash
python routes/harp/cli.py check --cpu-tests
python reproduce.py run --route harp --dataset dtd --seed 1 \
  --data-root /data/datasets --clip-root /data/pretrained/clip \
  --teacher-root /data/pretrained/teacher_model --output-root /runs/harp_dtd_s1
```

`run` 不指定数据集/seed即十集/1、2、3，默认R1+HARP。`--all-variants`为五角色150次；可重复 `--variant`。`cli.py train`转入同一入口，旧`plan`仍可检查单份完整配置，`preflight`仅核权重SHA并不代替GPU门槛。完整环境、资产与命令见[运行指南](../../docs/RUN_REPRODUCTION.md)。

## 不可改写的科学规则

- 原KD唯一损失；DTD/Aircraft/Flowers系数200，其他1000。20轮、FP16、batch8/100、workers8。
- 原SGD、LR0.005、momentum0.9、weight decay0.0005、余弦及首轮constant warmup。
- 原fix2的 `InputScaledLinear`/`ScaledResidualLinear` 自定义VJP，不用普通残差缩放导数替代。
- 新增组首轮grad=None，无更新/动量；第2轮按batch从1e-6升至0.0005，其后公共LR×0.1。
- 共同投影器第2轮从1e-5升至0.005，不重置动量。
- benchmark=True，Flash/Math=True，efficient=False，不强制确定性。
- Base/Novel无标签训练图均用于蒸馏；按Base-test严格最高选同一best，首次独立重载评两侧，不按Novel或重复成绩挑选。

中四层自然初态按主方法8–11→4–7映射核对，不重采样或复制共同权重；宽度消融保持对应形状的自然初始化。每次正式训练使用新进程/新无检查点目录，门槛更新后的模型不会接着训练。

`historical/`仅追溯原始现场，不可直接执行。冻结源码、原配置及旧报告不改；本轮CPU测试不是完整FP16 CLIP验收，真实执行仍须通过各项门槛。许可证随原源保留。
