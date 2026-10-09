# 参数、结果查询与排错

本页对应[reproduce.py](../reproduce.py)与[公共运行器](../runtime/execution.py)，不提供跳过核验的开关。先读[运行指南](RUN_REPRODUCTION.md)和[工程用途](PROJECT_MAP.md)。

## 1. 参数表

| 命令/参数 | 含义及默认 |
| --- | --- |
| `plan` | 只打印计划，不建目录、不读图片、不启GPU |
| `run` | 实际建立新初态、逐项门槛、训练与首次独立重载 |
| `evaluate` | 完整训练后尚未执行过的首次独立评估；队列通常自动调用 |
| `--route` | 必选r0/harp/larp |
| `--dataset` | 可重复；默认十集 |
| `--seed` | 可重复；只允许1/2/3，默认三者全跑 |
| `--variant` | 可重复本route登记角色，与all-variants互斥 |
| `--all-variants` | HARP/LARP各五角色，R0只有一角色 |
| `--data-root` | 整理好的读取根，不是ZIP下载目录 |
| `--clip-root` | 直接包含两个CLIP pt文件的目录 |
| `--teacher-root` | 包含DatasetClass子目录的教师根，不是单个checkpoint |
| `--output-root` | 必须全新、不存在，且与仓库/物料根分开 |
| `--gpu` | 单个可见设备标识，默认0；不接受逗号多GPU列表 |
| `--registration` | evaluate使用原训练根的registration.json |

十集键：`dtd`、`oxford_pets`、`fgvc_aircraft`、`oxford_flowers`、`caltech101`、`stanford_cars`、`ucf101`、`eurosat`、`sun397`、`food101`。Aircraft不是`aircraft`，Pets不是`pets`。

默认HARP是R1+HARP，LARP是R1+O4/r2。只请求候选仍自动加入匹配R1；必要主方法锚只建立初态，不额外训练未请求主方法。计划中formal_runs/formal_epochs/engineering_steps是预算，不是完成数。

## 2. 最小运行例子

在仓库根、固定环境/真实物料准备好后执行：

```bash
python reproduce.py plan --route harp --dataset dtd --seed 1 \
  --data-root /data/datasets --clip-root /data/pretrained/clip \
  --teacher-root /data/pretrained/teacher_model --output-root /runs/harp_dtd_s1

python reproduce.py run --route harp --dataset dtd --seed 1 \
  --data-root /data/datasets --clip-root /data/pretrained/clip \
  --teacher-root /data/pretrained/teacher_model --output-root /runs/harp_dtd_s1
```

run会训练两个角色各20轮，不是仅四步检查。路径带空格时加引号；示例路径必须换成自己的真实物料路径，不能创建空目录冒充准备完成。完整十集三seed/消融见运行指南。

多集/多seed时重复参数：

```bash
python reproduce.py plan --route larp --dataset dtd --dataset oxford_pets \
  --seed 2 --seed 3 --variant o4-r2 --variant o3-r2 \
  --data-root /data/datasets --clip-root /data/pretrained/clip \
  --teacher-root /data/pretrained/teacher_model --output-root /runs/larp_subset_s23
```

这里加R1后是2集×2seed×3角色=12次/240轮；只有plan改run才执行。不要将计划打印成功当作已经训练。

## 3. 进展与结果

查询输出根status.json：running包含current动作；completed是登记动作全完成；failed保留错误并停止后继。日志在`console/<dataset>_seed<seed>_<variant>_<action>.log`。DTD例子的正式成绩为：

```text
/runs/harp_dtd_s1/harp/dtd/seed_1/formal/r1/audited_metrics.json
/runs/harp_dtd_s1/harp/dtd/seed_1/formal/harp/audited_metrics.json
```

base/novel/hm是百分数；selected_epoch是选定best轮数；kd是KD系数，不是轮数；epochs固定20。checkpoint_sha256及best_last_sha256核文件，canonical_evaluation标明首次独立重载。只有status=passed进入正式汇总。

train_complete通过不等于独立重载已通过。metrics.json是原训练器native输出，新汇总使用正式audited_metrics，不挑native或多次重载高分。

```bash
python tools/summarize_run.py --run-root /runs/harp_dtd_s1
python tools/summarize_run.py --run-root /runs/r0_three_seed \
  --run-root /runs/harp_main_three_seed --run-root /runs/larp_main_three_seed \
  --output /runs/new_summary.json
```

默认Markdown表，output额外保存新JSON，拒绝覆盖。缺seed只显示单seed/缺项，无三seed统计；缺参考不生成增幅。重复键报错，不自动选择较高运行。

## 4. 常见问题

| 提示/现场 | 正确处理 |
| --- | --- |
| 缺参数/未知角色 | 查help及键名，不拼未知opts |
| 环境/显卡/CUDA不匹配 | 准备固定版本与RTX4090，不切CPU/AMP/新版torch放行 |
| ModuleNotFoundError如wilds | 按lock准备对应依赖，不删import或注入假模块 |
| 权重/元数据SHA不同、缺split | 保留文件，核来源/目录，取得匹配字节；不改期望SHA、不转存/随机生成split |
| 缺split_fewshot目录 | 运行前建立空目录，不复制不明pickle |
| 输出已存在 | 换新输出根，保留旧成绩/失败现场，不删目录循环重试 |
| GPU锁占用 | 等当前任务结束，不删除锁文件抢占 |
| 启动余盘不足40GiB/运行不足20GiB | 扩容或审阅明确可清理内容，不删除在用物料/检查点 |
| 源码/发布SHA变化 | 保留现场，核改动；修订审阅后新版本新目录，不覆盖原登记 |
| 初态/RNG/输入/公共LR不配对 | 查第一处错误及resolved/记录，不跳pair guard |
| loss/梯度/参数非有限或四步失败 | 保留failure.json及可取得的first_failure_state.pt，分析后另行登记，不自动调参 |
| 独立重载hard gate失败 | 保留首份指标与canonical/配对证据，不因分数高或改判少而放行，不择优复测 |
| 汇总缺seed/重复键/版本不符 | 对照登记范围补查，不混算不完整组或重复seed |

本项目没有自动续训/修参/跳项/重试。服务器进程在运行时本地电脑断网不影响，但服务器关机不等于进程还能继续；中断后不能声称自动无缝续训。不要用python -O或PYTHONOPTIMIZE关闭科学断言。

本地工程测试不是论文分数复现；R0完整官方import因本地缺wilds跳过，正式lock已登记。完整事实以[验证报告](VALIDATION.md)为准。
