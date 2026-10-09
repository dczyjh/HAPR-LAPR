# 从头复现论文实验

先用[工程用途总览](PROJECT_MAP.md)确定R0、某条主路线或消融的用途；不需要执行每个源码目录。

## 1. 环境与物料

目标环境：Linux x86_64、Python 3.10.8、RTX4090、PyTorch2.0.1+cu118、torchvision0.15.2+cu118、CUDA运行时11.8、NumPy1.24.4、Pillow9.5.0。使用独立环境，不改变系统Python。历史完整版本保存在两路线`environment/`，不照搬服务器绝对路径、不升级科学依赖。

若机器已提供Python3.10.8，可在仓库根建立独立venv；下面python3.10必须实际为3.10.8，先核版本：

```bash
python3.10 --version
python3.10 -m venv .venv
source .venv/bin/activate
python --version
```

venv也可建在仓库外，.venv已被Git忽略。这不是安装Python本身的命令；缺3.10.8时先准备对应解释器，不拿系统默认版本代替。

PyTorch[官方旧版本说明](https://pytorch.org/get-started/previous-versions/#v201)提供2.0.1/0.15.2的cu118索引。在已激活的Python3.10.8独立环境内执行：

```bash
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r routes/harp/environment/requirements-linux-cu118.txt
```

两路线分别从自己冻结的Dassl目录导入，不需要全局editable安装两份同名包，也不用远端最新版。R0官方实现还需要lock中的`wilds==1.2.2`。本轮尚未在空白Linux重新安装，历史版本下载可用性未逐项验证；若软件源缺版本，取得匹配wheel/环境，不偷偷改锁。

按[物料说明](ASSET_PREPARATION.md)取得图像、九个固定JSON、Aircraft四个TXT、两个CLIP与十个教师。每个数据集目录先建立**空的**`split_fewshot/`，NUM_SHOTS=0不读取缓存，不能复制来源不明的pickle。保持原字节及布局。权重、图片和输出都放在仓库外；每次输出根必须全新。开始至少留40GiB，运行保留20GiB，完整矩阵实际需要更多空间。

```bash
python tools/check_assets.py --data-root /data/datasets --pretrained-root /data/pretrained
python project.py verify
python -B routes/harp/cli.py check --cpu-tests
python -B -m unittest discover -s routes/larp/tests -p 'test_*.py' -v
python -B -m unittest discover -s tests -p 'test_*.py' -v
```

离线检查不代表GPU/精度通过；`run`仍核实际环境、数据读取身份及每角色四步门槛。

## 2. 先查看计划

```bash
python reproduce.py plan --route harp --dataset dtd --seed 1 \
  --data-root /data/datasets --clip-root /data/pretrained/clip \
  --teacher-root /data/pretrained/teacher_model --output-root /runs/harp_dtd_s1
```

`plan`只打印JSON，不建目录、不读图、不启GPU。`--dataset`、`--seed`、`--variant`可重复。没给数据集/seed就默认十集/1、2、3；不能按结果挑seed。请求候选时自动加本route的R1；必要的主方法初始化锚只建立初态，不额外训练未请求的主方法。

## 3. R0、主实验及消融

以下`run`是**实际启动**。可以放入tmux，服务器进程启动后本地断网不影响；本项目不创建定时器。只暴露一个GPU，各route共享主机/用户排他锁。

```bash
# R0：十集三seed，30次/600轮
python reproduce.py run --route r0 --gpu 0 \
  --data-root /data/datasets --clip-root /data/pretrained/clip \
  --teacher-root /data/pretrained/teacher_model --output-root /runs/r0_three_seed

# HARP：R1+主方法，60次/1200轮
python reproduce.py run --route harp --gpu 0 \
  --data-root /data/datasets --clip-root /data/pretrained/clip \
  --teacher-root /data/pretrained/teacher_model --output-root /runs/harp_main_three_seed

# LARP：R1+主方法，60次/1200轮
python reproduce.py run --route larp --gpu 0 \
  --data-root /data/datasets --clip-root /data/pretrained/clip \
  --teacher-root /data/pretrained/teacher_model --output-root /runs/larp_main_three_seed
```

加`--all-variants`同时跑本route的R1、主方法及三个消融，150次/3000轮，不含ImageNet。若已另跑主实验，新的all-variants是一套新配对实验，不自动偷用旧控制。CLI不提供arbitrary opts、改KD/LR、跳gate、force或续训。

| route | 可选角色 |
| --- | --- |
| r0 | `r0` |
| harp | `r1`、`harp`、`mid4_d32`、`last4_d16`、`last4_d64` |
| larp | `r1`、`o4-r2`、`o4-r1`、`o4-r4`、`o3-r2` |

R0是官方原SGD；R1为关闭新增模块的匹配控制，保留共同投影器第2轮渐升。HARP保留fix2 VJP及新增组停更/双渐升；LARP普通FP16 SGD从首batch更新O。详见[实验协议](EXPERIMENT_PROTOCOL.md)，不能把R0/R1合并或删掉差别。

## 4. 顺序、输出与失败

每个数据集/seed先生成新R1与候选初态，再新进程逐角色四步检查；随后新进程训练R1并首次独立重载，通过后才训练/评估候选。所有正式增强输入、标签/路径和实际公共LR配对；从门槛更新过的模型继续训练被禁止。

```text
output-root/
  registration.json               # 固定计划、物料路径、发布SHA
  status.json                     # 当前动作/已完成/错误
  console/                        # 各fresh action日志
  route/dataset/seed_N/
    initial/variant/              # 新增参数初态、共同参数/RNG hash
    gate/variant/                 # 四步、首步对照、保存恢复
    formal/variant/
      resolved.json               # 实际完整配置
      VLPromptLearner/            # best及第20轮last
      batches.jsonl               # 全输入身份、实际LR与loss
      epochs.jsonl                # 20轮Base/KD
      train_complete.json
      audited_metrics.json        # 独立重载全通过才有
    evaluation/variant/           # 第一份独立成绩与hard gate证据
```

队列自动评估。独立`evaluate`只用于完整训练且**从未执行该评估动作**的目录，使用相同登记环境与锁，不是挑分重测：

```bash
python reproduce.py evaluate --registration /runs/harp_dtd_s1/registration.json \
  --dataset dtd --seed 1 --variant harp
```

硬失败保留第一份现场及原目录，停止后继，无自动重试/放宽阈值/续训。修复须审阅后用新目录登记；现有输出和历史实验不得覆盖。

## 5. 汇总

```bash
python tools/summarize_run.py --run-root /runs/r0_three_seed \
  --run-root /runs/harp_main_three_seed --run-root /runs/larp_main_three_seed \
  --output /runs/summary.json
```

只汇总passed正式记录，输出每seed的Base/Novel/HM/best epoch及对同seed R1/R0的百分点增幅；三seed均值和样本标准差只有1/2/3全齐才计算，缺项明确列出。拒绝重复键、不同发布版本混算和覆盖JSON，不择优或把同seed重复当额外seed。历史`results/`是原论文证据，新输出独立保存；本轮没有GPU重現其分数。
