# 外部物料和环境

本仓库不包含数据集图片、教师或CLIP大权重，也不自动下载。它们需要按原提供方许可单独准备。不要把图片数据集或所有历史输出整体提交到Git。

## 数据根目录

`DATASET.ROOT` 指向**整理后的读取根目录**，不是下载压缩包所在目录。各读取器使用以下固定布局：

| 相对目录 | 图像目录 | 固定划分文件 |
| --- | --- | --- |
| caltech-101 | 101_ObjectCategories | split_zhou_Caltech101.json |
| dtd | images | split_zhou_DescribableTextures.json |
| eurosat | 2750 | split_zhou_EuroSAT.json |
| food-101 | images | split_zhou_Food101.json |
| oxford_flowers | jpg | split_zhou_OxfordFlowers.json |
| oxford_pets | images | split_zhou_OxfordPets.json |
| stanford_cars | cars_train、cars_test | split_zhou_StanfordCars.json |
| sun397 | SUN397 | split_zhou_SUN397.json |
| ucf101 | UCF-101-midframes | split_zhou_UCF101.json |
| fgvc_aircraft | images | variants.txt、images_variant_train.txt、images_variant_val.txt、images_variant_test.txt |

九个固定JSON加Aircraft四个TXT是正确结构。不要缺文件后让读取器重新随机生成划分，不能拿另一个版本的划分代替。

## 权重根目录

独立准备的 `pretrained` 根目录下须有：

```text
clip/ViT-B-16.pt
clip/ViT-L-14.pt
teacher_model/<数据集类名>/VLPromptLearner/model-best.pth.tar
```

教师目录类名：Caltech101、DescribableTextures、EuroSAT、FGVCAircraft、Food101、OxfordFlowers、OxfordPets、SUN397、StanfordCars、UCF101。十个教师加两个CLIP文件共约15.25GB（十进制）。`clip/bpe_simple_vocab_16e6.txt.gz` 是小型词表，已随路线源码保留，不属于可随意省略的缓存。

[external_assets.json](../metadata/external_assets.json) 给12份权重与13份划分/类别文件的精确大小及SHA256，可用以下命令只读核验：

```bash
python tools/check_assets.py --data-root /path/to/data --pretrained-root /path/to/pretrained
```

此检查不解包、不联网、不生成划分、不载入模型；它**没有核对全部图片的内容及计数**。原完整物料清单的SHA也已登记，未来训练仍须做读取器级别和图片完整性验收。

## 真实环境与安装边界

历史科学依赖：Linux x86_64、Python 3.10.8、torch 2.0.1+cu118、torchvision 0.15.2+cu118、CUDA运行时11.8、cuDNN8700；NumPy1.24.4、Pillow9.5.0、SciPy1.10.1。这是实际记录，不是建议改用当前最新版。

两路线分别保留原环境记录和移除服务器路径后的安装规格。LARP历史`freeze --all`比HARP普通`freeze`多记录pip24.3.1、setuptools75.6.0、wheel0.45.1，不据此断言科学库不同。原freeze中的本地torch wheel URI和editable Dassl路径不能在新机器原样执行；应使用相应路线源码里的Dassl，并取得相同版本/来源的wheel。

本轮未重新安装Linux环境，锁文件不等于当前软件源下载可用性或新机器CUDA验收。Windows/macOS CPU离线检查通过也不能代替Linux GPU模型测试。不要因为显卡驱动显示支持更高CUDA就自动升级torch；驱动、软件、硬件及后端差异仍须登记。

尤其保留路由差异：HARP memory-efficient SDP=False；LARP=True。两路线的benchmark=True、deterministic=False及Flash/Math启用记录不是逐位确定性承诺。
