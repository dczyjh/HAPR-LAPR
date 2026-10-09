# 获取与准备实验物料

本说明将公开来源、实际文件指纹和读取目录连起来。当前可确认两个CLIP文件的官方固定直链；十教师的官方分享入口可查，但本地历史审计没有证明它们与目前分享文件逐字节相同。数据有作者维护的固定版本来源，取得后必须核对本项目登记的划分和读取身份。**不能把已有SHA或本页列出链接写成“全部外部物料已经在新机器获取并通过验收”。** 本次只查网页与既有本地记录，没有下载图片、大权重或重训教师。

## 1. 先确定三个根目录

建议在仓库之外新建独立物料目录。下例中的路径由使用者替换；两个方法读取同一套外部图片和权重，但分别使用自己的模型、配置及运行后端。

```text
/path/to/materials/
  data/                         # 传入 --data-root / DATASET.ROOT
  pretrained/
    clip/                       # 路线入口的 --clip-root
      ViT-B-16.pt
      ViT-L-14.pt
    teacher_model/              # 路线入口的 --teacher-root
      <DatasetClass>/VLPromptLearner/model-best.pth.tar
```

`tools/check_assets.py --pretrained-root` 接受上面的 `pretrained`；路线入口分别接受其 `clip` 与 `teacher_model` 子目录。BPE词表已经随源码保留，无需重新生成。全部必需权重的大小和SHA在 [external_assets.json](../metadata/external_assets.json)，十教师加两个CLIP共15,248,687,896字节，另需图片和后续运行空间。

## 2. 两个CLIP主干：固定官方文件

链接与SHA同时见[OpenAI CLIP源代码](https://github.com/openai/CLIP/blob/main/clip/clip.py)，并由本地原实验资产清单再次核对。工具默认只打印计划；明确给出 `--download` 才联网。

```bash
python tools/prepare_assets.py --group clip --pretrained-root /path/to/materials/pretrained
python tools/prepare_assets.py --group clip --pretrained-root /path/to/materials/pretrained --download
```

下载流只写新 `.partial` 文件，按登记大小和SHA通过后发布到目标位置。已有同SHA文件复用，已有异字节文件拒绝覆盖；失败保留 `.partial` 供检查，不自动重试或删除旧文件。工具只用Python标准库，不导入模型、不安装包。Windows或特殊文件系统若不支持最后的同目录硬链接发布，会保留已下载部分并报错，不降级为覆盖写入。

| 文件 | 官方固定下载 | 精确字节数 |
| --- | --- | ---: |
| ViT-B-16.pt | [OpenAI文件](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) | 350837078 |
| ViT-L-14.pt | [OpenAI文件](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) | 932768134 |

## 3. 十教师：从作者分享取得，再逐项校验

[PromptKD官方说明](https://github.com/zhengli97/PromptKD#preliminary)给出[百度网盘](https://pan.baidu.com/s/1KNJ1mhNKoxdSli4ZldeZUg?pwd=mjf4)、[TeraBox](https://terabox.com/s/1X4mxJtSaR8W2lrK5bsrCkg)和[Google Drive](https://drive.google.com/drive/folders/1OdQ9WauZmYAzVSUTTw7tIKKChyECIS5B?usp=sharing)；官方特别说明Google Drive只含部分教师。百度链接中的 `mjf4` 是作者公开的分享提取码，不是本项目服务器凭据。

用这些入口取得base-to-novel教师包，解压到独立目录，并将每个数据集的 `VLPromptLearner/model-best.pth.tar` 放入上述布局。保持字节原样：不要通过 `torch.load/torch.save` 转存，不用自己重新训练的同名文件替代，不混入ImageNet跨数据集教师。

| 任务 | 必需教师目录类名 |
| --- | --- |
| Caltech101 | Caltech101 |
| DTD | DescribableTextures |
| EuroSAT | EuroSAT |
| Aircraft | FGVCAircraft |
| Food101 | Food101 |
| Flowers102 | OxfordFlowers |
| Pets | OxfordPets |
| SUN397 | SUN397 |
| StanfordCars | StanfordCars |
| UCF101 | UCF101 |

```bash
python tools/prepare_assets.py --group teachers --pretrained-root /path/to/materials/pretrained
```

该命令列出十个目标、大小、SHA和官方入口，不执行教师下载。当前未确认十份文件各自稳定的公共直链，因此工具不猜测网盘下载接口，也不绕过登录或确认页面。历史十教师经过结构检查及实际表现一致性检查，但当时记录明确未与作者公开SHA逐字节比较；本项目SHA是**所用实验文件的身份约束**。若公开分享版本不匹配，保留下载文件与差异记录，需向物料持有者取得这十份确切字节或由作者确认对应版本。分数接近、张量形状相同、同名文件均不能代替SHA一致。

## 4. 十数据集及固定划分

作者的[数据准备说明](https://github.com/zhengli97/PromptKD/blob/main/docs/DATASETS.md)指向[其Hugging Face物料仓库](https://huggingface.co/zhengli97/prompt_learning_dataset/tree/main)。本项目已用过的固定提交为 `ceb3112337fb6e1bf71534e355ca5569b63f19a6`。下面链接固定此提交，不跟随 `main`；2026-10-08查看作者文件列表确认了这些包名，但本轮没有重新下载全部包验收。查看固定版本页面失败也不能据此假定大文件下载一定可用。

在仓库之外下载、解压对应ZIP，再把图片目录与固定元数据放到表中位置。可用符号链接建立读取视图，无需复制所有图片。ZIP内若有 `split_fewshot`、`.pkl`、`.pickle`，不把它们带入读取根；本实验为 `NUM_SHOTS=0`，不需要作者机器的缓存。不要运行下载包中的代码。原始数据自身的说明/条款留在下载归档中。

| 读取根下目录 | 作者固定ZIP | 必需图像目录 | 固定元数据 |
| --- | --- | --- | --- |
| caltech-101 | [caltech-101.zip](https://huggingface.co/zhengli97/prompt_learning_dataset/resolve/ceb3112337fb6e1bf71534e355ca5569b63f19a6/caltech-101.zip?download=true) | 101_ObjectCategories | split_zhou_Caltech101.json |
| dtd | [dtd.zip](https://huggingface.co/zhengli97/prompt_learning_dataset/resolve/ceb3112337fb6e1bf71534e355ca5569b63f19a6/dtd.zip?download=true) | images | split_zhou_DescribableTextures.json |
| eurosat | [eurosat.zip](https://huggingface.co/zhengli97/prompt_learning_dataset/resolve/ceb3112337fb6e1bf71534e355ca5569b63f19a6/eurosat.zip?download=true) | 2750 | split_zhou_EuroSAT.json |
| food-101 | [food101.zip](https://huggingface.co/zhengli97/prompt_learning_dataset/resolve/ceb3112337fb6e1bf71534e355ca5569b63f19a6/food101.zip?download=true) | images | split_zhou_Food101.json |
| oxford_flowers | [oxford_flowers.zip](https://huggingface.co/zhengli97/prompt_learning_dataset/resolve/ceb3112337fb6e1bf71534e355ca5569b63f19a6/oxford_flowers.zip?download=true) | jpg | split_zhou_OxfordFlowers.json |
| oxford_pets | [oxford_pets.zip](https://huggingface.co/zhengli97/prompt_learning_dataset/resolve/ceb3112337fb6e1bf71534e355ca5569b63f19a6/oxford_pets.zip?download=true) | images | split_zhou_OxfordPets.json |
| stanford_cars | [stanford_cars.zip](https://huggingface.co/zhengli97/prompt_learning_dataset/resolve/ceb3112337fb6e1bf71534e355ca5569b63f19a6/stanford_cars.zip?download=true) | cars_train、cars_test | split_zhou_StanfordCars.json |
| sun397 | [sun397.zip](https://huggingface.co/zhengli97/prompt_learning_dataset/resolve/ceb3112337fb6e1bf71534e355ca5569b63f19a6/sun397.zip?download=true) | SUN397 | split_zhou_SUN397.json |
| ucf101 | [ucf101.zip](https://huggingface.co/zhengli97/prompt_learning_dataset/resolve/ceb3112337fb6e1bf71534e355ca5569b63f19a6/ucf101.zip?download=true) | UCF-101-midframes | split_zhou_UCF101.json |
| fgvc_aircraft | [fgvc_aircraft.zip](https://huggingface.co/zhengli97/prompt_learning_dataset/resolve/ceb3112337fb6e1bf71534e355ca5569b63f19a6/fgvc_aircraft.zip?download=true) | images | variants.txt、images_variant_train.txt、images_variant_val.txt、images_variant_test.txt |

例如作者包若解压得到 `caltech-101/101_ObjectCategories`，将它接到 `data/caltech-101/101_ObjectCategories`；不要再多套一层 `caltech-101`。Food包名 `food101.zip` 不等于读取器目录名，目标目录必须为 `food-101`。Aircraft官方原始包中是 `fgvc-aircraft-2013b/data`，应把该层里的 `images` 和TXT放入 `data/fgvc_aircraft`。UCF使用作者发布的静态中间帧，不能直接换成AVI或自行选择另一帧。

只有九个Zhou JSON；Aircraft直接读上述四个TXT，无需第十个JSON。没有固定JSON时，某些旧读取器会随机新建划分；应先准备并核验，不能让这个后备行为静默发生。下表列出官方文档的独立划分来源，适用于已有原始图片但缺划分的情况。下载结果仍以本项目SHA为准。

| 划分 | 官方文档提供的来源 |
| --- | --- |
| Caltech101 | [Google Drive](https://drive.google.com/file/d/1hyarUivQE36mY6jSomru6Fjd-JzwcCzN/view?usp=sharing) |
| DTD | [Google Drive](https://drive.google.com/file/d/1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x/view?usp=sharing) |
| EuroSAT | [Google Drive](https://drive.google.com/file/d/1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o/view?usp=sharing) |
| Food101 | [Google Drive](https://drive.google.com/file/d/1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl/view?usp=sharing) |
| Flowers102 | [Google Drive](https://drive.google.com/file/d/1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT/view?usp=sharing) |
| OxfordPets | [Google Drive](https://drive.google.com/file/d/1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs/view?usp=sharing) |
| StanfordCars | [Google Drive](https://drive.google.com/file/d/1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT/view?usp=sharing) |
| SUN397 | [Google Drive](https://drive.google.com/file/d/1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq/view?usp=sharing) |
| UCF101 | [Google Drive](https://drive.google.com/file/d/1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y/view?usp=sharing) |

原数据提供方的下载步骤仍以[官方逐数据集说明](https://github.com/zhengli97/PromptKD/blob/main/docs/DATASETS.md)为入口：其列有Caltech、Oxford Pets/Flowers/DTD/Aircraft、ETH Food101、Stanford Cars、Princeton SUN、EuroSAT和UCF中间帧来源。作者明确提醒部分原始链接可能失效，所以这里优先列作者整理包；不把整理仓库网页上的单一许可标签当作各原始数据集许可的替代。

### 读取器所需的空目录

在上述十个数据集读取目录中各自建立空的 `split_fewshot/`。最终配置是 NUM_SHOTS=0，不需要下载包的few-shot pickle缓存；HARP便携检查要求该目录已准备，避免旧读取器在来源目录自动创建文件。保留图像及固定JSON/TXT原字节，不生成新随机划分。

### 已有固定包证据与尚未核验的范围

本地历史记录对以下三个包已下载、逐包SHA及划分SHA核验；当时下载域名为 `hf-mirror.com`，源仓库与固定提交相同。表中登记可以核对以后从作者Hugging Face原站取得的同字节包，不代表本轮重新下载成功。

| 包 | 精确字节数 | 历史包SHA256 |
| --- | ---: | --- |
| stanford_cars.zip | 1959002685 | 50812812f33534ef2906b992762f96e8a5bb57b0eedc53800e91b7ed599abf8e |
| sun397.zip | 39182693483 | ce6dee6685c9f1d592c8297e193dde885aa9c50d008fd998a38ccc57a3dda722 |
| ucf101.zip | 140164538 | 2fe1bd40e64408d02ff173badf744da557033ae6c75dd4ea51519366551f3ae6 |

Flowers历史只按HTTP范围读取该提交ZIP内的划分，CRC通过且原JSON字节SHA为 `2f50c42fc1a17ca13c271cb4269b1347bdb43d2be672e34f74a4e4cbfbcca794`，与实验文件相同；这不是整个Flowers ZIP或全部图片的重下载证明。本地未找到可复制进当前发布包的九份split原字节，只有来源/指纹和部分核验回执，因此本次不制作替代JSON，也不声称已经随Git发布这些划分文件。

## 5. 对齐文件身份，再做读取器验收

```bash
python tools/prepare_assets.py --group all --data-root /path/to/materials/data --pretrained-root /path/to/materials/pretrained
python tools/check_assets.py --data-root /path/to/materials/data --pretrained-root /path/to/materials/pretrained
```

第一条只列来源/路径/期望值，无网络或写入；第二条只读校验12份权重和13份元数据。25项全部通过才能认为这些登记文件匹配。它没有验证全部图片、类别映射、解码、训练/测试互斥、实际运行环境或GPU数值；还应使用路线提供的读取器验收和工程门槛。计数目标可查LARP `protocol/datasets.json`及HARP实际配置/证据，不能靠文件总数代替读取身份。

所有新下载保留在独立目录，校验失败时不更改发布manifest的期望SHA，也不通过重新保存或生成元数据来“修复”差异。公开获取链目前仍有明确未闭合项：十教师的当前公开同字节版本、其余数据包与全部实际图片的字节身份，以及本轮未执行的大文件获取。这些应以成功下载和逐项验收回执补齐后，才可称完整物料已在新机器准备完成。

## 来源记录

网页入口核对日期：2026-10-08；网页可读不等于所有分享服务、大文件链路可用。历史依据是工作区的 `outputs/server_dataset_setup/evidence_20260927/v3_logs/asset_manifest.json`、同批 `datasets/_promptkd_downloads/{stanford_cars,sun397,ucf101}.verification.json`、`outputs/baseline_material_audit/evidence/flowers_official_split.json`及`原版PromptKD基线物料验收.md`。这些名称是历史来源标识，不声称全部原始报告都随发布包复制；当前文件身份以随包的 [external_assets.json](../metadata/external_assets.json) 为准。本页不新增第三方许可判断，参见[第三方说明](../THIRD_PARTY_NOTICES.md)。
