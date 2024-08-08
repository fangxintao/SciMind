# ChatGLM2

## 模型描述

ChatGLM**2**-6B 是开源中英双语对话模型 [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B) 的第二代版本，在保留了初代模型对话流畅、部署门槛较低等众多优秀特性的基础之上，ChatGLM**2**-6B引入了新特征：**更强大的性能**、**更长的上下文**、**更高效的推理**、**更开放的协议**。

```text
@article{zeng2022glm,
  title={Glm-130b: An open bilingual pre-trained model},
  author={Zeng, Aohan and Liu, Xiao and Du, Zhengxiao and Wang, Zihan and Lai, Hanyu and Ding, Ming and Yang, Zhuoyi and Xu, Yifan and Zheng, Wendi and Xia, Xiao and others},
  journal={arXiv preprint arXiv:2210.02414},
  year={2022}
}
```

## 模型性能

- 基于910A

**GLM2_6b**:

| config                                                   | task            | Datasets | metric                                  | phase               | score                                                                              | performance                                    |
|----------------------------------------------------------|-----------------|----------|-----------------------------------------|---------------------|------------------------------------------------------------------------------------|------------------------------------------------|
| [glm2_6b](../../configs/glm2/run_glm2_6b.yaml)           | text_generation | ADGEN    | -                                       | [finetune](#全量微调)   | -                                                                                  | 815.2059134 tokens/s/p                         |
| [glm2_6b_lora](../../configs/glm2/run_glm2_6b_lora.yaml) | text_generation | ADGEN    | -                                       | [finetune](#lora微调) | -                                                                                  | 3243.697479 tokens/s/p                         |
| [glm2_6b](../../configs/glm2/run_glm2_6b.yaml)           | text_generation | ADGEN    | rouge-1<br>rouge-2<br>rouge-l<br>bleu-4 | [eval](#评测)         | 30.784298224299064<br>7.073415046728972<br>24.773958598130843<br>7.466147757009345 | -                                              |
| [glm2_6b_lora](../../configs/glm2/run_glm2_6b_lora.yaml) | text_generation | ADGEN    | rouge-1<br>rouge-2<br>rouge-l<br>bleu-4 | [eval](#评测)         | 31.05639289719626<br>7.1753861682243<br>24.229674859813084<br>7.229435140186916    | -                                              |
| [glm2_6b](../../configs/glm2/run_glm2_6b.yaml)           | text_generation | -        | -                                       | [predict](#推理)      | -                                                                                  | 32.08 tokens/s (use_past=True, seq_length=512) |

## 仓库介绍

`chatGLM2-6B` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：`mindformers/models/glm2`

    ```text
    glm2
        ├── __init__.py
        ├── glm2.py                  # 模型实现
        ├── glm2_config.py           # 模型配置项
        ├── glm2_modules.py          # 模组实现
        ├── glm2_tokenizer.py        # tokenizer
        └── glm2_transformer.py      # transformer层实现
    ```

2. 模型配置：`configs/glm2`

    ```bash
    glm2
        ├── export_glm2_6b.yaml                # 导出mindir配置
        ├── run_glm2_6b_finetune_2k_910b.yaml  # 910b最佳性能全量微调启动配置
        ├── run_glm2_6b_finetune_2k.yaml       # 910a最佳性能全量微调启动配置
        ├── run_glm2_6b_finetune_910b.yaml     # 910b ADGEN全量微调启动配置
        ├── run_glm2_6b_finetune.yaml          # 910a ADGEN全量微调启动配置
        ├── run_glm2_6b_finetune_eval.yaml     # 全量微调评估配置
        ├── run_glm2_6b_lora_2k_910b.yaml      # 910b最佳性能lora微调启动配置
        ├── run_glm2_6b_lora_2k.yaml           # 910a最佳性能lora微调启动配置
        ├── run_glm2_6b_lora_910b.yaml         # 910a ADGEN lora微调启动配置
        ├── run_glm2_6b_lora.yaml              # 910a ADGEN lora微调启动配置
        └── run_glm2_6b_lora_eval.yaml         # lora微调评估配置
    ```

## 前期准备

### 生成RANK_TABLE_FILE

运行mindformers/tools/hccl_tools.py生成RANK_TABLE_FILE的json文件

```bash
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"
```

**注：若使用ModelArts的notebook环境，可从 `/user/config/jobstart_hccl.json` 路径下直接获取rank table，无需手动生成**

RANK_TABLE_FILE 单机8卡参考样例:

```json
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "xx.xx.xx.xx",
            "device": [
                {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0"},
                {"device_id": "1","device_ip": "192.2.27.6","rank_id": "1"},
                {"device_id": "2","device_ip": "192.3.27.6","rank_id": "2"},
                {"device_id": "3","device_ip": "192.4.27.6","rank_id": "3"},
                {"device_id": "4","device_ip": "192.1.27.7","rank_id": "4"},
                {"device_id": "5","device_ip": "192.2.27.7","rank_id": "5"},
                {"device_id": "6","device_ip": "192.3.27.7","rank_id": "6"},
                {"device_id": "7","device_ip": "192.4.27.7","rank_id": "7"}],
             "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

### 多机RANK_TABLE_FILE合并

- step 1. 首先根据上章节内容，在每个机器上生成各自的`RANK_TABLE_FILE`文件，然后将不同机器上生成的`RANK_TABLE_FILE`文件全部拷贝到同一台机器上。

```bash
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)" --server_ip xx.xx.xx.xx
```

**注：需要根据机器的ip地址指定 --server_ip，避免由于不同机器server_ip不同，导致多节点间通信失败。**

- step 2. 运行mindformers/tools/merge_hccl.py将不同机器上生成的`RANK_TABLE_FILE`文件合并

```bash
# 运行如下命令，合并每个机器上的RANK_TABLE_FILE的json文件。
python ./mindformers/tools/merge_hccl.py hccl*.json
```

- step 3. 将合并后的`RANK_TABLE_FILE`文件拷贝到所有机器中，保证不同机器上的`RANK_TABLE_FILE`相同。

RANK_TABLE_FILE 双机16卡参考样例:

```json
{
    "version": "1.0",
    "server_count": "2",
    "server_list": [
        {
            "server_id": "xx.xx.xx.xx",
            "device": [
                {
                    "device_id": "0", "device_ip": "192.168.0.0", "rank_id": "0"
                },
                {
                    "device_id": "1", "device_ip": "192.168.1.0", "rank_id": "1"
                },
                {
                    "device_id": "2", "device_ip": "192.168.2.0", "rank_id": "2"
                },
                {
                    "device_id": "3", "device_ip": "192.168.3.0", "rank_id": "3"
                },
                {
                    "device_id": "4", "device_ip": "192.168.0.1", "rank_id": "4"
                },
                {
                    "device_id": "5", "device_ip": "192.168.1.1", "rank_id": "5"
                },
                {
                    "device_id": "6", "device_ip": "192.168.2.1", "rank_id": "6"
                },
                {
                    "device_id": "7", "device_ip": "192.168.3.1", "rank_id": "7"
                }
            ],
            "host_nic_ip": "reserve"
        },
        {
            "server_id": "xx.xx.xx.xx",
            "device": [
                {
                    "device_id": "0", "device_ip": "192.168.0.1", "rank_id": "8"
                },
                {
                    "device_id": "1", "device_ip": "192.168.1.1", "rank_id": "9"
                },
                {
                    "device_id": "2", "device_ip": "192.168.2.1", "rank_id": "10"
                },
                {
                    "device_id": "3", "device_ip": "192.168.3.1", "rank_id": "11"
                },
                {
                    "device_id": "4", "device_ip": "192.168.0.2", "rank_id": "12"
                },
                {
                    "device_id": "5", "device_ip": "192.168.1.2", "rank_id": "13"
                },
                {
                    "device_id": "6", "device_ip": "192.168.2.2", "rank_id": "14"
                },
                {
                    "device_id": "7", "device_ip": "192.168.3.2", "rank_id": "15"
                }
            ],
            "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

### 模型权重下载与转换

开发者可以下载获取官方权重后，通过下面提供的**权重转换脚本**，将官方权重转换为MindSpore权重；或直接使用MindFormers提供的**已转换权重**

1. 使用官方权重进行转换

   克隆glm2-6b代码仓，下载分布式的模型文件。

   ```shell
   git lfs install
   git clone https://huggingface.co/THUDM/chatglm2-6b
   ```

   执行 python 脚本，合并模型权重。

   ```python
   from transformers import AutoTokenizer, AutoModel
   import torch

   tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
   model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)

   with open("pt_model_arch.txt", "w") as fp:
       print(model, file=fp, flush=True)
   with open("pt_ckpt.txt", "w") as fp:
       for name, param in model.named_parameters():
           fp.write(f"{name} {param.shape} {param.dtype}\n")
   torch.save(model.state_dict(), "glm2_6b.pth")
   ```

   执行转换脚本，得到转换后的输出文件`glm2_6b.ckpt`。

   ```python
   import mindspore as ms
   import torch as pt
   from tqdm import tqdm

   pt_ckpt_path = "glm2_6b.pth"
   pt_param = pt.load(pt_ckpt_path)

   type_map = {"torch.float16": "ms.float16",
               "torch.float32": "ms.float32"}
   ms_param = []
   with open("check_pt_ckpt.txt", "w") as fp:
       for k, v in tqdm(pt_param.items()):
           if "word_embeddings.weight" in k:
               k = k.replace("word_embeddings.weight", "embedding_table")
           fp.write(f"{k} {v.shape} {v.dtype}\n")
           ms_param.append({"name": k, "data": ms.Tensor(v.numpy())})

   ms.save_checkpoint(ms_param, "glm2_6b.ckpt")
   ```

2. 获取MindFormers提供的已转换权重

   可通过from_pretrained接口下载，也可直接从下面的链接获取

   [glm2_6b权重](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/glm2/glm2_6b.ckpt)

   [tokenizer文件](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/glm2/tokenizer.model)

### [分布式训练/微调权重合并](../feature_cards/Transform_Ckpt.md)

分布式训练/微调后所得到的权重文件为根据策略切分后的权重，需要手动将切分权重合一，以用于评估和推理。

涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档模型[权重切分与合并](../feature_cards/Transform_Ckpt.md)

- step 1. 获取模型切分策略文件：

在执行微调脚本时，模型完成编译后，将会在`output/strategy`路径下生成各卡的切分策略文件，用于权重合并。

> 注：lora微调时需要确认配置文件`parallel context config`中`only_trainable_params`设为`False`，以获取所有参数完整策略。

- step 2. 运行`mindformers/tools/transform_ckpt.py`脚本进行多卡权重合并：

```shell
python transform_ckpt.py \
--src_ckpt_strategy {path}/output/strategy/ \
--src_ckpt_dir {path}/output/checkpoint/ \
--dst_ckpt_dir {path}/target_checkpoint/ \
--prefix glm2_6b
```

```text
# 参数说明
src_ckpt_strategy: 步骤1中的切分策略文件路径
src_ckpt_dir: 原切分权重文件夹
dst_ckpt_dir: 目标路径
prefix: ckpt文件前缀名
```

> 注：`transform_checkpoints` 接口当前仅mindspore 2.0以上版本支持，如当前硬件环境只支持2.0以下版本，可以新建conda环境安装mindspore 2.0的cpu版本以执行该脚本

## 基于API的快速使用

### 基于AutoClass的快速使用

可以使用AutoClass接口，通过模型名称获取相应的model/preprocess/tokenizer等实例，并自动下载并加载权重

`from_pretrained()` 接口会自动从云上下载预训练的模型，存储路径：`./checkpoint_download/glm2`

```python
import mindspore
from mindformers import AutoConfig, AutoModel, AutoTokenizer

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

tokenizer = AutoTokenizer.from_pretrained('glm2_6b')

# model的实例化有以下两种方式，选择其中一种进行实例化即可
# 1. 直接根据默认配置实例化
model = AutoModel.from_pretrained('glm2_6b')
# 2. 自定义修改配置后实例化
config = AutoConfig.from_pretrained('glm2_6b')
config.use_past = True                  # 此处修改默认配置，开启增量推理能够加速推理性能
# config.xxx = xxx                      # 根据需求自定义修改其余模型配置
model = AutoModel.from_config(config)   # 从自定义配置项中实例化模型

inputs = tokenizer("你好")["input_ids"]
# 首次调用model.generate()进行推理将包含图编译时间，推理性能显示不准确，多次重复调用以获取准确的推理性能
outputs = model.generate(inputs, max_new_tokens=20, do_sample=True, top_k=3)
response = tokenizer.decode(outputs)
print(response)
# ['你好，作为一名人工智能助手，我欢迎您随时向我提问。']
```

### 基于Trainer的快速训练，微调，评测，推理

> 注：下面仅显示接口使用方式，模型启动训练需求多卡分布式训练，训练脚本需配合分布式脚本启动

```python
import mindspore
from mindformers.trainer import Trainer

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

# 初始化预训练任务
trainer = Trainer(task='text_generation',
                  model='glm2_6b',
                  train_dataset='path/to/train_dataset',
                  eval_dataset='path/to/eval_dataset')

# 开启预训练
# 请参照多卡训练，glm2_6b不支持单卡启动训练
# trainer.train()

# 开启全量微调
# 请参照多卡微调，glm2_6b不支持单卡启动全量微调
# trainer.finetune()

# 开启评测
# 需要在configs/glm2/run_glm2_6b.yaml中将seq_length修改为256
trainer.evaluate()

# 开启推理
predict_result = trainer.predict(input_data="你好")
print(predict_result)
# [{'text_generation_text': ['你好，我是 ChatGLM2-6B， 一个人工智能助手。我背后使用的模型是 GLM2-6B， 是一种大型语言模型， 具有超过 2000 亿参数，支持多种任务。']}]
```

### 基于Pipeline的快速推理

```python
from mindformers import pipeline, TextGenerationPipeline
task_pipeline = pipeline(task='text_generation', model='glm2_6b', max_length=2048)
task_pipeline('你好')
# [{'text_generation_text': ['你好，我是 ChatGLM2-6B， 一个人工智能助手。我背后使用的模型是 GLM2-6B， 是一种大型语言模型， 具有超过 2000 亿参数，支持多种任务。']}]
pipeline = TextGenerationPipeline(model='glm2_6b', max_length=2048)
predict_result = pipeline("你好")
print(predict_result)
# [{'text_generation_text': ['你好，我是 ChatGLM2-6B， 一个人工智能助手。我背后使用的模型是 GLM2-6B， 是一种大型语言模型， 具有超过 2000 亿参数，支持多种任务。']}]
```

## 微调

下面以 [ADGEN](https://aclanthology.org/D19-1321.pdf) (广告生成) 数据集为例介绍代码的使用方法

### 数据集准备

ADGEN 数据集任务为根据输入（content）生成一段广告词（summary）。

```json
{
    "content": "类型#上衣*版型#宽松*版型#显瘦*图案#线条*衣样式#衬衫*衣袖型#泡泡袖*衣款式#抽绳",
    "summary": "这件衬衫的款式非常的宽松，利落的线条可以很好的隐藏身材上的小缺点，穿在身上有着很好的显瘦效果。领口装饰了一个可爱的抽绳，漂亮的绳结展现出了十足的个性，配合时尚的泡泡袖型，尽显女性甜美可爱的气息。"
}
```

从 [Google Drive](https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view?usp=sharing) 或者 [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1) 下载处理好的 ADGEN 数据集，目录结构为

```text
AdvertiseGen
  ├── train.json
  └── dev.json
```

将任务配置文件 `configs/glm2/run_glm2_6b_*.yaml` 中的 `==== dataset config ====` 部分替换成：

```yaml
train_dataset: &train_dataset
  data_loader:
    type: ADGenDataLoader
    dataset_dir: "/path/to/AdvertiseGen/train.json"
    shuffle: True
    phase: "train"
    version: 2
    origin_columns: ["content", "summary"]
  tokenizer:
    type: ChatGLM2Tokenizer
    vocab_file: "/path/to/tokenizer.model"
  input_columns: ["input_ids", "labels"]
  max_source_length: 64
  max_target_length: 128
  ignore_pad_token_for_loss: True
  num_parallel_workers: 8
  python_multiprocessing: False
  drop_remainder: True
  batch_size: 1
  repeat: 1
  numa_enable: False
  prefetch_size: 1
  seed: 0

train_dataset_task:
  type: KeyWordGenDataset
  dataset_config: *train_dataset

eval_dataset: &eval_dataset
  data_loader:
    type: ADGenDataLoader
    dataset_dir: "/path/to/AdvertiseGen/dev.json"
    shuffle: False
    phase: "eval"
    version: 2
    origin_columns: ["content", "summary"]
  tokenizer:
    type: ChatGLM2Tokenizer
    vocab_file: "/path/to/tokenizer.model"
  max_source_length: 256
  max_target_length: 256
  ignore_pad_token_for_loss: True
  input_columns: ["input_ids", "labels"]
  num_parallel_workers: 8
  python_multiprocessing: False
  drop_remainder: True
  batch_size: 1
  repeat: 1
  numa_enable: False
  prefetch_size: 1
  seed: 0

eval_dataset_task:
  type: KeyWordGenDataset
  dataset_config: *eval_dataset
```

> 注意：微调时的模型`seq_length`需要等于微调数据集的`max_source_length + max_target_length + 1`。
> yaml文件中默认的`seq_length: 193`以及`max_source_length: 64`和`max_target_length: 128`适用于ADGEN数据集

### 全参微调

全参微调使用 `configs/glm2/run_glm2_6b.yaml` 配置文件，配置文件中定义了微调所需的各配置项

修改数据集/模型权重配置路径：

- 数据集：修改 `configs/glm2/run_glm2_6b.yaml` 脚本中`train_dataset` 的 `dataset_dir` 为前文生成的数据集路径。
- 加载预训练模型权重：修改 `configs/glm2/run_glm2_6b.yaml` 脚本中的 `load_checkpoint` 为预训练模型权重路径。

#### 单卡微调

```shell
cd scripts
# Usage Help: bash run_stanalone.sh [CONFIG_PATH] [DEVICE_ID] [RUN_STATUS]
bash run_standalone.sh ../configs/glm2/run_glm2_6b_finetune.yaml 0 finetune
```

训练的log日志路径：mindformers/scripts/mf_standalone/

checkpoint存储路径：mindformers/scripts/mf_standalone/output/checkpoint

#### 多卡微调

- 单机多卡

多卡运行需要RANK_FILE_TABLE，请参考前期准备-[生成RANK_TABLE_FILE](#生成ranktablefile)

```shell
cd scripts
# Usage Help: bash run_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH] [DEVICE_RANGE] [RUN_STATUS]
bash run_distribute.sh /path/to/hccl_8p_01234567_127.0.1.1.json ../configs/glm2/run_glm2_6b.yaml '[0,8]' finetune
# 将此处rank_table_file替换为实际路径
```

参数说明

```text
RANK_TABLE_FILE: 由mindformers/tools/hccl_tools.py生成的分布式json文件
CONFIG_PATH: 为configs文件夹下面的glm2/run_glm2_6b.yaml配置文件
DEVICE_RANGE: 为单机分布式卡的范围，如 '[0,8]' 为8卡分布式，不包含8本身
RUN_STATUS: 为任务运行状态，支持关键字 train\finetune\eval\predict
```

训练的log日志路径：mindformers/output/log

checkpoint存储路径：mindformers/output/checkpoint

- 多机多卡

多机多卡运行需要合并不同机器的RANK_FILE_TABLE，参考前期准备-[多机RANK_TABLE_FILE合并](#多机ranktablefile合并)

在每台机器上启动`bash run_distribute.sh`。

```bash
server_count=12
device_num=8*$server_count
# launch ranks in the 0th server
cd scripts
bash run_distribute.sh $RANK_TABLE_FILE path/to/config.yaml [0,8] finetune $device_num

# launch ranks in the 1-11 server via ssh
for idx in {1..11}
do
    let rank_start=8*$idx
    let rank_end=$rank_start+8
    ssh ${IP_LIST[$idx]} "cd scripts; bash run_distribute.sh $RANK_TABLE_FILE path/to/config.yaml [$rank_start,$rank_end] finetune $device_num"
done
```

其中

- `RANK_TABLE_FILE`为上一步汇总并分发的总rank table文件；
- `IP_LIST`为12台服务器的IP地址。如192.168.0.[0-11]

```bash
IP_LIST=("192.168.0.0", "192.168.0.1", ..., "192.168.0.11")
```

### LoRA微调

全参微调能够在微调数据集上取得良好效果，但存在遗忘预训练知识的现象。
因此推荐使用低参微调算法，冻结原模型权重，仅在小规模参数量上进行训练，在微调数据集上取得良好效果的同时，缓解模型遗忘现象

使用LoRA算法进行低参微调时，使用 `configs/glm2/run_glm2_6b_lora.yaml` 配置文件，该配置文件包含了lora低参微调算法所需的配置项

修改数据集/模型权重配置路径：

- 数据集：修改 `mindformers/configs/glm2/run_glm2_6b_lora.yaml` 脚本中`train_dataset` 的 `dataset_dir` 为前文生成的数据集路径。
- 加载预训练模型权重：修改 `mindformers/configs/glm2/run_glm2_6b_lora.yaml` 脚本中的 `load_checkpoint` 为预训练模型权重路径。

#### 单卡微调

```shell
cd scripts
# Usage Help: bash run_stanalone.sh [CONFIG_PATH] [DEVICE_ID] [RUN_STATUS]
bash run_standalone.sh ../configs/glm2/run_glm2_6b_lora.yaml 0 finetune
```

训练的log日志路径：mindformers/scripts/mf_standalone/

checkpoint存储路径：mindformers/scripts/mf_standalone/output/checkpoint

#### 多卡微调

在每台机器上启动`bash run_distribute.sh`。

```bash
server_count=12
device_num=8*$server_count
# launch ranks in the 0th server
cd scripts
bash run_distribute.sh $RANK_TABLE_FILE path/to/config_lora.yaml [0,8] finetune $device_num

# launch ranks in the 1-11 server via ssh
for idx in {1..11}
do
    let rank_start=8*$idx
    let rank_end=$rank_start+8
    ssh ${IP_LIST[$idx]} "cd scripts; bash run_distribute.sh $RANK_TABLE_FILE path/to/config_lora.yaml [$rank_start,$rank_end] finetune $device_num"
done
```

其中

- `RANK_TABLE_FILE`为上一步汇总并分发的总rank table文件；
- `IP_LIST`为12台服务器的IP地址。如192.168.0.[0-11]

```bash
IP_LIST=("192.168.0.0", "192.168.0.1", ..., "192.168.0.11")
```

### 边训边推

将训练配置文件的 `do_eval: False` 设置为 `do_eval: True`，并且需要将 `train_dataset` 和 `eval_dataset` 的 `max_source_length`、`max_target_length` 以及 `batch_size`项设置为相同值，并且保持 `max_source_length + max_target_length + 1 = seq_length`，如下所示：

```yaml
model:
  model_config:
    seq_length: 193
train_dataset: &train_dataset
  max_source_length: 64
  max_target_length: 128
  batch_size: 8
eval_dataset: &eval_dataset
  max_source_length: 64
  max_target_length: 128
  batch_size: 8

eval_step_interval: 500 # 表示每 500 step 评估 1 次，-1 表示step不评估
eval_epoch_interval: -1 # 表示间隔多少 epoch 评估 1 次，-1 表示epoch不评估
```

## 评测

### 文本生成

### 数据集准备-文本生成

见微调章节的[数据集准备](#数据集准备)

评测时模型`seq_length`需要等于评测数据集的`max_source_length`和`max_target_length`。因此修改yaml中模型`seq_length`为256：

```yaml
model:
  model_config:
    seq_length: 256
```

### 单卡评测

#### 1. 全参微调

使用全参微调权重时，启动如下shell脚本，执行单卡评估

配置文件选择 `configs/glm2/run_glm2_6b_finetune_eval.yaml` glm2模型推理配置，修改其中`model`字段下`model_config`中`use_past: True`开启增量推理使评估速度更快

```bash
python run_mindformer.py --config configs/glm2/run_glm2_6b_finetune_eval.yaml --run_mode eval --load_checkpoint /path/to/glm2_6b_finetune.ckpt --device_id 0 --use_parallel False
```

#### 2. LoRA 微调

使用LoRA低参微调权重时，启动如下shell脚本，执行单卡评估

配置文件选择 `configs/glm2/run_glm2_6b_lora_eval.yaml` glm2_lora模型推理配置，此配置可用于lora模型，修改其中`model`字段下`model_config`中`use_past: True`开启增量推理使评估速度更快

```bash
python run_mindformer.py --config configs/glm2/run_glm2_6b_lora_eval.yaml --run_mode eval --load_checkpoint /path/to/glm2_6b_lora.ckpt --device_id 0 --use_parallel False
```

### 多卡评测

- 单机多卡

多卡运行需要RANK_FILE_TABLE，请参考前期准备-[生成RANK_TABLE_FILE](#生成ranktablefile)

```shell
cd scripts
bash run_distribute.sh RANK_TABLE_FILE path/to/config.yaml [0,8] eval 8
```

- 多机多卡

多机多卡运行需要合并不同机器的RANK_FILE_TABLE，参考前期准备-[多机RANK_TABLE_FILE合并](#多机ranktablefile合并)

在每台机器上启动`bash run_distribute.sh`。

```bash
server_count=12
device_num=8*$server_count
# launch ranks in the 0th server
cd scripts
bash run_distribute.sh $RANK_TABLE_FILE path/to/config.yaml [0,8] eval $device_num

# launch ranks in the 1-11 server via ssh
for idx in {1..11}
do
    let rank_start=8*$idx
    let rank_end=$rank_start+8
    ssh ${IP_LIST[$idx]} "cd scripts; bash run_distribute.sh $RANK_TABLE_FILE path/to/config.yaml [$rank_start,$rank_end] eval $device_num"
done
```

其中

- `RANK_TABLE_FILE`为上一步汇总并分发的总rank table文件；
- `IP_LIST`为12台服务器的IP地址。如192.168.0.[0-11]

```bash
IP_LIST=("192.168.0.0", "192.168.0.1", ..., "192.168.0.11")
```

## 推理

### 基于generate的推理

下面提供一个模型推理样例脚本 `infer.py`

**注意**： LoRA微调模型替换成 `glm2_6b_lora`

```python
from mindformers import AutoConfig, AutoModel, AutoTokenizer
import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)

# **注意** LoRA微调模型替换成 “glm2_6b_lora”
config = AutoConfig.from_pretrained("glm2_6b")

# 可以在此使用下行代码指定自定义权重进行推理，默认使用自动从obs上下载的预训练权重
# config.checkpoint_name_or_path = "/path/to/glm2_6b_finetune.ckpt"
config.use_past = True
model = AutoModel.from_config(config)
tokenizer = AutoTokenizer.from_pretrained("glm2_6b")

inputs = tokenizer(tokenizer.build_prompt("你好"))["input_ids"]
outputs = model.generate(inputs, max_length=128)
print(tokenizer.decode(outputs))
# ['[Round 1]\n\n问：你好\n\n答： 你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。']
inputs = tokenizer(tokenizer.build_prompt("请介绍一下华为"))["input_ids"]
outputs = model.generate(inputs, max_length=128)
print(tokenizer.decode(outputs))
# ['[Round 1]\n\n问：请介绍一下华为\n\n答： 华为是一家总部位于中国的全球知名科技公司,成立于1987年,是全球领先的信息与通信技术(ICT)解决方案供
# 应商之一。\n\n华为的业务范围涵盖了网络、终端、云计算、软件、芯片等多个领域,旗下的智能手机、电脑、平板电脑等消费电子产品在国内外市场上都享有较高
# 的声誉。此外,华为还在5G、人工智能、云计算等领域取得了重要的进展,为全球用户提供了更高效、更智能的科技体验。\n\n华为一直致力于技术创新,研发投入
# 占公司总收入的比例超过']
inputs = tokenizer(tokenizer.build_prompt("晚上睡不着应该怎么办"))["input_ids"]
outputs = model.generate(inputs, max_length=128)
print(tokenizer.decode(outputs))
# ['[Round 1]\n\n问：晚上睡不着应该怎么办\n\n答： 以下是一些有助于晚上睡觉的技巧:\n\n1. 创建一个规律的睡眠时间表:每天在相同的时间上床并尽量
# 在同一时间起床,有助于身体建立规律的生物钟。\n\n2. 创建一个舒适的睡眠环境:确保房间安静、黑暗、凉爽和舒适。如果需要,可以使用睡眠面罩、耳塞或空气
# 净化器来帮助创造一个更舒适的睡眠环境。\n\n3. 避免使用电子设备:在睡觉前一两个小时内避免使用电子设备,如']
inputs = tokenizer(tokenizer.build_prompt("类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞"))["input_ids"]
outputs = model.generate(inputs, max_length=128)
print(tokenizer.decode(outputs))
# ['[Round 1]\n\n问：类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞\n\n答： 上衣 材质:牛仔布 颜色:白色 风格:
# 简约 图案:刺绣 衣样式:外套 衣款式:破洞\n\n这件上衣由牛仔布制成,采用了简约的风格,图案为刺绣设计,衣样式为外套,衣款式为破洞。']
```

### 脚本启动

> GLM2使用脚本进行推理时需要手动对输入问题添加prompt，prompt模板的形式为`[Round 1]\n\n问：{此处填写问题}\n\n答：`。
>
> 如果问题是`为什么说地球是独一无二的`，添加prompt后为`[Round 1]\n\n问：为什么说地球是独一无二的\n\n答：`。

#### 单卡推理

```bash
python run_mindformer.py --config path/to/config.yaml --run_mode predict --predict_data "[Round 1]\n\n问：你好\n\n答："
#  [{'text_generation_text': ['[Round 1]\n\n问：你好\n\n答： 你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。']}]
```

#### 多卡推理

多卡运行需要RANK_FILE_TABLE，请参考前期准备-[生成RANK_TABLE_FILE](#生成ranktablefile)

- 单机多卡

```shell
cd scripts
bash run_distribute.sh RANK_TABLE_FILE path/to/config.yaml [0,8] predict 8 "[Round 1]\n\n问：你好\n\n答："
```

多机多卡运行需要合并不同机器的RANK_FILE_TABLE，参考前期准备-[多机RANK_TABLE_FILE合并](#多机ranktablefile合并)

- 多机多卡

在每台机器上启动`bash run_distribute.sh`。

```bash
server_count=12
device_num=8*$server_count
# launch ranks in the 0th server
cd scripts
bash run_distribute.sh $RANK_TABLE_FILE path/to/config.yaml [0,8] predict $device_num 你好

# launch ranks in the 1-11 server via ssh
for idx in {1..11}
do
    let rank_start=8*$idx
    let rank_end=$rank_start+8
    ssh ${IP_LIST[$idx]} "cd scripts; bash run_distribute.sh $RANK_TABLE_FILE path/to/config.yaml [$rank_start,$rank_end] predict $device_num \"[Round 1]\n\n问：你好\n\n答：\""
done
```

其中

- `RANK_TABLE_FILE`为上一步汇总并分发的总rank table文件；
- `IP_LIST`为12台服务器的IP地址。如192.168.0.[0-11]

```bash
IP_LIST=("192.168.0.0", "192.168.0.1", ..., "192.168.0.11")
```

## Mindspore-Lite 推理及量化

### 基本介绍

　　MindFormers 定位打造训练->微调->部署的端到端大模型工具套件，为了更好性能地部署已经微调训练好的大模型，我们利用MindSpore打造的推理引擎 [MindSpore_lite](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Flite)，为用户提供了开箱即用的推理部署方案，为用户提供端到端的大模型解决方案，帮助用户使能大模型业务。

　　Lite 推理大致分两步：权重转换导出 MindIR -> Lite 推理，接下来分别描述上述两个过程。

### MindIR 导出

1. 修改模型相关的配置文件 configs/glm2/export_glm2_6b.yaml，其中需要关注这几项：

   ```yaml
   # export
   infer:
       prefill_model_path: "glm2_export/glm2_6b_prefill_seq512.mindir" # 保存mindir的位置
       increment_model_path: "glm2_export/glm2_6b_inc_seq512.mindir"   # 保存mindir的位置
       infer_seq_length: 512 # 需要保持跟 model-model_config-seq_length 一致

   # ==== model config ====
   model:
     model_config:
       seq_length: 512
       checkpoint_name_or_path: "/path/to/your/*.ckpt"
   ```

2. 执行export.py，完成模型转换

   ```bash
   python mindformers/tools/export.py --config_path configs/glm2/export_glm2_6b.yaml
   ```

### 执行推理

1. 新建推理配置文件：lite.ini

    ```ini
    [ascend_context]
    provider=ge

    [ge_session_options]
    ge.exec.formatMode=1
    ge.exec.precision_mode=must_keep_origin_dtype
    ```

2. 执行命令：

   ```bash
   python run_infer_main.py --device_id 0 --model_name glm2_6b --prefill_model_path glm2_export/glm2_6b_prefill_seq512_graph.mindir --increment_model_path glm2_export/glm2_6b_inc_seq512_graph.mindir --config_path lite.ini --is_sample_acceleration False --seq_length 512 --add_special_tokens True
   ```

   注：如果是int8量化后推理，将 `prefill_model_path`​ 和 `increment_model_path`​ 修改为 int8 量化后的 MindIR 即可。

　　等待模型载入、编译后，出现：

   ```bash
   Please enter your predict data:
   ```

　　输入：

   ```bash
   [Round 1]

   问：你好。

   答：
   ```

　　输出：

   ```bash
   ['[Round 1]\n\n问：你好。\n\n答： 你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。']
   ```