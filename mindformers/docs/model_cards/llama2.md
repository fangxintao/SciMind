# Llama 2

## 模型描述

Llama 2，是Meta基于LLaMA 1的更新版本，基于新的公开可用数据混合进行训练，同时将预训练语料库的大小增加了40%，最后将模型的上下文长度翻倍（由2048提高到4096），并采用了分组查询注意力机制。Llama 2模型是类GPT模型，是一个生成式的语言模型，主要是用于预测下一个单词。Llama 2按照参数量，目前有三个版本：Llama 2-7B（7B）、Llama 2-13B（13B）、Llama 2-70B（70B），本仓库已全部支持三版权重，权重文件来源于MetaLLama2。Llama 2 的7B和13B 模型结构与LLaMA 1一致，70B 则加入分组查询注意力（GQA）。

[Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/pdf/2307.09288.pdf%C3%82%C2%A0)

``` text
@article{touvron2023llama,
  title={Llama 2: Open foundation and fine-tuned chat models},
  author={Touvron, Hugo and Martin, Louis and Stone, Kevin and Albert, Peter and Almahairi, Amjad and Babaei, Yasmine and Bashlykov, Nikolay and Batra, Soumya and Bhargava, Prajjwal and Bhosale, Shruti and others},
  journal={arXiv preprint arXiv:2307.09288},
  year={2023}
}
```

## 模型性能

基于910B

llama2_7b:

| config                                                       | task                  | Datasets  | SeqLength | metric | phase             | score     | performance  |
| ------------------------------------------------------------ | --------------------- | --------- | --------- | ------ | ----------------- | --------- | ------------ |
| [llama2_7b](../../configs/llama2/run_llama2_7b_910b.yaml)    | text_generation       | wiki      | 4096      | -      | [train](#预训练)  | -         | 2433 tks/s/p |
| [llama2_7b](../../configs/llama2/run_llama2_7b_910b_finetune.yaml) | text_generation       | alpaca    | 2048      | -      | [finetune](#微调) | -         | 3523 tks/s/p |
| [llama2_7b_lora](../../configs/llama2/run_llama2_7b_lora_910b.yaml) | text_generation       | alpaca    | 2048      | -      | [finetune](#微调) | -         | 4269 tks/s/p |
| [llama2_7b](../../configs/llama2/run_llama2_7b_910b.yaml)    | text_generation       | WikiText2 | -         | PPL    | [eval](#评测)     | 6.58      | -            |
| [llama2_7b](../../configs/llama2/run_llama2_7b_910b.yaml)    | reading comprehension | SQuAD 1.1 | -         | EM/F1  | [eval](#评测)     | 39.6/60.5 | -            |

llama2_13b:

| config                                                       | task                  | Datasets  | SeqLength | metric | phase             | score       | performance   |
| ------------------------------------------------------------ | --------------------- | --------- | --------- | ------ | ----------------- | ----------- | ------------- |
| [llama2_13b](../../configs/llama2/run_llama2_13b_910b.yaml)  | text_generation       | wiki      | 4096      | -      | [train](#预训练)  | -           | 1285  tks/s/p |
| [llama2_13b](../../configs/llama2/run_llama2_13b_910b_finetune.yaml) | text_generation       | alpaca    | 2048      | -      | [finetune](#微调) | -           | 1575 tks/s/p  |
| [llama2_13b_lora](../../configs/llama2/run_llama2_13b_lora_910b.yaml) | text_generation       | alpaca    | 2048      | -      | [finetune](#微调) | -           | 2275 tks/s/p  |
| [llama2_13b](../../configs/llama2/run_llama2_13b_910b.yaml)  | text_generation       | WikiText2 | -         | PPL    | [eval](#评测)     | 6.14        | -             |
| [llama2_13b](../../configs/llama2/run_llama2_13b_910b.yaml)  | reading comprehension | SQuAD 1.1 | -         | EM/F1  | [eval](#评测)     | 27.91/44.23 | -             |

llama2_70b 待补充。

## 仓库介绍

`Llama 2` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：`mindformers/models/llama`

   ```bash
   llama
       ├── __init__.py
       ├── convert_weight.py         # 权重转换脚本
       ├── llama.py                  # 模型实现
       ├── llama_config.py           # 模型配置项
       ├── llama_layer.py            # llama网络层定义
       ├── llama_processor.py        # llama预处理
       ├── llama_tokenizer.py        # tokenizer
       └── llama_transformer.py      # transformer层实现
   ```

2. 模型配置：`configs/llama2`

   ```bash
   llama
       ├── run_llama2_7b.yaml         # 7b模型全量微调启动配置
       ├── run_llama2_13b.yaml        # 13b全量微调启动配置
       └── run_llama2_70b.yaml        # 70b全量微调启动配置
   ```

3. 数据预处理脚本：

   ```bash
   mindformers/tools/dataset_preprocess/llama/
       ├── alpaca_converter.py     # 基于fschat的alpaca数据集格式转换脚本
       ├── llama_preprocess.py     # llama模型的mindrecord数据处理脚本
       └── squad_data_process.py   # squad数据集格式转换脚本
   ```

## 前期准备

### 环境要求

- 硬件：Ascend 910A/910B
- MindSpore：2.2.0
- CANN: 7.0
- MindFormers版本：r0.8

注：910b芯片：7b,13b推理可在单机单卡上完成部署；70b推理至少使用8卡，全参微调至少需要4机32卡，推荐使用8机64卡。

### [mindformers安装](../../README.md#二mindformers安装)

### 生成RANK_TABLE_FILE(多卡运行必须环节)

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

### 多机RANK_TABLE_FILE合并(多机多卡必备环节)

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

1.从huggingface下载英文预训练权重（权重来源于MetaLLama2）：

- [llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b)
- [llama2-13b](https://huggingface.co/meta-llama/Llama-2-13b)
- [llama2-70b](https://huggingface.co/meta-llama/Llama-2-70b)

注：Llama 2的所有权重都需要向Meta提交[申请](https://ai.meta.com/resources/models-and-libraries/llama-downloads)，如有需要，请开发者自行申请。

下载完成后，运行如下转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python mindformers/models/llama/convert_weight.py \
--torch_ckpt_dir TORCH_CKPT_DIR \
--mindspore_ckpt_path {path}/MS_CKPT_NAME
```

```text
# 参数说明
torch_ckpt_dir: huggingface权重保存目录路径
mindspore_ckpt_path: 权重保存文件名，可以指定自定义保存路径
```

2. 获取MindFormers提供的已转换权重
    可通过from_pretrained接口下载，也可直接从下面的链接获取

- [llama2_7b权重](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/llama2_7b.ckpt)
- [llama2_13b权重](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/llama2-13b-fp16.ckpt)
- [tokenizer文件](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/tokenizer.model)

### [分布式训练/微调权重合并](../feature_cards/Transform_Ckpt.md)

分布式训练/微调后所得到的权重文件为根据策略切分后的权重，需要手动将切分权重合一，以用于评估和推理。

涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档模型[权重切分与合并](../feature_cards/Transform_Ckpt.md)

- step 1. 获取模型切分策略文件：

在执行微调脚本时，模型完成编译后，将会在`output/strategy`路径下生成各卡的切分策略文件，用于权重合并。

- step 2. 运行`mindformers/tools/transform_ckpt.py`脚本进行多卡权重合并：

```shell
python transform_ckpt.py \
--src_ckpt_strategy {path}/output/strategy/ \
--src_ckpt_dir {path}/output/checkpoint/ \
--dst_ckpt_dir {path}/target_checkpoint/ \
--prefix llama2_7b
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

`from_pretrained()` 接口会自动从云上下载预训练的模型，存储路径：`mindformers/checkpoint_download/llama2`

```python
import mindspore
from mindformers import AutoConfig, AutoModel, AutoTokenizer

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

tokenizer = AutoTokenizer.from_pretrained('llama2_7b')

# model的实例化有以下两种方式，选择其中一种进行实例化即可
# 1. 直接根据默认配置实例化
model = AutoModel.from_pretrained('llama2_7b')
# 2. 自定义修改配置后实例化
config = AutoConfig.from_pretrained('llama2_7b')
config.use_past = True                  # 此处修改默认配置，开启增量推理能够加速推理性能
# config.xxx = xxx                      # 根据需求自定义修改其余模型配置
model = AutoModel.from_config(config)   # 从自定义配置项中实例化模型

inputs = tokenizer("I love Beijing, because")["input_ids"]
# 首次调用model.generate()进行推理将包含图编译时间，推理性能显示不准确，多次重复调用以获取准确的推理性能
outputs = model.generate(inputs, max_new_tokens=30, do_sample=False)
response = tokenizer.decode(outputs)
print(response)
# ['<s>I love Beijing, because it’s a city that is constantly changing. I have been living here for 10 years and I have seen the city change so much.I']
```

### 基于Trainer的快速评测，推理

> 注：下面仅显示接口使用方式，模型启动训练需求多卡分布式训练，训练脚本需配合分布式脚本启动

```python
import mindspore
from mindformers.trainer import Trainer

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

# 初始化预训练任务
trainer = Trainer(task='text_generation',
                  model='llama2_7b',
                  train_dataset='path/to/train_dataset',
                  eval_dataset='path/to/eval_dataset')

# 开启评测
trainer.evaluate()

# 开启推理
predict_result = trainer.predict(input_data="I love Beijing, because")
# [{'text_generation_text': ['<s>I love Beijing, because it’s a a city that is constantly changing. I have been living here for 10 years and I have']}]
```

### 基于Pipeline的快速推理

```python
import mindspore
from mindformers.pipeline import pipeline

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

pipeline_task = pipeline("text_generation", model='llama2_7b', max_length=30)
pipeline_result = pipeline_task("I love Beijing, because", do_sample=False)
print(pipeline_result)
# [{'text_generation_text': ['<s>I love Beijing, because it’s a a city that is constantly changing. I have been living here for 10 years and I have']}]
```

## 预训练

### 数据集准备

以Wikitext2数据集为例:

- 数据集下载：[WikiText2数据集](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip)

- 分词模型下载：例如下载申请通过后huggingface里对应Files 中的tokenizer.model

- 使用以下预处理脚本生成mindrecord训练数据

```bash
# 使用tools/dataset_preprocess/llama/llama_preprocess.py进行数据预处理+Mindrecord数据生成
python llama_preprocess.py \
--dataset_type wiki \
--input_glob  /{path}/wiki.train.tokens \
--model_file /{path}/tokenizer.model \
--seq_length 4096 \
--output_file /{path}/wiki4096.mindrecord
```

### 脚本启动（Llama 2-7B为例）

多卡运行需要RANK_FILE_TABLE，请参考前期准备-[生成RANK_TABLE_FILE](#生成rank_table_file多卡运行必须环节)

#### 多卡训练

##### 单机多卡

- step 1. 修改模型对应的配置文件。

在模型对应的配置文件`configs/llama/run_llama2_{7/13/70}b.yaml`中，用户可自行修改模型、训练相关参数，并通过`train_dataset`的`dataset_dir`参数，指定训练数据集的路径。

配置文件中各参数含义详见[Config配置说明文档](https://gitee.com/mindspore/mindformers/blob/master/configs/README.md)。

- step2：进入`scripts`文件夹，启动运行脚本，进行8卡分布式运行。

```shell
cd scripts
bash run_distribute.sh hccl_xxxx.json ../configs/llama2/run_llama2_7b.yaml [0,8] train
```

```text
# 脚本启动格式：
bash run_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH] [DEVICE_RANGE] [RUN_MODE]

# 参数说明
RANK_TABLE_FILE: 由mindformers/tools/hccl_tools.py生成的分布式json文件
CONFIG_PATH: 为configs文件夹下面的llama/run_llama2_7b.yaml配置文件
DEVICE_RANGE: 为单机分布式卡的范围，如[0,8]为8卡分布式，不包含8本身
RUN_MODE: 为任务运行状态，支持关键字 train\finetune\eval\predict
```

##### 多机多卡

- step 1. 多机多卡运行需要合并不同机器的RANK_FILE_TABLE，参考前期准备-[多机RANK_TABLE_FILE合并](#多机rank_table_file合并多机多卡必备环节)

> **注：需要保证执行的节点和RANK_TABLE_FIEL的节点顺序保持一致，即rank_id匹配。**

- step 2. 根据服务器节点数等信息，修改相应的配置。

```yaml
# 以llama2-13b模型两机训练为例，默认配置2机16卡，如果节点数有变，需要修改相应的配置。
# 配置文件路径：../configs/llama2/run_llama2_13b.yaml
parallel_config:
  data_parallel: 2
  model_parallel: 4
  pipeline_stage: 2
  micro_batch_num: 16
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

- step 3. 执行运行脚本。

在多机上同时拉起任务，每台机器拉起方式参考单机多卡启动方式。需注意，多机多卡的拉起方式，相对于单机多卡，多了一个总卡数`[RANK_SIZE]`的入参。

```shell
# 第一台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the first device} ../configs/llama2/run_llama2_13b.yaml [0,8] train 16
# 第二台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the second device} ../configs/llama2/run_llama2_13b.yaml [8,16] train 16
```

## 微调

### 数据集准备

目前提供alpaca数据集的预处理脚本用于全参微调任务。

数据集下载链接如下：

- [alpaca_data](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)

alpaca数据集原始格式样例：

```text
# alpaca examples:
    {
        "instruction": "Describe a time when you had to make a difficult decision.",
        "input": "",
        "output": "I had to make a difficult decision when I was working as a project manager at a construction company. I was in charge of a project that needed to be completed by a certain date in order to meet the client\u2019s expectations. However, due to unexpected delays, we were not able to meet the deadline and so I had to make a difficult decision. I decided to extend the deadline, but I had to stretch the team\u2019s resources even further and increase the budget. Although it was a risky decision, I ultimately decided to go ahead with it to ensure that the project was completed on time and that the client\u2019s expectations were met. The project was eventually successfully completed and this was seen as a testament to my leadership and decision-making abilities."
    },
    {
        "instruction": "Identify the odd one out.",
        "input": "Twitter, Instagram, Telegram",
        "output": "Telegram"
    },
```

- step 1. 执行`alpaca_converter.py`，使用fastchat工具添加prompts模板，将原始数据集转换为多轮对话格式。

``` bash
# 脚本路径：tools/dataset_preprocess/llama/alpaca_converter.py
# 执行转换脚本
python alpaca_converter.py \
--data_path /{path}/alpaca_data.json \
--output_path /{path}/alpaca-data-conversation.json
```

```text
# 参数说明
data_path: 存放alpaca数据的路径
output_path: 输出转换后对话格式的数据路径
```

转换后格式样例：

```text
{
    "id": "1",
    "conversations": [
      {
        "from": "human",
        "value": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nGive three tips for staying healthy.\n\n### Response:"
      },
      {
        "from": "gpt",
        "value": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
      }
    ]
  },
```

- step 2. 执行`llama_preprocess.py`，进行数据预处理、Mindrecord数据生成，将带有prompt模板的数据转换为mindrecord格式。

```bash
# 脚本路径：tools/dataset_preprocess/llama/llama_preprocess.py
# 由于此工具依赖fschat工具包解析prompt模板，请提前安装fschat >= 0.2.13 python = 3.9
python llama_preprocess.py \
--dataset_type qa \
--input_glob /{path}/alpaca-data-conversation.json \
--model_file /{path}/tokenizer.model \
--seq_length 2048 \
--output_file /{path}/alpaca-fastchat2048.mindrecord
```

### 全参微调

以llama2 7b为例

- step 1. 参考`config/llama2/run_llama2_7b_910b_finetune.yaml`中训练数据集路径为微调数据集路径，并在`input_columns`中添加`labels`。

```python
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "/{path}/alpaca-fastchat2048.mindrecord"
    shuffle: True
  input_columns: ["input_ids", "labels"]
```

- step 2. 修改微调时学习率, 优化器参数，`seq_length`, 新增 `context`中参数, 与预训练不同，微调配置如下：

```python
# optimizer
optimizer:
  type: FP32StateAdamWeightDecay
  beta1: 0.9
  beta2: 0.999
  eps: 1.e-8
  learning_rate: 1.e-5

# lr sechdule
lr_schedule:
  type: CosineWithWarmUpLR
  learning_rate: 1.e-5
  lr_end: 0
  warmup_ratio: 0.03
  total_steps: -1 # -1 means it will load the total steps of the dataset

# model config
model:
  model_config:
    type: LlamaConfig
    batch_size: 1 # add for increase predict
    seq_length: 2048

# context
context:
  runtime_num_threads: 1
```

>注意：alpaca数据集最长不超过2048，因此seq_length采用2048即可。

- step 3. 微调`llama2-7b`时修改并行策略配置，配置如下：

```python
# parallel_config
parallel_config:
    data_parallel: 2
    model_parallel: 1
    pipeline_stage: 4
```

- step4. 设置环境变量，变量配置如下：

```bash
export MS_DEV_SIDE_EFFECT_LOAD_ELIM=3  # 去除TensorMove
export MS_MEMORY_POOL_RECYCLE=1  # 内存优化
export GE_NOT_CUT=1   # 内存优化
export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"  # llama2_7b 不用设置该项
```

- step 5. 添加预训练权重路径，修改配置文件中的`load_checkpoint`，配置预训练权重路径。
- step 6. 启动微调任务，llama2-7b模型以单机八卡为例进行微调，命令如下：

```shell
cd scripts
bash run_distribute.sh [RANK_TABLE_FILE] ../configs/llama2/run_llama2_7b_910b_finetune.yaml [0,8] finetune
```

多机多卡微调任务启动参考[预训练章节](#预训练)，添加预训练权重，修改启动脚本中的`RUN_MODE`为`finetune`即可。

### LoRA微调

使用LoRA低参微调算法，冻结原模型权重，仅在小规模参数量上进行训练，使大模型在少量资源的情况下也能训练。

使用LoRA算法进行低参微调时，使用 `configs/llama2/run_llama2_7b_lora_910b.yaml` 配置文件，该配置文件包含了lora低参微调算法所需的配置项

修改数据集/模型权重配置路径：

- 数据集：修改 `mindformers/configs/llama2/run_llama2_7b_lora_910b.yaml` 脚本中`train_dataset` 的 `dataset_dir` 为前文生成的数据集路径。

- 加载预训练模型权重：修改 `mindformers/configs/llama2/run_llama2_7b_lora_910b.yaml` 脚本中的 `load_checkpoint` 为预训练模型权重路径。以llama2-7b 为例，有以下两种导入方式。

  1. 直接导入完整权重：

  ```yaml
  # 以llama2-7b为例
  load_checkpoint: {path}/llama2_7b.ckpt
  auto_trans_ckpt: False
  ```

  2. 使用分布式导入权重，路径设置为rank_0的上一层路径

  ```yaml
  # 将llama2_7b.ckpt 放入文件夹名称为rank_0的文件夹中，
  load_checkpoint: path/to/your/rank_0_file
  anto_trans_ckpt: True
  ```

#### 脚本启动

- step 1. 修改配置文件，参考全参微调修改训练数据集路径与预训练权重路径。

- step 2. 启动lora微调任务。

> 注：llama2_7b_lora模型支持单卡启动，需将配置文件中的`use_parallel`参数置为`False`。

```shell
cd scripts
# 单卡启动
bash run_standalone.sh ../configs/llama2/run_llama2_7b_910b_lora.yaml [DEVICE_ID] finetune
# 多卡启动（以单机八卡为例）
bash run_distribute.sh [RANK_TABLE_FILE] ../configs/llama2/run_llama2_7b_910b_lora.yaml [0,8] finetune
```

## 评测

Llama 2当前支持使用based model(初始权重) 进行评测任务如下：

| 任务类型 |  评测指标  |  数据集   |
| :------: | :--------: | :-------: |
| 文本生成 | Perplexity | WikiText2 |
| 阅读理解 |   Em/F1    | SQuAD 1.1 |

评测时加入`vocab_file`配置相应`tokenizer.model`路径；若使用910B进行评测，则还需在yaml中加入`ascend_config`配置：

```python
# context_config
context:
  ascend_config:
    precision_mode: "must_keep_origin_dtype"

# tokenizer 配置
processor:
  return_tensors: ms
  tokenizer:
    vocab_file: "path/to/tokenizer.model"
```

- 文本生成：

step 1. 获取数据集

[WikiText2数据集](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip)是从维基百科上经过验证的优质文章集中提取的超过1亿个token的集合。

step 2. 处理数据成mindrecord格式

```bash
# 使用tools/dataset_preprocess/llama/llama_preprocess.py进行数据预处理+Mindrecord数据生成
python llama_preprocess.py \
--dataset_type wiki \
--input_glob  /{path}/wiki.valid.tokens \
--model_file /{path}/tokenizer.model \
--seq_length 4095 \
--output_file /{path}/wiki4096.mindrecord
```

step 3. 开启评测，指标为PPL

```bash
python run_mindformer.py \
--config configs/llama2/run_llama2_7b.yaml \
--eval_dataset_dir /{path}/wiki4096.mindrecord \
--run_mode eval \
--load_checkpoint /{path}/llama2_7b.ckpt \
--epochs 1 \
--use_parallel False \
--device_id 0

# PerplexityMetric = {'PerplexityMetric': {'loss': 2.1142693907022476, 'PPL': 6.58}}
```

- 阅读理解：

step 1. 获取数据集

[SQuAD 1.1](https://data.deepai.org/squad1.1.zip)包含针对500+文章的10万+问答对,是一个阅读理解数据集，由维基百科文章上提出的问题组成，其中每个问题的答案都是相应文章中的一段文本。

step 2. 处理数据成mindrecord格式

```bash
# 使用tools/dataset_preprocess/llama/squad_data_process.py进行数据预处理+Mindrecord数据生成
python squad_data_process.py \
--input_file /{path}/squad/dev-v1.1.json \
--output_file /{path}/squad2048.mindrecord \
--mode eval \
--max_length 2048 \
--tokenizer_type "llama2_7b"
```

预处理后数据格式举例：

```text
Read the passage and answer the question below.

### Instruction:
The Panthers finished the regular season with a 15–1 record, and quarterback Cam Newton was named the NFL Most Valuable Player (MVP). They defeated the Arizona Cardinals 49–15 in the NFC Championship Game and advanced to their second Super Bowl appearance since the franchise was founded in 1995. The Broncos finished the regular season with a 12–4 record, and denied the New England Patriots a chance to defend their title from Super Bowl XLIX by defeating them 20–18 in the AFC Championship Game. They joined the Patriots, Dallas Cowboys, and Pittsburgh Steelers as one of four teams that have made eight appearances in the Super Bowl.

### Input:
Which Carolina Panthers player was named Most Valuable Player?

### Response:
Cam Newton
```

step 3. 修改配置文件，eval_dataset的input_columns中增加`labels`，修改metric类型为`EmF1Metric`，修改seq_length为`2048`

```python
# eval dataset
eval_dataset: &eval_dataset
  data_loader:
    type: MindDataset
    dataset_dir: ""
    shuffle: False
  input_columns: ["input_ids", "labels"]      # 增加"labels"

# metric
metric:
  type: EmF1Metric     # metric type设为EmF1Metric

# model config
model:
  model_config:
    type: LlamaConfig
    batch_size: 1 # add for increase predict
    seq_length: 2048
```

此外，要提高推理速度，可以进行如下配置，设置增量推理`use_past`，并限制生成最大长度`max_new_tokens`。

```python
# model config
use_past: True          # 开启增量推理
pretrain_seqlen: 4096
extend_method: "None"
offset: 0
checkpoint_name_or_path: "llama2_7b"
repetition_penalty: 1
max_decode_length: 512
top_k: 3
top_p: 1
do_sample: False
max_new_tokens: 20      #设置最大生成长度
```

step 4. 开启评测，指标为`Em/F1`

```bash
python run_mindformer.py \
--config configs/llama2/run_llama2_7b.yaml \
--eval_dataset_dir /{path}/squad2048.mindrecord \
--run_mode eval \
--load_checkpoint /{path}/llama2_7b.ckpt \
--epochs 1 \
--batch_size 1 \
--use_parallel False \
--device_id 0

# F1 score: 60.5, Em score: 39.6, total_count: 2067
```

### 分布式评测

对于较大模型比如llama2_70b，模型无法完全导入到单卡中进行评测，则需要进行分布式评测。可参考[权重切分与合并](../feature_cards/Transform_Ckpt.md) 中的案例进行评测相应修改，本实例参考案例三完整权重切分自动评测。

step 1. 修改权重文件夹目录结构如下，将模型权重放入rank_0的文件夹中。

```shell
path/to/checkpoint_dir
    ├──rank_0
        ├──model.ckpt
```

step 2. 修改config配置，`auto_trans_ckpt` 设为`True`，`model_parallel`设置为相应需要进行评测的卡数。`load_checkpoint` 路径设置为rank_0上一层的`path/to/checkpoint_dir`。

```python
load_checkpoint: path/to/checkpoint_dir
use_parallel: True
# model config
parallel_config:
  data_parallel: 1
  model_parallel: 8  # 改为相应卡数。70b推荐8卡推理
  pipeline_stage: 1
  use_seq_parallel: False
```

step 3. 按照之前的单卡评测的指导，将`eval_dataset` 中的配置相应修改，将评测数据集路径写入`dataset_dir`中。

```python
# eval dataset，以squad的mindrecord路径为例
eval_dataset: &eval_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "{path}/squad2048.mindrecord"
```

step 4. 参考[生成RANK_TABLE_FILE](#生成RANK_TABLE_FILE(多卡运行必须环节)) 生成相应卡数的RANK_TABLE_FILE。
step 5. 执行以下命令进行分布式评测

```shell
cd script
bash run_distribute.sh RANK_TABLE_FILE configs/llama2/predict_llama2_70b_910b.yaml [0,8] eval
```

## 推理

推理时将配置文件中`param_init_type`修改为`float32`；若为910B推理，则加入`ascend_config`配置。

```python
# model config
model:
  model_config:
    param_init_type: "float32"

# context_config 910B推理添加ascend_config
context:
  ascend_config:
    precision_mode: "must_keep_origin_dtype"
```

### 基于generate的推理

以下为基于model.generate接口的自定义推理脚本，支持多卡多batch推理。

```python
# predict_custom.py 文件
import argparse
import mindspore as ms
import numpy as np
import os
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train import Model

from mindformers import MindFormerConfig, LlamaConfig, TransformerOpParallelConfig, AutoTokenizer, LlamaForCausalLM
from mindformers import init_context, ContextConfig, ParallelContextConfig
from mindformers.tools.utils import str2bool
from mindformers.trainer.utils import get_last_checkpoint


def main(args):
    """main function."""
    # 多batch输入
    inputs = ["I love Beijing, because",
              "LLaMA is a",
              "Huawei is a company that"]

    # set model config
    config = MindFormerConfig(args.yaml_file)

    # 初始化环境
    init_context(use_parallel=config.use_parallel,
                 context_config=config.context,
                 parallel_config=config.parallel)

    model_config = LlamaConfig(**config.model.model_config)
    model_config.parallel_config = TransformerOpParallelConfig(**config.parallel_config)
    model_config.batch_size = len(inputs)
    model_config.use_past = args.use_past
    model_config.seq_length = args.seq_length
    if args.checkpoint_path and not config.use_parallel:
        model_config.checkpoint_name_or_path = args.checkpoint_path
    print(f"config is: {model_config}")

    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    # build model from config
    model = LlamaForCausalLM(model_config)

    # if use parallel, load distributed checkpoints
    if config.use_parallel:
        # find the sharded ckpt path for this rank
        ckpt_path = os.path.join(args.checkpoint_path, "rank_{}".format(os.getenv("RANK_ID", "0")))
        ckpt_path = get_last_checkpoint(ckpt_path)
        print("ckpt path: %s", str(ckpt_path))

        # shard model and load sharded ckpt
        warm_up_model = Model(model)
        warm_up_model.infer_predict_layout(ms.Tensor(np.ones(shape=(1, model_config.seq_length)), ms.int32))
        checkpoint_dict = load_checkpoint(ckpt_path)
        not_load_network_params = load_param_into_net(model, checkpoint_dict)
        print("Network parameters are not loaded: %s", str(not_load_network_params))

    inputs_ids = tokenizer(inputs, max_length=model_config.seq_length, padding="max_length")["input_ids"]
    outputs = model.generate(inputs_ids,
                             max_length=model_config.max_decode_length,
                             do_sample=model_config.do_sample,
                             top_k=model_config.top_k,
                             top_p=model_config.top_p)
    for output in outputs:
        print(tokenizer.decode(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='llama_7b', type=str,
                        help='which model to use.')
    parser.add_argument('--device_id', default=0, type=int,
                        help='set device id.')
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='set checkpoint path.')
    parser.add_argument('--use_past', default=True, type=str2bool,
                        help='whether use past.')
    parser.add_argument('--yaml_file', default="", type=str,
                        help='predict yaml path')
    parser.add_argument('--seq_length', default=512, type=int,
                        help='predict max length')
    args = parser.parse_args()
    main(args)

# 多batch输出
# <s>I love Beijing,because it is a city that is constantly changing. I have been living here for 10 years ...
# <s>LlaMa is a large-scale, open-source, multimodal, multilingual, multitask, and mulyimodal pretrained language model....
# <s>Huawei is a company that has been around for a long time. ...
```

以下为多卡运行自定义多batch推理的脚本

```bash
# >>> `run_predict.sh`文件
export RANK_TABLE_FILE=$1
CHECKPOINT_PATH=$2
YAML_FILE=$3
MODEL_TYPE=$4
# define variable
export RANK_SIZE=$5
export START_RANK=$6 # this server start rank
let END_RANK=START_RANK+RANK_SIZE # this server end rank

# run
for((i=${START_RANK}; i<${END_RANK}; i++))
do
    export RANK_ID=$((i-START_RANK))
    export DEVICE_ID=$i
    echo "Start distribute running for rank $RANK_ID, device $DEVICE_ID"
    python3 ./predict_custom.py --checkpoint_path $CHECKPOINT_PATH --yaml_file ${YAML_FILE} --model_type ${MODEL_TYPE} &> mindformers_$RANK_ID.log &
done
```

#### 单卡generate推理

1. 修改yaml文件

```python
use_parallel: False
```

2. 执行以下命令

```bash
# 以llama2-7b 单卡推理为例,checkpoint_path为权重文件，后缀为.ckpt
python predict_custom.py --yaml_file path/to/config_yaml --checkpoint_path path/to/checkpoint.ckpt --model_type llama2_7b
```

#### 多卡generate推理

```bash
# 以llama2-7b 2卡推理为例,此时的checkpoint必须是已经切分好的ckpt,shard_checkpoint_dir文件夹下为rank_{}的文件夹。
bash run_predict.sh RANK_TABLE_FILE path/to/shard_checkpoint_dir path/to/config_yaml llama2_7b 2 0
```

**注**：几卡推理就要在yaml配置中将相应的parallel_config 中的model_parallel置为2，其余置为1。

```python
use_parallel: True
# model config
parallel_config:
  data_parallel: 1
  model_parallel: 2  # 改为相应卡数。
  pipeline_stage: 1
  use_seq_parallel: False
  micro_batch_num: 1
  vocab_emb_dp: True
  gradient_aggregation_group: 4
# when model parallel is greater than 1, we can set micro_batch_interleave_num=2, that may accelerate the train process.
micro_batch_interleave_num: 1
```

### 基于pipeline的推理

以下为基于pipeline接口的自定义推理脚本，支持多卡推理。

```python
# predict_custom.py 文件
import argparse
import mindspore as ms
import numpy as np
import os
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train import Model

from mindformers import MindFormerConfig, LlamaConfig, TransformerOpParallelConfig, AutoTokenizer, LlamaForCausalLM, pipeline
from mindformers import init_context, ContextConfig, ParallelContextConfig
from mindformers.tools.utils import str2bool
from mindformers.trainer.utils import get_last_checkpoint


def main(args):
    """main function."""
    # 多输入
    inputs = ["I love Beijing, because",
              "LLaMA is a",
              "Huawei is a company that"]

    # set model config
    config = MindFormerConfig(args.yaml_file)

    # 初始化环境
    init_context(use_parallel=config.use_parallel,
                 context_config=config.context,
                 parallel_config=config.parallel)

    model_config = LlamaConfig(**config.model.model_config)
    model_config.parallel_config = TransformerOpParallelConfig(**config.parallel_config)
    model_config.use_past = args.use_past
    model_config.seq_length = args.seq_length
    if args.checkpoint_path and not config.use_parallel:
        model_config.checkpoint_name_or_path = args.checkpoint_path
    print(f"config is: {model_config}")

    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)

    model = LlamaForCausalLM(model_config)
    model.set_train(False)

    # if use parallel, load distributed checkpoints
    if config.use_parallel:
        # find the sharded ckpt path for this rank
        ckpt_path = os.path.join(args.checkpoint_path, "rank_{}".format(os.getenv("RANK_ID", "0")))
        ckpt_path = get_last_checkpoint(ckpt_path)
        print("ckpt path: %s", str(ckpt_path))

        # shard model and load sharded ckpt
        warm_up_model = Model(model)
        warm_up_model.infer_predict_layout(ms.Tensor(np.ones(shape=(1, model_config.seq_length)), ms.int32))
        checkpoint_dict = load_checkpoint(ckpt_path)
        not_load_network_params = load_param_into_net(model, checkpoint_dict)
        print("Network parameters are not loaded: %s", str(not_load_network_params))

    text_generation_pipeline = pipeline(task="text_generation", model=model, tokenizer=tokenizer)
    outputs = text_generation_pipeline(inputs,
                                       max_length=model_config.max_decode_length,
                                       do_sample=model_config.do_sample,
                                       top_k=model_config.top_k,
                                       top_p=model_config.top_p)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='llama_7b', type=str,
                        help='which model to use.')
    parser.add_argument('--device_id', default=0, type=int,
                        help='set device id.')
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='set checkpoint path.')
    parser.add_argument('--use_past', default=True, type=str2bool,
                        help='whether use past.')
    parser.add_argument('--yaml_file', default="", type=str,
                        help='predict yaml path')
    parser.add_argument('--seq_length', default=512, type=int,
                        help='predict max length')
    args = parser.parse_args()
    main(args)

# 单输出
# 'text_generation_text':['I love Beijing,because it is a city that is constantly changing. I have been living here for 10 years ...
# 'text_generation_text':['LlaMa is a large-scale, open-source, multimodal, multilingual, multitask, and multimodal pretrained language model....
# 'text_generation_text':['Huawei is a company that has been around for a long time. ...
```

以下为多卡运行自定义多batch推理的脚本

```bash
# >>> `run_predict.sh`文件
export RANK_TABLE_FILE=$1
CHECKPOINT_PATH=$2
YAML_FILE=$3
MODEL_TYPE=$4
# define variable
export RANK_SIZE=$5
export START_RANK=$6 # this server start rank
let END_RANK=START_RANK+RANK_SIZE # this server end rank

# run
for((i=${START_RANK}; i<${END_RANK}; i++))
do
    export RANK_ID=$((i-START_RANK))
    export DEVICE_ID=$i
    echo "Start distribute running for rank $RANK_ID, device $DEVICE_ID"
    python3 ./predict_custom.py --checkpoint_path $CHECKPOINT_PATH --yaml_file ${YAML_FILE} --model_type ${MODEL_TYPE} &> mindformers_$RANK_ID.log &
done
```

#### 单卡pipeline推理

与基于generate推理的单卡推理命令一致。

1. 修改yaml文件

```python
use_parallel: False
```

2. 执行以下命令

```bash
# 以llama2-7b 单卡推理为例,checkpoint_path为权重文件，后缀为.ckpt
python predict_custom.py --yaml_file path/to/config_yaml --checkpoint_path path/to/checkpoint.ckpt --model_type llama2_7b
```

#### 多卡pipeline推理

```bash
# 以llama2-7b 2卡推理为例,此时的checkpoint必须是已经切分好的ckpt
bash run_predict.sh RANK_TABLE_FILE path/to/shard_checkpoint_dir path/to/config_yaml llama2_7b 2 0
```

> 注：config_yaml的配置也要和基于generate的多卡推理一样将model_parallel 修改为相应卡数，而data_parallel 和 pipeline_stage设置为1。

### 基于run_mindformer分布式推理

#### 单卡推理

```bash
python run_mindformer.py --config configs/llama2/run_llama2_7b.yaml --run_mode predict --predict_data 'I love Beijing, because' --use_parallel False
```

**注**：推理时加入`vocab_file` 配置`tokenizer.model`路径；要提高推理速度，可在对应模型配置文件中进行如下配置，设置增量推理`use_past`为True。

```python
# model config
use_past: True          # 开启增量推理
pretrain_seqlen: 2048
extend_method: "None"
offset: 0
checkpoint_name_or_path: "llama2_7b"
repetition_penalty: 1
max_decode_length: 512
top_k: 3
top_p: 1
do_sample: False
max_new_tokens: 128      #设置最大生成长度

# tokenizer 配置
processor:
  return_tensors: ms
  tokenizer:
    vocab_file: "path/to/tokenizer.model"
```

#### 多卡推理

可参考[权重切分与合并](../feature_cards/Transform_Ckpt.md)中的分布式推理方法， 支持分布式推理，不支持batch推理

```bash
# 以llama2-7b 2卡推理为例,参考案例三，使用完整权重推理2卡
cd script
bash run_distribute.sh rank_table_2.json configs/llama2/run_llama2_7b.yaml [0,2] predict "I love beijing, because"
```

## Mindspore-Lite 推理

### 基本介绍

　　MindFormers 定位打造训练->微调->部署的端到端大模型工具套件，为了更好性能地部署已经微调训练好的大模型，我们利用MindSpore打造的推理引擎 [MindSpore_lite](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Flite)，为用户提供了开箱即用的推理部署方案，为用户提供端到端的大模型解决方案，帮助用户使能大模型业务。

　　Lite 推理大致分两步：权重转换导出 MindIR -> Lite 推理，接下来分别描述上述两个过程。

### MindIR 导出

  　　1. 以llama2_7b为例，修改模型相关的配置文件 configs/llama2/export_llama2_7b.yaml，其中需要关注这几项：

```yaml
# export
infer:
    prefill_model_path: "llama2_export/llama2_7b_prefill_seq512.mindir" # 保存mindir的位置
    increment_model_path: "llama2_export/llama2_7b_inc_seq512.mindir"   # 保存mindir的位置
    infer_seq_length: 512 # 需要保持跟 model-model_config-seq_length 一致

# ==== model config ====
model:
  model_config:
    seq_length: 512
    checkpoint_name_or_path: "/path/to/your/*.ckpt"
```

2. 执行export.py，完成模型转换

```bash
python mindformers/tools/export.py --config_path configs/llama2/export_llama2_7b.yaml
```

### 执行推理

1. 新建推理配置文件：lite.ini

   910A配置如下：

   ```ini
   [ascend_context]
   plugin_custom_ops=All
   provider=ge
   [ge_session_options]
   ge.externalWeight=1
   ge.exec.atomicCleanPolicy=1
   ge.event=notify
   ge.exec.staticMemoryPolicy=2
   ge.exec.precision_mode=must_keep_origin_dtype
   ```

   910B默认配置如下：

   ```ini
   [ascend_context]
   plugin_custom_ops=All
   provider=ge
   [ge_session_options]
   ge.exec.formatMode=1
   ge.exec.precision_mode=must_keep_origin_dtype
   ```

   910B 高性能配置如下：

   > 注: 高性能暂不支持llama2_7b

   ```ini
   [ascend_context]
   plugin_custom_ops=All
   provider=ge
   [ge_session_options]
   ge.externalWeight=1
   ge.exec.formatMode=1
   ge.exec.atomicCleanPolicy=1
   ge.event=notify
   ge.exec.staticMemoryPolicy=2
   ge.exec.precision_mode=must_keep_origin_dtype
   ```

2. 执行命令

```bash
python run_infer_main.py --device_id 0 --model_name llama2 --prefill_model_path llama2_export/llama2_7b_prefill_seq512_graph.mindir --increment_model_path llama2_export/llama2_7b_inc_seq512_graph.mindir --config_path lite.ini --is_sample_acceleration False --seq_length 512 --add_special_tokens True：
```

　　等待模型载入、编译后，出现：

```bash
Please enter your predict data:
```

　　输入：

```bash
I love Beijing, because
```

　　输出：

```bash
I love Beijing, because it is a city that is constantly changing. I have been living here for 10 years and I...
```
