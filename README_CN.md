## SciMind: A Multimodal Mixture-of-Experts Model for Advancing Pharmaceutical Sciences

这个仓库包含了:
- [SciMind: A Multimodal Mixture-of-Experts Model for Advancing Pharmaceutical Sciences](https://openreview.net/forum?id=xbyPquFUB4)的代码实现

#### 目录
- 介绍
- 模型和数据
- 环境要求
- 仓库介绍
- 快速开始
- 模型微调
- 引用

#### 介绍

mindspore框架下，基于llama2模型开发的多模态混合专家大模型

与llama2对比改进点：

- 将SMILES分子，蛋白质分子，核酸分子各自单独视作一种模态，构造特殊token并扩充进词表
- 对llama的前馈层进行前向专家划分，在进行前向传播时，针对不同模态的token可以选取不同数量的专家

参考图片如下:

![这是图片](https://github.com/fangxintao/SciMind/blob/main/picture/scimind.png?raw=true "SciMind")

####  模型和数据
##### 模型
- 没经过下游任务微调 : [*Scimind.ckpt*]()
- 经过LPM-24 text2smiles数据微调 : [*Scimind-text2smiles.ckpt*]()
- 经过LPM-24 smiles2text 数据微调: [*Scimind-smiles2text.ckpt*]()

##### 数据
- LPM-24数据请参考 *Repository Introduction*
- 其他数据:

| Dataset  |
| ------------- |
| ChEBI  |
| MoA |
| Protein-oriented_Instructions |

#### 环境要求

1. 硬件：Ascend 910A/B
2. MindSpore：2.2
3. MindFormers版本：0.8.0
4. 其他依赖库参考requirement.txt
#### 仓库介绍

1. 模型具体实现为：*./mindformers/model/llama*\
   **llama**\
   ├── __init__.py\
   ├── convert_weight.py         # 权重转换脚本\
   ├── llama.py                  # 模型实现\
   ├── llama_config.py           # 模型配置项\
   ├── llama_layer.py            # llama网络层定义,增加了混合专家系统\
   ├── llama_processor.py        # llama预处理\
   ├── llama_tokenizer.py        # tokenizer，扩充了词表\
   └── llama_transformer.py      # transformer层实现
2. 模型参数配置文件：*./configs/llama*\
   **llama**\
   ├── run_llama_7b_finetune.yaml         # 7b模型全量微调启动配置
3. LPM24数据：*./LPM-24-data*\
   **LPM-24-data**\
   ├── smiles_to_text
   
   > ├── LPM-24_train_qa_conversation.json\
   > ├── LPM-24_test_qa_conversation.json\
   > ├── LPM-24_test.json\
   > ├── LPM-24_train.json
   
   ├── smiles_to_text_generate
   
   > ├── LPM-24_smiles2text_generate.txt # 处理好的直接用于推理\
   > ├── eval-text.txt  # LPM-24原始验证集\
   > ├── eval-text-special_token.txt  # 特殊token处理过的LPM-24数据集
   
   ├── text_to_smiles
   
   > ├── LPM-24_train_qa_conversation.json\
   > ├── LPM-24_test_qa_conversation.json\
   > └── LPM-24_test.json\
   > └── LPM-24_test.json
   
   ├── text_to_smiles_generate
   
   > ├── LPM-24_text2smile_generate.txt # 处理好的直接用于推理\
   > ├── eval-molgen.txt # LPM-24原始验证集

#### 开源模型下载
- 未经过微调初始模型：*Scimind.ckpt*
- LPM-24 text2smiles任务微调好的模型：*Scimind-text2smiles.ckpt*
- LPM-24 smiles2text任务微调好的模型：*Scimind-smiles2text.ckpt*

#### 快速开始

要快速使用我们的模型，你需要先在参数文件中调整两个参数，一个是你的模型路径参数：load_checkpoint，另一个是为了加速推理的参数use_past

- Configuration(*.mindformers/configs/llama2/run_llama2_7b_finetune.yaml*)
  - *load_checkpoint* : /path/to/your/model file
  - *use_past* : True

然后，你需要准备正确格式的数据并运行脚本

#####1. 数据预处理

> - For smiles2text task generation, construct data like:
>    **./LPM-24-data/smiles2text_generate/LPM-24_smile2text_generate.txt**

> - For text2smiles task generation, construct data like:
>   **./LPM-24-data/text2smiles_generate/LPM-24_text2smiles_generate.txt**

#####2. 模型推理

-  运行 *run_mindformer.py* 脚本 (运行在单卡上)

```
> python run_mindformer.py\
--config {config_file_path}\
--run_mode predict\
--predict_data {input_data_path}\
--use_parallel False --device_id 0\
--save_file {output_path}
```
例子：

 ```
 python run_mindformer.py\
 --config configs/llama2/run_llama2_7b_finetune.yaml\
 --run_mode predict\
 --predict_data ./LPM-24-data/smiles2text_generate/LPM-24_smile2text_generate.txt\
 --use_parallel False\
 --device_id 0\
 --save_file ./LPM-24-data/smiles2text_generate/output_LPM_smiles2text.txt
 ```

#### 模型微调

1. 数据准备（以LPM-24数据集smile2text任务为例），

- 将原始LPM-24数据处理为对话格式的文本，保存为json文件，json参考格式：

*./LPM-24-data/text2smiles/LPM-24_train_qa_conversation.json*

- 将json文件再进行处理得到mindrecord格式文件用于模型微调，使用以下预处理脚本生成mindrecord格式的训练数据
  python llama_preprocess_2.py\
  --dataset_type qa\
  --input_glob {json/data/path} # 例如./LPM-24-data/text2smiles/LPM-24_train_qa_conversation.json\
  --model_file ./mindformers/checkpoint_download/llama2/tokenizer.model\
  --seq_length 2048\
  --output_file {output_path}example.mindrecord

2. 微调(微调时默认设置为8卡)

微调模型需要在参数中设置模型的位置以及mindrecord格式数据的位置

- 参数设置(*.mindformers/configs/llama2/run_llama2_7b_finetune.yaml*)
  - *load_checkpoint* : /path/to/your/model file
  - *dataset_dir* : /path/to/your/mindrecord file

微调脚本路径：./mindformers/scripts

```
bash run_distribute.sh /user/config/jobstart_hccl.json ../configs/llama2/run_llama2_7b_finetune.yaml [0,8] train
```

#### 引用

如果需要引用我们的工作：
```
@inproceedings{
xiong2024scimind,
title={SciMind: A Multimodal Mixture-of-Experts Model for Advancing Pharmaceutical Sciences},
author={Zhaoping Xiong and Xintao Fang and Haotian CHU and Xiaozhe Wan and Liwei Liu and Yameng Li and Wenkai Xiang and Mingyue Zheng},
booktitle={ACL 2024 Workshop Language + Molecules},
year={2024},
url={https://openreview.net/forum?id=xbyPquFUB4}
}
```