# SciMind: A Multimodal Mixture-of-Experts Model for Advancing Pharmaceutical Sciences

The repo contains:
- The official implementation of [SciMind: A Multimodal Mixture-of-Experts Model for Advancing Pharmaceutical Sciences](https://openreview.net/forum?id=xbyPquFUB4)

## Content
- Introduction
- Model and Data
- Environment
- Repository Introduction
- Quick start
- Fine-tuning instructions
- Citation

## Introduction
SciMind is a multimodal mixture of experts large model developed based on the Llama2 model within the MindSpore framework

Improvements compared to Llama2:

- Treat SMILES molecules, protein molecules, and nucleic acid molecules each as a separate modality, construct special tokens, and expand them into the vocabulary.

- Perform forward expert partitioning on the feedforward layer of Llama, and during forward propagation, different numbers of experts can be selected for tokens of different modalities

![Picture](https://github.com/fangxintao/SciMind/blob/main/picture/scimind.png?raw=true "SciMind")

##  Model and Data
### model
- Pre-trained model without fine-tuning is available at : [*Scimind.ckpt*]()
- LPM-24 text2smiles finetuned model is available at : [*Scimind-text2smiles.ckpt*]()
- LPM-24 smiles2text finetuned model is available at : [*Scimind-smiles2text.ckpt*]()

### data
- LPM-24 dataset refer to *Repository Introduction*
- other data:
	
| Dataset  |
| ------------- |
| ChEBI  |
| MoA |
| Protein-oriented_Instructions |

## Environment

1. Hardware: Ascend 910A/B
2. MindSpore：2.2
3. MindFormers version: 0.8.0
4. Refer to requirement.txt for other dependencies

## Repository Introduction

1. Files related to the model are located in: *./mindformers/model/llama*\
   **llama**\
   ├── __init__.py\
   ├── convert_weight.py         # Weight conversion script\
   ├── llama.py                  # Model\
   ├── llama_config.py           # Model configuration \
   ├── llama_layer.py            # Llama network layer definition, with the addition of a mixture of experts system\
   ├── llama_processor.py        # llama preprocess\
   ├── llama_tokenizer.py        # The tokenizer has an expanded vocabulary\
   └── llama_transformer.py      # transformer layer
2. The model parameter configuration file is located in: *./configs/llama*\
   **llama**\
   ├── run_llama_7b_finetune.yaml         # 7b model full fine-tuning startup configuration
3. LPM24 data: *./LPM-24-data*\
   **LPM-24-data**\
   ├── smiles_to_text
   + ├── LPM-24_train_qa_conversation.json\
   + ├── LPM-24_test_qa_conversation.json\
   + ├── LPM-24_test.json\
   + ├── LPM-24_train.json
   
   ├── smiles_to_text_generate
   
   + ├── LPM-24_smiles2text_generate.txt # Already processed and ready for direct inference\
   + ├── eval-text.txt  # LPM-24 original validation set\
   + ├── eval-text-special_token.txt  # LPM-24 dataset processed with special tokens
   
   ├── text_to_smiles
   
   + ├── LPM-24_train_qa_conversation.json\
   + ├── LPM-24_test_qa_conversation.json\
   + └── LPM-24_test.json\
   + └── LPM-24_test.json
   
   ├── text_to_smiles_generate
   
   + ├── LPM-24_text2smile_generate.txt # Already processed and ready for direct inference\
   + ├── eval-molgen.txt # LPM-24 original validation set

## Quick start

For quick use our model, first you need to configure the model path and parameters for accelerating inference in the parameter configuration file

- Configuration(*.mindformers/configs/llama2/run_llama2_7b_finetune.yaml*)
  > -- *load_checkpoint* : /path/to/your/model file
  > --*use_past* : True

then, you need to prepare the correct format data and run scripts

### 1. data preprocessing

> - For smiles2text task generation, construct data like:
>    **./LPM-24-data/smiles2text_generate/LPM-24_smile2text_generate.txt**

> - For text2smiles task generation, construct data like:
>   **./LPM-24-data/text2smiles_generate/LPM-24_text2smiles_generate.txt**

### 2. Generation

-  run script *run_mindformer.py* (run by one NPU)

```
> python run_mindformer.py\
--config {config_file_path}\
--run_mode predict\
--predict_data {input_data_path}\
--use_parallel False --device_id 0\
--save_file {output_path}
```
example：

 ```
 python run_mindformer.py\
 --config configs/llama2/run_llama2_7b_finetune.yaml\
 --run_mode predict\
 --predict_data ./LPM-24-data/smiles2text_generate/LPM-24_smile2text_generate.txt\
 --use_parallel False\
 --device_id 0\
 --save_file ./LPM-24-data/smiles2text_generate/output_LPM_smiles2text.txt
 ```


## Fine-tuning instructions

### 1. Data preprecessing (Taking the smile2text task of the LPM-24 dataset as an example)

- Process the original LPM-24 data into a conversational text format and save it as a JSON file. JSON reference format:

	- *./LPM-24-data/text2smiles/LPM-24_train_qa_conversation.json*

- Further process the JSON file to obtain a MindRecord format file for model fine-tuning. Use the following preprocessing script to generate MindRecord format training data
```
  python llama_preprocess_2.py\
  --dataset_type qa\
  --input_glob {json/data/path} # Example: ./LPM-24-data/text2smiles/LPM-24_train_qa_conversation.json\
  --model_file ./mindformers/checkpoint_download/llama2/tokenizer.model\
  --seq_length 2048\
  --output_file {output_path}example.mindrecord
```

### 2. Fine-tuning (Fine-tuning defaults to requiring 8 NPUs)

Specify the model location and the MindRecord format data location in the configuration file

- Configuration(*.mindformers/configs/llama2/run_llama2_7b_finetune.yaml*)
  - *load_checkpoint* : /path/to/your/model file
  - *dataset_dir* : /path/to/your/mindrecord file

Turn to the path：*./mindformers/scripts* , then run shell script:

```
bash run_distribute.sh /user/config/jobstart_hccl.json ../configs/llama2/run_llama2_7b_finetune.yaml [0,8] train
```
## Citation

If you use Scimind in your work, please  cite our paper:
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

