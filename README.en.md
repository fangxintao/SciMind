# SciMind

#### Description
A multimodal mixture of experts large model developed based on the Llama2 model within the MindSpore framework

Improvements compared to Llama2:

- Treat SMILES molecules, protein molecules, and nucleic acid molecules each as a separate modality, construct special tokens, and expand them into the vocabulary.

- Perform forward expert partitioning on the feedforward layer of Llama, and during forward propagation, different numbers of experts can be selected for tokens of different modalities

![Picture](https://gitee.com/gitee_fangxintao/sci-mind/raw/master/picture/scimind.png "SciMind")

#### Environmental Requirements

1. Hardware: Ascend 910A/B
2. MindSpore：2.2
3. MindFormers version: 0.8.0
4. Refer to requirement.txt for other dependencies

#### Repository Introduction

1. Files related to the model are located in: *./mindformers/model/llama*
   **llama**\
   ├── __init__.py\
   ├── convert_weight.py         # Weight conversion script\
   ├── llama.py                  # Model\
   ├── llama_config.py           # Model configuration \
   ├── llama_layer.py            # Llama network layer definition, with the addition of a mixture of experts system\
   ├── llama_processor.py        # llama preprocess\
   ├── llama_tokenizer.py        # The tokenizer has an expanded vocabulary\
   └── llama_transformer.py      # transformer layer
2. The model parameter configuration file is located in: *./configs/llama*
   **llama**\
   ├── run_llama_7b_finetune.yaml         # 7b model full fine-tuning startup configuration
3. LPM24 data: *./LPM-24-data*
   **LPM-24-data**\
   ├── smiles_to_text
   
   > ├── LPM-24_train_qa_conversation.json\
   > ├── LPM-24_test_qa_conversation.json\
   > ├── LPM-24_test.json\
   > ├── LPM-24_train.json
   
   ├── smiles_to_text_generate
   
   > ├── LPM-24_smiles2text_generate.txt # Already processed and ready for direct inference\
   > ├── eval-text.txt  # LPM-24 original validation set\
   > ├── eval-text-special_token.txt  # LPM-24 dataset processed with special tokens
   
   ├── text_to_smiles
   
   > ├── LPM-24_train_qa_conversation.json\
   > ├── LPM-24_test_qa_conversation.json\
   > └── LPM-24_test.json\
   > └── LPM-24_test.json
   
   ├── text_to_smiles_generate
   
   > ├── LPM-24_text2smile_generate.txt # Already processed and ready for direct inference\
   > ├── eval-molgen.txt # LPM-24 original validation set

#### Open-source model download
- Pre-trained model without fine-tuning: *Scimind.ckpt*
- LPM-24 text2smiles finetuned model: *Scimind-text2smiles.ckpt*
- LPM-24 smiles2text finetuned model: *Scimind-smiles2text.ckpt*

#### Model fine-tuning instructions

1. Data preprecessing (Taking the smile2text task of the LPM-24 dataset as an example)

- Process the original LPM-24 data into a conversational text format and save it as a JSON file. JSON reference format:

*./LPM-24-data/text2smiles/LPM-24_train_qa_conversation.json*

- Further process the JSON file to obtain a MindRecord format file for model fine-tuning. Use the following preprocessing script to generate MindRecord format training data
  python llama_preprocess_2.py\
  --dataset_type qa\
  --input_glob {json/data/path} # Example: ./LPM-24-data/text2smiles/LPM-24_train_qa_conversation.json\
  --model_file ./mindformers/checkpoint_download/llama2/tokenizer.model\
  --seq_length 2048\
  --output_file {output_path}example.mindrecord

2. Fine-tuning (Fine-tuning defaults to requiring 8 NPUs)

Specify the model location and the MindRecord format data location in the configuration file

- Configuration(*.mindformers/configs/llama2/run_llama2_7b_finetune.yaml*)
  - *load_checkpoint* : /path/to/your/model file
  - *dataset_dir* : /path/to/your/mindrecord file

#### Model inference instructions

During fine-tuning, need to configure the model path and parameters for accelerating inference in the parameter configuration file

- Configuration(*.mindformers/configs/llama2/run_llama2_7b_finetune.yaml*)
  - *load_checkpoint* : /path/to/your/model file
  - *use_past* : True

#### Contribution

1.  Fork the repository
2.  Create Feat_xxx branch
3.  Commit your code
4.  Create Pull Request


#### Gitee Feature

1.  You can use Readme\_XXX.md to support different languages, such as Readme\_en.md, Readme\_zh.md
2.  Gitee blog [blog.gitee.com](https://blog.gitee.com)
3.  Explore open source project [https://gitee.com/explore](https://gitee.com/explore)
4.  The most valuable open source project [GVP](https://gitee.com/gvp)
5.  The manual of Gitee [https://gitee.com/help](https://gitee.com/help)
6.  The most popular members  [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
