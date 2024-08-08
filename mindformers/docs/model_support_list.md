# 模型支持列表

## NLP

### masked_language_modeling

|       模型 <br> model       | 模型规格<br>type  | 数据集 <br> dataset | 评估指标 <br> metric | 评估得分 <br> score |                        配置<br>config                        |
| :-------------------------: | ----------------- | :-----------------: | :------------------: | :-----------------: | :----------------------------------------------------------: |
| [bert](model_cards/bert.md) | bert_base_uncased |        wiki         |          -           |          -          | [configs](https://gitee.com/mindspore/mindformers/blob/r0.8/configs/bert) |

### [text_classification](task_cards/text_classification.md)

|                 模型 <br> model                  | 模型规格<br/>type                                          | 数据集 <br> dataset |   评估指标 <br> metric   | 评估得分 <br> score |                        配置<br>config                        |
| :----------------------------------------------: | ---------------------------------------------------------- | :-----------------: | :----------------------: | :-----------------: | :----------------------------------------------------------: |
| [txtcls_bert](task_cards/text_classification.md) | txtcls_bert_base_uncased<br/>txtcls_bert_base_uncased_mnli |   Mnli <br> Mnli    | Entity F1 <br> Entity F1 |   - <br>   84.80%   | [configs](https://gitee.com/mindspore/mindformers/blob/r0.8/configs/txtcls) |

### [token_classification](task_cards/token_classification.md)

|                  模型 <br> model                  | 模型规格<br/>type                                            | 数据集 <br> dataset  |   评估指标 <br> metric   | 评估得分 <br> score |                        配置<br>config                        |
| :-----------------------------------------------: | ------------------------------------------------------------ | :------------------: | :----------------------: | :-----------------: | :----------------------------------------------------------: |
| [tokcls_bert](task_cards/token_classification.md) | tokcls_bert_base_chinese<br/>tokcls_bert_base_chinese_cluener | CLUENER <br> CLUENER | Entity F1 <br> Entity F1 |  - <br>    0.7905   | [configs](https://gitee.com/mindspore/mindformers/blob/r0.8/configs/tokcls) |

### [question_answering](task_cards/question_answering.md)

|               模型 <br> model               | 模型规格<br/>type                                     |    数据集 <br> dataset     | 评估指标 <br> metric | 评估得分 <br> score  |                        配置<br>config                        |
| :-----------------------------------------: | ----------------------------------------------------- | :------------------------: | :------------------: | :------------------: | :----------------------------------------------------------: |
| [qa_bert](task_cards/question_answering.md) | qa_bert_base_uncased<br/>qa_bert_base_chinese_uncased | SQuAD v1.1 <br> SQuAD v1.1 | EM / F1 <br> EM / F1 | 80.74 / 88.33 <br> - | [configs](https://gitee.com/mindspore/mindformers/blob/r0.8/configs/qa) |

### translation

|     模型 <br> model     | 模型规格<br/>type | 数据集 <br> dataset | 评估指标 <br> metric | 评估得分 <br> score |                        配置<br>config                        |
| :---------------------: | ----------------- | :-----------------: | :------------------: | :-----------------: | :----------------------------------------------------------: |
| [t5](model_cards/t5.md) | t5_small          |        WMT16        |          -           |          -          | [configs](https://gitee.com/mindspore/mindformers/blob/r0.8/configs/t5) |

### [text_generation](task_cards/text_generation.md)

|                  模型 <br> model                   | 模型规格<br/>type                                            |              数据集 <br> dataset               |             评估指标 <br> metric             |                     评估得分 <br> score                      |                        配置<br>config                        |
| :------------------------------------------------: | ------------------------------------------------------------ | :--------------------------------------------: | :------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  [baichuan2](../research/baichuan2/baichuan2.md)   | baichuan2_7b <br/>baichuan2_13b  <br/>baichuan2_7b_lora <br/>baichuan2_13b_lora |                   - belle -                    |                  - <br>  -                   |                           - <br> -                           | [configs](https://gitee.com/mindspore/mindformers/blob/r0.8/research/baichuan2) |
|          [llama2](model_cards/llama2.md)           | llama2_7b <br/>llama2_13b <br/>llama2_7b_lora <br/>llama2_13b_lora <br/>llama2_70b |     alpaca <br> alpaca <br> alpaca <br> -      |         - <br>   - <br>   - <br>   -         |                 - <br>   - <br>   - <br>   -                 | [configs](https://gitee.com/mindspore/mindformers/blob/r0.8/configs/llama) |
|            [glm2](model_cards/glm2.md)             | glm2_6b<br/>glm2_6b_lora                                     |                ADGEN <br> ADGEN                | BLEU-4 / Rouge-1 / Rouge-2 / Rouge-l <br>  - | 7.47 / 30.78 / 7.07 / 24.77 <br> 7.23 / 31.06 / 7.18 / 24.23 | [configs](https://gitee.com/mindspore/mindformers/blob/r0.8/configs/glm2) |
|           [llama](model_cards/llama.md)            | llama_7b <br/>llama_13b <br/>llama_7b_lora                   |     alpaca <br> alpaca <br> alpaca <br> -      |         - <br>   - <br>   - <br>   -         |                 - <br>   - <br>   - <br>   -                 | [configs](https://gitee.com/mindspore/mindformers/blob/r0.8/configs/llama/run_llama_7b.yaml) |
|    [internlm](../research/internlm/internlm.md)    | Internlm_7b                                                  |             wikitext-2 <br> alpaca             |                  - <br>  -                   |                           - <br> -                           | [configs](https://gitee.com/mindspore/mindformers/blob/r0.8/research/internlm/run_internlm_7b.yaml) |
|          [ziya](../research/ziya/ziya.md)          | ziya_13b                                                     |                    - <br> -                    |                  - <br>  -                   |                           - <br> -                           | [configs](https://gitee.com/mindspore/mindformers/blob/r0.8/research/baichuan/run_ziya_13b.yaml) |
|    [baichuan](../research/baichuan/baichuan.md)    | baichuan_7b <br/>baichuan_13b                                |                    - <br> -                    |                  - <br>  -                   |                           - <br> -                           | [configs](https://gitee.com/mindspore/mindformers/blob/r0.8/research/baichuan/run_baichuan_7b.yaml) |
|           [bloom](model_cards/bloom.md)            | bloom_560m<br/>bloom_7.1b <br/>                              | alpaca <br> alpaca <br> alpaca <br>     alpaca |         - <br>   - <br>   - <br>   -         |                 - <br>   - <br>   - <br>   -                 | [configs](https://gitee.com/mindspore/mindformers/blob/r0.8/configs/bloom) |
|             [glm](model_cards/glm.md)              | glm_6b<br/>glm_6b_lora                                       |                ADGEN <br> ADGEN                | BLEU-4 / Rouge-1 / Rouge-2 / Rouge-l <br>  - |              8.42 / 31.75 / 7.98 / 25.28 <br> -              | [configs](https://gitee.com/mindspore/mindformers/blob/r0.8/configs/glm) |
|            [gpt2](model_cards/gpt2.md)             | gpt2_small <br/>gpt2_13b <br/>                               |   wikitext-2 <br> wikitext-2 <br> wikitext-2   |             - <br>   - <br>   -              |                     - <br>   - <br>   -                      | [configs](https://gitee.com/mindspore/mindformers/blob/r0.8/configs/gpt2) |
|      [pangualpha](model_cards/pangualpha.md)       | pangualpha_2_6_b<br/>pangualpha_13b                          |           悟道数据集 <br> 悟道数据集           |                  - <br>   -                  |                          - <br>   -                          | [configs](https://gitee.com/mindspore/mindformers/blob/r0.8/configs/pangualpha) |

## CV

### masked_image_modeling

|      模型 <br> model      | 模型规格<br/>type | 数据集 <br> dataset | 评估指标 <br> metric | 评估得分 <br> score |                        配置<br>config                        |
| :-----------------------: | ----------------- | :-----------------: | :------------------: | :-----------------: | :----------------------------------------------------------: |
| [mae](model_cards/mae.md) | mae_vit_base_p16  |     ImageNet-1k     |          -           |          -          | [configs](https://gitee.com/mindspore/mindformers/blob/r0.8/configs/mae/run_mae_vit_base_p16_224_800ep.yaml) |

### [image_classification](task_cards/image_classification.md)

|       模型 <br> model       | 模型规格<br/>type | 数据集 <br> dataset | 评估指标 <br> metric | 评估得分 <br> score |                        配置<br>config                        |
| :-------------------------: | ----------------- | :-----------------: | :------------------: | :-----------------: | :----------------------------------------------------------: |
|  [vit](model_cards/vit.md)  | vit_base_p16      |     ImageNet-1k     |       Accuracy       |       83.71%        | [configs](https://gitee.com/mindspore/mindformers/blob/r0.8/configs/vit/run_vit_base_p16_224_100ep.yaml) |
| [swin](model_cards/swin.md) | swin_base_p4w7    |     ImageNet-1k     |       Accuracy       |       83.44%        | [configs](https://gitee.com/mindspore/mindformers/blob/r0.8/configs/swin/run_swin_base_p4w7_224_100ep.yaml) |

## Multi-Modal

### [zero_shot_image_classification](task_cards/zero_shot_image_classification.md) (by [contrastive_language_image_pretrain](task_cards/contrastive_language_image_pretrain.md))

|        模型 <br> model        | 模型规格<br/>type                                            |                数据集 <br> dataset                 |                    评估指标 <br> metric                     |            评估得分 <br> score             |                        配置<br>config                        |
| :---------------------------: | ------------------------------------------------------------ | :------------------------------------------------: | :---------------------------------------------------------: | :----------------------------------------: | :----------------------------------------------------------: |
|  [clip](model_cards/clip.md)  | clip_vit_b_32<br/>clip_vit_b_16 <br/>clip_vit_l_14<br/>clip_vit_l_14@336 | Cifar100 <br> Cifar100 <br> Cifar100 <br> Cifar100 | Accuracy   <br>  Accuracy   <br>  Accuracy   <br>  Accuracy | 57.24% <br> 61.41% <br> 69.67% <br> 68.19% | [configs](https://gitee.com/mindspore/mindformers/blob/r0.8/configs/clip/run_clip_vit_b_32_pretrain_flickr8k.yaml) |
| [blip2](model_cards/blip2.md) | blip2_vit_g                                                  |              - <br> flickr30k <br> -               |                      - <br> ITM <br> -                      |              - <br> - <br> -               | [configs](https://gitee.com/mindspore/mindformers/blob/r0.8/configs/blip2/run_blip2_vit_g_qformer_pretrain.yaml) |

## LLM大模型能力支持一览

|    模型  \  特性    |    低参微调     |  边训边评  | 并行推理 | 流式推理 | Chat | 多轮对话 | Lite推理 |
| :-----------------: | :-------------: | :--------: | :------: | :------: | :--: | :------: | :------: |
|  BaiChuan2-7B/13B   |      Lora       |    PPL     |  dp/mp   |    √     |  √   |    √     |    √     |
|  Llama2-7B/13B/70B  |      Lora       |    PPL     |  dp/mp   |    √     |  √   |    √     |    √     |
|       GLM2-6B       | Lora/P-TuningV2 | Bleu/Rouge |  dp/mp   |    √     |  √   |    √     |    √     |
|        BILP2        |        ×        |     ×      |  dp/mp   |    √     |  ×   |    ×     |    ×     |
|   BaiChuan-7B/13B   |        ×        |    PPL     |  dp/mp   |    √     |  √   |    √     |    √     |
|    Llama-7B/13B     |      Lora       |    PPL     |  dp/mp   |    √     |  ×   |    ×     |    √     |
|     InternLM-7B     |      Lora       |    PPL     |  dp/mp   |    √     |  √   |    √     |    ×     |
|      ZiYa-13B       |        ×        |    PPL     |  dp/mp   |    √     |  ×   |    ×     |    ×     |
|   Bloom-560m/7.1B   |        ×        |    PPL     |  dp/mp   |    √     |  √   |    √     |    √     |
|       GLM-6B        |      Lora       | Bleu/Rouge |  dp/mp   |    √     |  √   |    √     |    √     |
|    GPT2-128m/13B    |      Lora       |    PPL     |  dp/mp   |    √     |  ×   |    ×     |    √     |
| PanGuAlpha-2.6B/13B |        ×        |    PPL     |  dp/mp   |    √     |  ×   |    ×     |    ×     |
