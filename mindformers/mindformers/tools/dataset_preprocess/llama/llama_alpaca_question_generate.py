# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
transform wikitext-2, wikitext-103, lambada, openwebtext dataset to mindrecord.
"""
import argparse
import json
import os
import re
import numpy as np

np.set_printoptions(threshold=np.inf)
from mindspore.mindrecord import FileWriter

from mindformers.models.llama.llama_tokenizer import LlamaTokenizer

from conversation import get_default_conv_template

IGNORE_TOKEN_ID = -100


def chunks(lst, n):
    """ yield n sized chunks from list"""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def package_file(it, n):
    """ package multiple files"""
    stop = False
    while not stop:
        batch = []
        for _ in range(n):
            try:
                batch.append(next(it))
            except StopIteration:
                stop = True
        if not batch:
            break
        yield batch


def clean_wikitext(string):
    """ cleaning wikitext dataset"""
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" "+chr(176)+" ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def preprocess(sources):
    """conversation preprocess."""
    conv = get_default_conv_template("vicuna").copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    return conversations
    # print(conversations[:4])
    # quit()
    # sep = conv.sep + conv.roles[1] + ": "
    # # Tokenize conversations
    # input_ids = []
    # targets = []
    # # attention_mask = []
    # for idx, conversation in enumerate(conversations):
    #     print(conversation)
    #     quit()
    #     if idx==2:
    #         quit()
    #     rounds = conversation.split(conv.sep2)
    #     print(rounds)
    #     quit()
    #     ids = [tokenizer.bos_token_id]
    #     mask = [1]
    #     for i, rou in enumerate(rounds):
    #         if rou == "":
    #             break
    #         conv_out = tokenizer(rou)
    #         ids.extend(conv_out['input_ids'][1:])
    #         mask.extend(conv_out['attention_mask'][1:])
    #     d = {'input_ids': ids, 'attention_mask': mask}
    #     # pylint: disable=W0212
    #     # print("len(d)", len(d["input_ids"]))
    #     d = tokenizer._pad(d, max_length=seq_length, padding_strategy='max_length')
    #     if len(d['input_ids']) > seq_length+1:
    #         print(idx, len(d['input_ids']), flush=True)
    #         continue
    #     if idx % 1000 == 0:
    #         print(idx)
    #     input_ids.append(d['input_ids'])
    #     # attention_mask.append(d['attention_mask'])

    #     target = np.array(d['input_ids'])
    #     total_len = int(np.not_equal(target, tokenizer.pad_token_id).sum())
    #     cur_len = 1
    #     target[:cur_len] = IGNORE_TOKEN_ID
    #     for i, rou in enumerate(rounds):
    #         if rou == "":
    #             break
    #         parts = rou.split(sep)
    #         if len(parts) != 2:
    #             break
    #         parts[0] += sep
    #         round_len = len(tokenizer(rou)['input_ids']) - 1
    #         instruction_len = len(tokenizer(parts[0])['input_ids']) - 3

    #         target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID

    #         cur_len += round_len
    #     target[cur_len:] = IGNORE_TOKEN_ID

    #     if cur_len < seq_length:
    #         if cur_len != total_len:
    #             target[:] = IGNORE_TOKEN_ID
    #     targets.append(target.tolist())

    # input_ids = np.array(input_ids)
    # targets = np.array(targets)

    # return dict(
    #     input_ids=input_ids,
    #     labels=targets,
    # )


class SupervisedDataset:
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer, seq_length):
        super(SupervisedDataset, self).__init__()

        sources = [example["conversation"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, seq_length)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i]
        )


def tokenize_wiki(tokenizer, file_path, seq_length, repeat):
    """tokenize wikitext-2/wikitext-103 dataset"""
    content = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for para in clean_wikitext(f.read()).split("\n\n"):
            if para and para.strip().startswith('=') is False:
                content += tokenizer(para)['input_ids']
    content_out = []
    for _ in range(repeat):
        content_out.extend(content)
    content = content_out
    for chunk in chunks(content, seq_length):
        sample = {}
        if len(chunk) == seq_length:
            sample['input_ids'] = np.array(chunk, dtype=np.int32)
            yield sample


def tokenize_qa(tokenizer, file_path, seq_length):
    raw_data = json.load(open(file_path, "r"))
    dataset_cls = SupervisedDataset(raw_data, tokenizer, seq_length)
    for i in range(len(dataset_cls)):
        yield dataset_cls[i]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_glob', type=str, default='/mnt/luolan/wikitext-2/wiki.train.tokens')
    parser.add_argument('--output_file', type=str, default='./dataset/wiki2048/wiki2048')
    args = parser.parse_args()

    out_dir, out_file = os.path.split(os.path.abspath(args.output_file))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    raw_data = json.load(open(args.input_glob, "r"))
    sources = [example["conversation"] for example in raw_data]
    conversations = preprocess(sources)

    with open(args.output_file+"_question", "w") as f:
        for conv in conversations:
            split_idx = conv.index("ASSISTANT: ") + len("ASSISTANT: ")
            print(conv[:split_idx])
            f.write(conv[:split_idx]+"\r")

    with open(args.output_file+"_answer", "w") as f:
        for conv in conversations:
            split_idx = conv.index("ASSISTANT: ") + len("ASSISTANT: ")
            print(conv[split_idx:])
            f.write(conv[split_idx:]+"\r")

    print("Transform finished, output files refer: {}".format(args.output_file))
