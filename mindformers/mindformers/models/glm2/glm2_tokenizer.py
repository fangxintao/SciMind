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
"""ChatGLM2 Tokenizer."""
import os
from typing import List, Optional, Union

from mindformers.mindformer_book import MindFormerBook
from mindformers.models.base_tokenizer import Tokenizer
from mindformers.tools.register import MindFormerModuleType, MindFormerRegister
from sentencepiece import SentencePieceProcessor

__all__ = ['ChatGLM2Tokenizer']


class SPTokenizer:
    """Tokenizer process special tokens."""

    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.unk_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

        special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"]
        self.special_tokens = {}
        self.index_special_tokens = {}
        for token in special_tokens:
            self.special_tokens[token] = self.n_words
            self.index_special_tokens[self.n_words] = token
            self.n_words += 1

    def tokenize(self, s: str, pair=None, add_special_tokens=True, **kwargs):
        # unused in this tokenizer.
        _, _, _ = pair, add_special_tokens, kwargs
        return self.sp_model.EncodeAsPieces(s)

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        assert isinstance(s, str)
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int], skip_special_tokens=False, clean_up_tokenization_spaces=None, **kwargs) -> str:
        # unused in this tokenizer.
        _, _, _ = skip_special_tokens, clean_up_tokenization_spaces, kwargs
        return self.sp_model.decode(t)

    def decode_tokens(self, tokens: List[str]) -> str:
        text = self.sp_model.DecodePieces(tokens)
        return text

    def convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token in self.special_tokens:
            return self.special_tokens[token]
        return self.sp_model.PieceToId(token)

    def convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.index_special_tokens or index in [self.eos_id, self.bos_id, self.pad_id] or index < 0:
            return ""
        return self.sp_model.IdToPiece(index)


@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class ChatGLM2Tokenizer(Tokenizer):
    """
    Construct a ChatGLM2 tokenizer. Based on byte-level Byte-Pair-Encoding.

    Args:
        vocab_file(str): The vocabulary file path.
        padding_side(str): ["left", "right"] Lower input text. Default False.
        **kwargs: Other kwargs that will be passed into the base class of the `Tokenizer`.

    Examples:
        >>> from mindformers import AutoTokenizer
        >>> tokenize = AutoTokenizer.from_pretrained('glm2_6b')
        >>> tokenize("你好")
        {'input_ids': [64790, 64792, 36474, 54591], 'attention_mask': [1, 1, 1, 1]}
        >>> from mindformers import ChatGLM2Tokenizer
        >>> tokenizer = ChatGLM2Tokenizer('tokenizer.model')
        >>> prompts = ["晚上睡不着应该怎么办"]
        >>> token_id = tokenizer(prompts)
        >>> input_ids = token_id['input_ids']
        >>> print(input_ids)
        [[64790, 64792, 30910, 32820, 54266, 31876, 35153]]
        >>> response = tokenizer.decode(input_ids)
        >>> print(response)
        ['晚上睡不着应该怎么办']


    Outputs:
        A dict contains the processed ids, attention_mask that specific by the member `MODEL_INPUT_NAME`
        of the subclass.
    """
    vocab_files_names = {"vocab_file": "tokenizer.model"}

    model_input_names = ["input_ids", "attention_mask", "position_ids"]
    _support_list = MindFormerBook.get_tokenizer_support_list()['glm2']

    def __init__(self,
                 vocab_file,
                 bos_token='<sop>',
                 eos_token='<eop>',
                 end_token='</s>',
                 mask_token='[MASK]',
                 gmask_token='[gMASK]',
                 pad_token='<pad>',
                 unk_token='<unk>',
                 **kwargs):
        super().__init__(bos_token=bos_token,
                         eos_token=eos_token,
                         end_token=end_token,
                         mask_token=mask_token,
                         gmask_token=gmask_token,
                         pad_token=pad_token,
                         unk_token=unk_token,
                         **kwargs)
        self.name = "ChatGLM2Tokenizer"

        self.vocab_file = vocab_file
        self.tokenizer = SPTokenizer(vocab_file)
        self.special_tokens = {
            "<bos>": self.tokenizer.bos_id,
            "<eos>": self.tokenizer.eos_id,
            "<pad>": self.tokenizer.pad_id
        }

        self._bos_token = bos_token
        self._eos_token = eos_token
        self._end_token = end_token
        self._mask_token = mask_token
        self._gmask_token = gmask_token

    def get_command(self, token):
        if token in self.special_tokens:
            return self.special_tokens[token]
        assert token in self.tokenizer.special_tokens, f"{token} is not a special token for {self.name}"
        return self.tokenizer.special_tokens[token]

    @property
    def pad_token_id(self):
        return self.get_command("<pad>")

    @property
    def eos_token_id(self):
        return self.get_command("<eos>")

    @property
    def vocab_size(self):
        return self.tokenizer.n_words

    def get_vocab(self):
        """ Returns vocab as a dict """
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def build_prompt(self, query, history=None):
        if history is None:
            history = []
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(
                i + 1, old_query, response)
        prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
        return prompt

    def tokenize(self, text, pair=None, add_special_tokens=True, **kwargs):
        """ Returns a tokenized string. """
        return self._tokenize(text)

    def _tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize(text)

    def _convert_token_to_id(self, token):
        return self.tokenizer.convert_token_to_id(token)

    def convert_tokens_to_ids(self, tokens: List[str]) -> list:
        """ Converts tokens to ids using the vocab. """
        ids = []
        for token in tokens:
            ids.append(self.tokenizer.convert_token_to_id(token))

        return ids

    def _decode(self,
                token_ids: Union[int, List[int]],
                skip_special_tokens: bool = False,
                clean_up_tokenization_spaces: bool = None,
                **kwargs) -> str:
        _, _ = skip_special_tokens, clean_up_tokenization_spaces
        tokens = []
        for token_id in token_ids:
            tokens.append(self._convert_id_to_token(token_id))
        return self.convert_tokens_to_string(tokens)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index >= self.vocab_size:
            raise IndexError(
                f"The token id {index} is out of the size of vocabulary, please check your tokenizer "
                f"and corresponding vocabulary files.")
        return self.tokenizer.convert_id_to_token(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.tokenizer.decode_tokens(tokens)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, self.vocab_files_names["vocab_file"]
            )
        else:
            vocab_file = save_directory

        with open(self.vocab_file, 'rb') as fin:
            proto_str = fin.read()

        with open(vocab_file, "wb") as writer:
            writer.write(proto_str)

        return (vocab_file,)

    def get_prefix_tokens(self):
        prefix_tokens = [self.get_command("[gMASK]"), self.get_command("sop")]
        return prefix_tokens

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        prefix_tokens = self.get_prefix_tokens()
        token_ids_0 = prefix_tokens + token_ids_0
        if token_ids_1 is not None:
            token_ids_0 = token_ids_0 + token_ids_1 + [self.get_command("<eos>")]
        return token_ids_0
