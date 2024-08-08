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
"""ChatGLM2 config"""
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.modules.transformer.transformer import default_transformer_config
from ..base_config import BaseConfig
from ..utils import convert_mstype
from ...mindformer_book import MindFormerBook

__all__ = ['ChatGLM2Config']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class ChatGLM2Config(BaseConfig):
    """
    ChatGLM2 model config class.
    """
    _support_list = MindFormerBook.get_config_support_list()['glm2']

    def __init__(self,
                 batch_size=1,   # only for incremental infer
                 num_layers=28,
                 padded_vocab_size=65024,
                 hidden_size=4096,
                 ffn_hidden_size=13696,
                 kv_channels=128,
                 num_attention_heads=32,
                 seq_length=2048,
                 hidden_dropout=0.0,
                 attention_dropout=0.0,
                 layernorm_epsilon=1e-5,
                 rmsnorm=True,
                 apply_residual_connection_post_layernorm=False,
                 post_layer_norm=True,
                 add_bias_linear=False,
                 add_qkv_bias=True,
                 bias_dropout_fusion=True,
                 multi_query_attention=True,
                 multi_query_group_num=2,
                 apply_query_key_layer_scaling=True,
                 attention_softmax_in_fp32=True,
                 fp32_residual_connection=False,
                 quantization_bit=0,
                 pre_seq_len=None,
                 prefix_projection=False,
                 param_init_type: str = "float16",
                 compute_dtype: str = "float16",
                 layernorm_compute_type: str = "float32",
                 use_past=False,
                 eos_token_id=2,
                 pad_token_id=0,
                 repetition_penalty=1.0,
                 parallel_config=default_transformer_config,
                 **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.vocab_size = padded_vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.rmsnorm = rmsnorm
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        self.param_init_type = convert_mstype(param_init_type)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.layernorm_compute_type = convert_mstype(layernorm_compute_type)
        self.use_past = use_past
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.repetition_penalty = repetition_penalty
        self.parallel_config = parallel_config
