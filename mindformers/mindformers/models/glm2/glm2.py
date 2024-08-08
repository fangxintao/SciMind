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
"""ChatGLM2 model."""
import mindspore.ops as ops
import mindspore.common.dtype as mstype
import mindspore.ops.operations as P
from mindspore import Tensor, nn

import numpy as np
from mindformers.mindformer_book import MindFormerBook
from mindformers.modules import VocabEmbedding, EmbeddingOpParallelConfig
from mindformers.modules.layers import Linear
from mindformers.tools.register import MindFormerModuleType, MindFormerRegister
from mindformers.core.loss import CrossEntropyLoss
from mindformers.version_control import get_tril

from ..base_model import BaseModel
from .glm2_config import ChatGLM2Config
from .glm2_modules import precompute_rotary_emb_cache
from .glm2_transformer import ChatGLM2Transformer

__all__ = ['ChatGLM2ForConditionalGeneration', 'ChatGLM2Model']


class ChatGLM2Model(nn.Cell):
    r"""
    The backbone of ChatGLM2 network

    Args:
        config (GLMConfig): The config of network.
    """
    def __init__(self, config: ChatGLM2Config, **kwargs):
        super(ChatGLM2Model, self).__init__(config, **kwargs)
        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels
        self.seq_length = config.seq_length
        self.compute_dtype = config.compute_dtype
        self.use_past = config.use_past
        self.is_first_iteration = True
        # vocab embedding
        embed_parallel_config = EmbeddingOpParallelConfig()
        embed_parallel_config.data_parallel = config.parallel_config.data_parallel
        embed_parallel_config.model_parallel = config.parallel_config.model_parallel
        embed_parallel_config.vocab_emb_dp = False
        self.embedding = VocabEmbedding(vocab_size=config.vocab_size, embedding_size=config.hidden_size,
                                        parallel_config=embed_parallel_config)
        self.embedding.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

        # rotary embedding
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )
        self.rotary_pos_emb = precompute_rotary_emb_cache(
            seq_len=self.seq_length,
            dim=rotary_dim // 2
        )
        self.rotary_pos_emb = Tensor(self.rotary_pos_emb, config.compute_dtype)

        self.encoder = ChatGLM2Transformer(config)

        self.output_layer = Linear(config.hidden_size,
                                   config.vocab_size,
                                   has_bias=False,
                                   param_init_type=config.param_init_type,
                                   compute_dtype=config.compute_dtype)
        self.output_layer.shard(strategy_matmul=((config.parallel_config.data_parallel, 1),
                                                 (config.parallel_config.model_parallel, 1)))
        self.output_layer.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

        self.tril = get_tril()
        self.ones = P.Ones()
        self.less = P.Less()
        self.gather = P.Gather()
        self.expand_dims = P.ExpandDims()
        self.reshape = P.Reshape()
        self.mul = P.Mul()
        self.tile = ops.Tile()
        low_triangle = np.tril(np.ones((1, self.seq_length, self.seq_length)))
        self.low_triangle = Tensor(low_triangle, mstype.int32)

    def get_masks(self, batch_size, padding_mask=None, input_position=None):
        """Get attention mask."""
        # [1, seq_length, seq_length] -> [batch_size, seq_length, seq_length]
        low_triangle = self.tile(self.low_triangle, (batch_size, 1, 1))
        if padding_mask is not None:
            low_triangle = self.mul(low_triangle, self.expand_dims(padding_mask, 1))
        if self.use_past and padding_mask is not None:
            low_triangle -= self.expand_dims(padding_mask, -1) - 1
        attention_mask = self.less(low_triangle, 0.5)
        if self.use_past and not self.is_first_iteration:
            # [bs, 1, seq_len] for incremental infer
            attention_mask = self.gather(attention_mask.view(-1, self.seq_length), input_position, 0)
        # [bs, 1, seq_len, seq_len] for normal, [bs, 1, 1, seq_len] for incremental infer
        attention_mask = self.reshape(attention_mask, (batch_size, 1, -1, self.seq_length))
        return attention_mask

    def construct(self, input_ids, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None, full_attention_mask=None):
        """ChatGLM2 model."""
        _ = position_ids
        batch_size, _ = input_ids.shape
        if input_embeds is None:
            input_embeds, _ = self.embedding(input_ids)  # (bs, seq_len, hs)

        if full_attention_mask is None:
            # (bs, 1, seq_len, seq_len)
            full_attention_mask = self.get_masks(batch_size, attention_mask, input_position)

        # (sen length, kv_channels // 4, 2)
        rotary_pos_emb = self.rotary_pos_emb
        if self.use_past and not self.is_first_iteration and batch_valid_length is not None:
            # only take [bs, 1, kv_channels // 4, 2]
            batch_gather_position = batch_valid_length.view(-1, 1) - 1  # [bs, seq_len=1]
            rotary_pos_emb = self.gather(rotary_pos_emb, batch_gather_position, 0)

        # Run encoder.
        hidden_states = self.encoder(
            input_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
            init_reset=init_reset, batch_valid_length=batch_valid_length)

        return hidden_states


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class ChatGLM2ForConditionalGeneration(BaseModel):
    r"""
    Provide gpt training loss or logits through network.

    Args:
        config (GLMConfig): The config of ChatGLM2Model.

    Returns:
        Tensor, the loss or logits of the network.
    """
    _support_list = MindFormerBook.get_model_support_list()['glm2']

    def __init__(self, config: ChatGLM2Config, **kwargs):
        super(ChatGLM2ForConditionalGeneration, self).__init__(config, **kwargs)
        self.max_seq_len = config.max_length
        self.transformer = ChatGLM2Model(config=config)
        self.cast = P.Cast()
        self.gather = P.Gather()
        self.is_first_iteration = True
        self.loss = CrossEntropyLoss(parallel_config=config.parallel_config)
        self.gmask = config.gmask_token_id
        self.bos_token_id = config.bos_token_id
        self.use_past = config.use_past
        self.is_first_iteration = True
        self.not_equal = P.NotEqual()
        self.load_checkpoint(config)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """prepare inputs for generation."""
        input_position = kwargs.get("current_index", None)
        if input_position is not None:
            input_position = Tensor(input_position, mstype.int32)
        return {
            "input_ids": Tensor(input_ids, mstype.int32),
            "input_position": input_position
        }

    def construct(self, input_ids=None, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None):
        """ChatGLM2 for conditional generation model."""
        # input_ids: (bs, seq_len)
        # position_ids: (bs, seq_len)
        # attention_mask: (bs, seq_len)
        hidden_states = self.transformer(
            input_ids=input_ids,
            input_position=input_position,
            position_ids=position_ids,
            attention_mask=attention_mask,
            input_embeds=input_embeds,
            init_reset=init_reset,
            batch_valid_length=batch_valid_length,
        )
        lm_logits = self.transformer.output_layer(hidden_states)

        if labels is not None:
            logits = self.cast(lm_logits, mstype.float32)
            logits_shape = logits.shape
            labels = labels.reshape((-1,))
            logits = logits.reshape((-1, logits_shape[-1]))
            input_mask = self.not_equal(labels, -100).astype(logits.dtype)
            input_mask = input_mask.reshape((-1,))
            loss = self.loss(logits, labels, input_mask)
            return loss

        lm_logits = lm_logits.reshape((-1, lm_logits.shape[-1]))
        if (not self.use_past or self.is_first_iteration) and input_position is not None:
            lm_logits = self.gather(lm_logits, input_position, 0)
        return (lm_logits,)
