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
"""ChatGLM2 Transformer."""
import math
import numpy as np

import mindspore.ops.functional as F
import mindspore.ops.operations as P
from mindspore import Parameter, Tensor, nn, ops
from mindspore import dtype as mstype

from mindformers.modules import LayerNorm
from mindformers.modules.layers import Linear
from mindformers.version_control import get_dropout

from .glm2_config import ChatGLM2Config
from .glm2_modules import ChatGLM2MLP, ChatGLM2RMSNorm


class CoreAttention(nn.Cell):
    """ChatGLM2 core attention."""
    def __init__(self, config: ChatGLM2Config, layer_number):
        super(CoreAttention, self).__init__()
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        projection_size = config.kv_channels * config.num_attention_heads

        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        # Strided linear layer.
        self.attention_dropout = get_dropout(config.attention_dropout)

        parallel_config = config.parallel_config

        self.batch_matmul_q_k = P.BatchMatMul(transpose_b=True)
        self.batch_matmul_q_k.shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
             (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
        self.batch_matmul = P.BatchMatMul().shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
             (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
        self.softmax = nn.Softmax(axis=-1)

        self.merger_head_transpose = P.Transpose().shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))
        self.reshape = P.Reshape()

        self.compute_dtype = config.compute_dtype

    def construct(self, query_layer, key_layer, value_layer, attention_mask):
        """
        calculate attention function
        """
        # query_layer [b, heads, seq, hidden_size_per_head]
        # key_layer [b, heads, seq, hidden_size_per_head]
        # value_layer # [bs, heads, seq_len, hidden_size_per_head]

        # seqlen, batch, head, hidden_size

        if self.apply_query_key_layer_scaling:
            query_layer = query_layer / self.norm_factor

        # ===================================
        # Raw attention scores. [b, heads, s, s]
        # ===================================
        # [b, heads, seq_q, hidden_size_per_head] × [b, heads, seq_k, hidden_size_per_head]^T -> [b, heads, seq_q, seq_k]
        matmul_result = self.batch_matmul_q_k(query_layer, key_layer)

        # record original score dtype
        attention_scores_dtype = matmul_result.dtype
        # [b, heads, seq, seq]
        attention_scores = matmul_result

        if self.coeff is not None:
            attention_scores = attention_scores * self.coeff
        if attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
            attention_mask = ops.ones((attention_scores.shape[0],
                                       1,
                                       attention_scores.shape[2],
                                       attention_scores.shape[3]), dtype=mstype.bool_)
            attention_mask.tril()
            attention_mask = ~attention_mask
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask, -10000)

        if self.attention_softmax_in_fp32:
            attention_scores = F.cast(attention_scores, mstype.float32)

        attention_probs = self.softmax(attention_scores)
        attention_probs = F.cast(attention_probs, attention_scores_dtype)

        attention_probs = self.attention_dropout(attention_probs)

        # [bs, heads, seq_q, seq_k] x [bs, heads, seq_v, hidden_size_per_head] -> [b, heads, seq_q, hidden_size_per_head]
        context_layer = self.batch_matmul(attention_probs, value_layer)
        context_layer = F.cast(context_layer, self.compute_dtype)

        context_layer = self._merge_heads(context_layer)

        return context_layer

    def _merge_heads(self, x):
        """
        convert a 4d input to a 2d output

        Inputs:
            x: input tensor

        Output:
            x_merge: the 2d output
        """
        x = self.merger_head_transpose(x, (0, 2, 1, 3))  # bs, seq_length, head, size_per_head
        x_shape = x.shape
        new_shape = (x_shape[0], x_shape[1], -1)
        x_merge = self.reshape(x, new_shape)
        return x_merge


class ChatGLM2SelfAttention(nn.Cell):
    """ChatGLM2 self-attention."""
    def __init__(self, config: ChatGLM2Config, layer_number):
        super(ChatGLM2SelfAttention, self).__init__()
        self.layer_number = max(1, layer_number)

        self.projection_size = config.kv_channels * config.num_attention_heads
        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads
        self.params_dtype = config.param_init_type
        self.compute_dtype = config.compute_dtype
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length

        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size

        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num)

        parallel_config = config.parallel_config
        self.query_key_value = Linear(config.hidden_size,
                                      self.qkv_hidden_size,
                                      has_bias=config.add_bias_linear or config.add_qkv_bias,
                                      param_init_type=self.params_dtype,
                                      compute_dtype=self.compute_dtype)
        self.query_key_value.shard(strategy_matmul=((parallel_config.data_parallel, 1),
                                                    (parallel_config.model_parallel, 1)),
                                   strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                                                  (parallel_config.model_parallel,))
                                   )

        self.core_attention = CoreAttention(config, self.layer_number)

        self.dense = Linear(self.projection_size,
                            config.hidden_size,
                            has_bias=config.add_bias_linear,
                            param_init_type=self.params_dtype,
                            compute_dtype=self.compute_dtype)
        self.dense.shard(strategy_matmul=((parallel_config.data_parallel, 1),
                                          (parallel_config.model_parallel, 1)),
                         strategy_bias=((parallel_config.data_parallel, 1), (1,)))

        self.reshape = P.Reshape()
        self.stack = P.Stack(axis=-1)
        self.gather = P.Gather()
        self.index_0 = Tensor(0)
        self.index_1 = Tensor(1)
        self.mul = P.Mul()
        self.sub = P.Sub()
        self.add = P.Add()
        self.concat = P.Concat(axis=-1)
        self.split_3 = P.Split(axis=-1, output_num=3)
        self.transpose = P.Transpose()

        self.use_past = config.use_past
        if self.use_past:
            self.is_first_iteration = True
            seq_range = np.arange(self.seq_length).reshape(1, 1, -1)
            self.range = Tensor(
                np.tile(seq_range, (self.batch_size, 1, 1)), mstype.int32)
            self.less = P.Less()
            self.mul1 = P.Mul().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
            self.expand_dims = P.ExpandDims().shard(((1, 1, 1),))
            self.equal = P.Equal().shard(((1, 1, 1), (1, 1, 1)))
            self.tile = P.Tile().shard(((1, 1, 1, 1),))

    def apply_rotary_pos_emb(self, x: Tensor, rope_cache: Tensor) -> Tensor:
        """apply rotary position embedding to q,k."""
        # x: [b, heads, seq, hidden_size_per_head]
        bs, num_heads, seq_len, _ = x.shape  # 1, 32，4, 128
        # rope_cache: first (seq_len, kv_channels//4, 2), other (1, kv_channels//4, 2)
        rot_dim = rope_cache.shape[-2] * 2  # kv_channels // 2
        x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
        # ms not support variable sizes
        # truncate to support variable sizes
        # rope_cache = rope_cache[:sq]
        # [bs, nh, sq, kv_channels//4, 2]
        xshaped = self.reshape(x, (bs, num_heads, seq_len, rot_dim // 2, 2))
        # [bs, 1, sq, kv_channels//4, 2]
        rope_cache = self.reshape(rope_cache, (-1, 1, seq_len, xshaped.shape[3], 2))

        xshaped_0, xshaped_1 = ops.split(xshaped, 1, -1)
        rope_cache_0, rope_cache_1 = ops.split(rope_cache, 1, -1)
        x_out1 = self.sub(self.mul(xshaped_0, rope_cache_0), self.mul(xshaped_1, rope_cache_1))
        x_out2 = self.add(self.mul(xshaped_1, rope_cache_0), self.mul(xshaped_0, rope_cache_1))
        x_out = self.stack((x_out1, x_out2))
        x_out = self.reshape(x_out, (x_out.shape[0], x_out.shape[1], x_out.shape[2], -1))
        # [bs, sq, nh, hidden_size_per_head]
        return self.concat((x_out, x_pass))

    def construct(self, hidden_states, attention_mask, rotary_pos_emb, key_past=None, value_past=None,
                  batch_valid_length=None):
        """Forward process of self-attention."""
        # hidden_states: [bs, seq_len, hidden_size]
        # attention_mask (bs, 1, seq_len, seq_len)
        # rotary_pos_emb: first: (sen length, kv_channels//4, 2)， after:(1, kv_channels//4, 2]

        # [bs, seq_len, qkv_hidden_size]
        mixed_raw_layer = self.query_key_value(hidden_states)

        # not compatible with ms below 2.0
        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_raw_layer.split(
                [self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                 self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                 self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                 ],
                axis=-1,
            )
            # [bs, seq_len, nh, hidden_size_per_attention_head] -> [bs, nh, seq_len, hidden_size_per_attention_head]
            query_layer = query_layer.view(
                query_layer.shape[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            query_layer = self.transpose(query_layer, (0, 2, 1, 3))
            # [bs, seq_len, multi_query_groups, hidden_size_per_attention_head]
            # -> [bs, multi_query_groups, seq_len, hidden_size_per_attention_head]
            key_layer = key_layer.view(
                key_layer.shape[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
            key_layer = self.transpose(key_layer, (0, 2, 1, 3))
            # [bs, seq_len, multi_query_groups, hidden_size_per_attention_head]
            # -> [bs, multi_query_groups, seq_len, hidden_size_per_attention_head]
            value_layer = value_layer.view(
                value_layer.shape[:-1]
                + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = self.transpose(value_layer, (0, 2, 1, 3))
        else:
            # [b, seq, (heads * 3 * hidden_size_per_head)] --> [b, seq, heads, 3 * hidden_size_per_head]
            new_tensor_shape = mixed_raw_layer.shape[:-1] + (
                self.num_attention_heads_per_partition, 3 * self.hidden_size_per_attention_head,
            )
            mixed_raw_layer = mixed_raw_layer.view(*new_tensor_shape)
            # [b, seq, heads, hidden_size_per_head]
            (query_layer, key_layer, value_layer) = self.split_3(mixed_raw_layer)
            # [b, seq, heads, hidden_size_per_head] -> [bs, num_heads, seq_len, hidden_size_per_head]
            query_layer = self.transpose(query_layer, (0, 2, 1, 3))
            key_layer = self.transpose(key_layer, (0, 2, 1, 3))
            value_layer = self.transpose(value_layer, (0, 2, 1, 3))

        # rotary_pos_emb: first: (seq_length, kv_channels//4, 2)， after:(1, kv_channels//4, 2)
        if rotary_pos_emb is not None:
            # [b, heads, seq, hidden_size_per_head]
            query_layer = self.apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            # [bs, multi_query_groups, seq_len, hidden_size_per_attention_head]
            key_layer = self.apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # key and value for current token(s)
        # [bs, heads, seq_len, hidden_size_per_head]
        key_present = key_layer
        value_present = value_layer
        if self.use_past:
            # The first graph with the input size of (bs, seq_length)
            if self.is_first_iteration:
                # Get the valid input length without padding
                valid_length_vector = F.cast(self.less(self.range, batch_valid_length.view(-1, 1, 1)),
                                             self.params_dtype)  # [bs, 1, seq_len]
                # Cover the key and value numbers corresponding to the padding position
                key_present = self.mul1(key_present, self.expand_dims(valid_length_vector, 3))
                value_present = self.mul1(value_present, self.expand_dims(valid_length_vector, 3))
            # The second graph with the inpus size of (bs, 1)
            # the shape of query is (bs, num_heads, 1, size_per_head)
            # the shape of key is   (bs, multi_query_groups, 1, size_per_head)
            # the shape of value is (bs, multi_query_groups, 1, size_per_head)
            else:
                # Get the current token position index
                # key_past: [batch_size, multi_query_groups, seq_length, size_per_head]
                valid_length = batch_valid_length - 1
                valid_length = self.reshape(valid_length, (-1, 1, 1))  # [bs, 1, 1]
                # self.range: [bs, 1, config.seq_len]
                valid_length_vector = F.cast(self.equal(valid_length, self.range), self.params_dtype)
                # Pad the key and value to seq_length with only the position index not zero
                current_key = self.mul1(self.tile(key_present, (1, 1, self.seq_length, 1)),
                                        self.expand_dims(valid_length_vector, 3))
                current_value = self.mul1(self.tile(value_present, (1, 1, self.seq_length, 1)),
                                          self.expand_dims(valid_length_vector, 3))
                # Concat the previous saved state and current state
                # [batch_size, multi_query_groups, seq_length, size_per_head]
                key_present = self.add(key_past, current_key)
                value_present = self.add(value_past, current_value)
            # update k v for attention
            # [batch_size, multi_query_groups, seq_length, size_per_head]
            key_layer = key_present
            # [batch_size, multi_query_groups, seq_length, size_per_head]
            value_layer = value_present

        layer_present = (key_present, value_present)

        # tile k,v to num_heads
        if self.multi_query_attention:
            bs, heads, _, hs_ph = key_layer.shape
            key_layer = key_layer.view((bs, heads, -1))

            key_layer = key_layer.unsqueeze(2)
            key_layer = key_layer.tile(
                (1, 1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, 1))
            # [b, heads, seq, hidden_size_per_head]
            key_layer = key_layer.view((bs, self.num_attention_heads_per_partition, -1, hs_ph))

            value_layer = value_layer.view((bs, heads, -1))
            value_layer = value_layer.unsqueeze(2)
            value_layer = value_layer.tile(
                (1, 1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, 1))
            # [b, heads, seq, hidden_size_per_head]
            value_layer = value_layer.view((bs, self.num_attention_heads_per_partition, -1, hs_ph))

        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
        # # =================
        # # Output. [bs, seq_len, hidden_size]
        # # =================

        output = self.dense(context_layer)

        return output, layer_present


class ChatGLM2Block(nn.Cell):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config: ChatGLM2Config, layer_number: int):
        super(ChatGLM2Block, self).__init__()
        self.layer_number = layer_number
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.fp32_residual_connection = config.fp32_residual_connection
        self.use_past = config.use_past
        self.params_dtype = config.param_init_type
        self.layernorm_dtype = config.layernorm_compute_type
        self.compute_dtype = config.compute_dtype
        self.seq_length = config.seq_length

        layer_norm_func = ChatGLM2RMSNorm if config.rmsnorm else LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = layer_norm_func(config.hidden_size, eps=config.layernorm_epsilon,
                                               param_init_type=self.layernorm_dtype)

        self.input_layernorm.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

        # Self attention.
        self.self_attention = ChatGLM2SelfAttention(config, layer_number)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = layer_norm_func(config.hidden_size, eps=config.layernorm_epsilon,
                                                        param_init_type=self.layernorm_dtype)
        # self.post_attention_layernorm.shard()

        # MLP
        self.mlp = ChatGLM2MLP(config)

        self.dropout = get_dropout(self.hidden_dropout)
        self.dropout.dropout.shard(((config.parallel_config.data_parallel, 1, 1),))

        self.cast = P.Cast()

        self.key_past = None
        self.value_past = None
        if self.use_past:
            size_per_head = config.hidden_size // config.num_attention_heads
            kv_num_partition = config.num_attention_heads
            if config.multi_query_attention:
                kv_num_partition = config.multi_query_group_num
            kv_shape = (config.batch_size, kv_num_partition, self.seq_length, size_per_head)
            # parameters saving key and value states
            self.key_past = Parameter(
                Tensor(np.zeros(shape=kv_shape), self.params_dtype), name="key_past")
            self.value_past = Parameter(
                Tensor(np.zeros(shape=kv_shape), self.params_dtype), name="value_past")
            self.mul = P.Mul().shard(((1, 1, 1, 1), (1,)))
            self.assign = P.Assign().shard(((1, 1, 1, 1), (1, 1, 1, 1)))

    def construct(self, hidden_states, attention_mask, rotary_pos_emb,
                  init_reset=True, batch_valid_length=None):
        """Forward process of the transformer layer."""
        # hidden_states: [bs, seq_len, hidden_size]
        # attention_mask first: (bs, 1, seq_len, seq_len), after: (bs, 1, 1, seq_len)
        # rotary_pos_emb: first: (seq_len, kv_channels//4, 2)， after: (1, kv_channels//4, 2)

        # Layer norm at the beginning of the transformer layer.
        hidden_states = self.cast(hidden_states, self.layernorm_dtype)
        layernorm_output = self.input_layernorm(hidden_states)
        # fp32 -> fp16
        layernorm_output = self.cast(layernorm_output, self.compute_dtype)

        key_reset = None
        value_reset = None
        if self.use_past:
            # reset states, init_reset True for reuse and False for reset
            key_reset = self.assign(self.key_past, self.mul(
                self.key_past, F.cast(init_reset, self.params_dtype)))
            value_reset = self.assign(self.value_past, self.mul(
                self.value_past, F.cast(init_reset, self.params_dtype)))
            # add dependency for desired execution order
            layernorm_output = F.depend(layernorm_output, key_reset)
            layernorm_output = F.depend(layernorm_output, value_reset)

        # Self attention.
        attention_output, layer_present = self.self_attention(
            layernorm_output,
            attention_mask,
            rotary_pos_emb,
            self.key_past,
            self.value_past,
            batch_valid_length
        )

        # Residual connection.
        # False on default.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = self.dropout(attention_output)
        layernorm_input = residual + layernorm_input

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        layernorm_output = self.cast(layernorm_output, self.compute_dtype)

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        value_update = None
        key_update = None
        if self.use_past:
            # current key and value
            key_present, value_present = layer_present
            # update key and value calculated this step
            self.assign(self.key_past, key_present)
            key_update = self.key_past
            self.assign(self.value_past, value_present)
            value_update = self.value_past
            # add dependency for desired execution order
            key_update = F.depend(key_update, key_reset)
            value_update = F.depend(value_update, value_reset)

        # add dependency for desired execution order
        mlp_output = F.depend(mlp_output, value_update)
        mlp_output = F.depend(mlp_output, key_update)

        # Second residual connection.
        # False on default.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = self.dropout(mlp_output)
        output = residual + output

        return output


def set_parallel_configure_for_layer(network, layer_id, offset, parallel_config, layers):
    r"""
        Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.


        Args:
            network(Cell) - Represents the transformer block
            layer_id(int) - Means the layer index for the current module, counts from zero.
            offset(int) - Means the layer_index needs a offset, if there are other modules in the net.
            layers(int) - The total layers used for the model.
    """
    # Used for optimizer's fusion tag
    dis = max(int((layers + 1) / parallel_config.gradient_aggregation_group), 1)
    if parallel_config.pipeline_stage > 1:
        # we give the fusion in pipeline mode a fixed value, otherwise the performance may become worse.
        network.set_comm_fusion(2)
    else:
        network.set_comm_fusion(int((layer_id + offset) / dis) + 1)
    # Used for enabling recomputation of the block
    if isinstance(parallel_config.recompute, bool):
        if parallel_config.recompute:
            network.recompute()
    else:
        if parallel_config.recompute.recompute:
            network.recompute(recompute_slice_activation=parallel_config.recompute.recompute_slice_activation)


class ChatGLM2Transformer(nn.Cell):
    """Transformer class."""

    def __init__(self, config: ChatGLM2Config):
        super(ChatGLM2Transformer, self).__init__()

        self.post_layer_norm = config.post_layer_norm
        self.compute_dtype = config.compute_dtype

        # Number of layers.
        self.num_layers = config.num_layers

        # Transformer layers.
        def build_layer(layer_number):
            return ChatGLM2Block(config, layer_number)

        self.layers = nn.CellList()
        for i in range(self.num_layers):
            layer = build_layer(i + 1)
            set_parallel_configure_for_layer(layer, layer_id=i, layers=self.num_layers,
                                             offset=0, parallel_config=config.parallel_config)
            self.layers.append(layer)

        if self.post_layer_norm:
            layer_norm_func = ChatGLM2RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = layer_norm_func(config.hidden_size, eps=config.layernorm_epsilon,
                                                   param_init_type=config.layernorm_compute_type)
            # self.final_layernorm.shard()
            self.final_layernorm.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

    def construct(self,
                  hidden_states,
                  attention_mask,
                  rotary_pos_emb,
                  init_reset=True,
                  batch_valid_length=None):
        """Forward process of the transformer."""
        # hidden_states (bs, seq_len, hs)
        # attention_mask (bs, 1, seq_len, seq_len)
        # rotary_pos_emb: first: (sen length, kv_channels//2, 2)， after:[1, kv_channels // 2, 2]
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                init_reset=init_reset,
                batch_valid_length=batch_valid_length,
            )

        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)
            hidden_states = self.cast(hidden_states, self.compute_dtype)

        return hidden_states
