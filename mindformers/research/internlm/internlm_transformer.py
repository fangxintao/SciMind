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
"""InternLM transformer Layer's APIs."""
from typing import Tuple, Optional
import math
import numpy as np

# pylint: disable=W0611
try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator

from mindspore import nn, ops
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.context import ParallelMode
from mindspore.ops import operations as P

from mindformers.models.llama.llama_layer import LlamaFeedForward, LlamaRMSNorm, LlamaRotaryEmbedding
from mindformers.modules.layers import _check_past_none_input_none, _check_input_dtype, Linear
from mindformers.modules.transformer import TransformerOpParallelConfig
from mindformers.modules.transformer.op_parallel_config import _check_config


class InternLMAttention(nn.Cell):
    r"""
    This is an implementation of multihead attention in InternLM.

    Args:
            - **batch_size** (int): The batch size of the input tensor when do increnmental prediction. Should be a
                positive value.
                When do training or prediction, the argument will not work and the user can just pass None to the
                argument.
            - **src_seq_length** (int): The sequence length of the query vector.
            - **tgt_seq_length** (int): The sequence length of the key and value vector.
            - **dim** (int): The hidden size of the input.
            - **head_dim** (int): The dim of head.
            - **n_heads** (int): The number of the heads.
            - **compute_dtype** (dtype.Number): The computation type of dense. Default mstype.float16.
                Should be mstype.float32 or mstype.float16.
            - **softmax_compute_type** (dtype.Number): The type of softmax computation module. Default mstype.float32.
                Should be mstype.float32 or mstype.float16.
            - **param_init_type** (dtype.Number): The parameter initialization type of the module. Default mstype.
                float32. Should be mstype.float32 or mstype.float16.
            - **use_past** (bool): Use the past state to compute, used for incremental prediction.
                For example, if we have two words and want to generate the ten more words.
                We just need to compute the two words' state only once, and generate the next word one by one.
                When use_past is True, there are two steps to run the prediction.
                In the first step, set the is_first_iteration to be True by
                `model.add_flags_recursive(is_first_iteration=True)`, and pass the full inputs. Then, set the
                is_first_iteration to be False by `model.add_flags_recursive(is_first_iteration=False)`. At this moment,
                pass the single step's input tensor, and loop it. Default False.
            - **parallel_config** (OpParallelConfig): The parallel configure. Default `default_dpmp_config`,
                an instance of `OpParallelConfig` with default args.

    Inputs:
            - **x** (Tensor) - The input tokens with shape (batch_size, src_seq_length, hidden_size) or
                (batch_size * src_seq_length, hidden_size), if the use_past is False or is_first_iteration=True.
                Otherwise, must be (batch_size, 1, hidden_size)
            - **freqs_cis** (Tuple) - The precompute freqs and mask for rotary position embedding used in attention.
            - **attention_mask** (Tensor) - If the use_past is False or is_first_iteration=True, the attention mask
                matrix should ba (batch_size, src_seq_length, tgt_seq_length), or None. None means there will be no mask
                in softmax computation. Otherwise, the mask must be (batch_size, 1, tgt_seq_length)
            - **key_past** (Tensor) - Float16 tensor with shape (batch_size, num_heads, size_per_head, tgt_seq_length).
                The past calculated key vector. Used for incremental prediction when the use_past is True.
                Default None.
            - **value_past** (Tensor) - Float16 tensor with shape (batch_size, num_heads, tgt_seq_length,
                size_per_head).
                The past calculated value vector. Used for incremental prediction when the use_past is True.
                Default None.
            - **batch_valid_length** (Tensor) - Int32 tensor with shape (batch_size,) the past calculated the index.
                Used for incremental prediction when the use_past is True. Default None.

    Outputs:
            Tuple, a tuple contains(`output`, `layer_present`)

            - **output** (Tensor) - Tensor, the float tensor of the output of the layer with
                shape (batch_size, src_seq_length, hidden_size) or (batch_size * src_seq_length, hidden_size),
                if the use_past is False or is_first_iteration=True. Otherwise, it will be (batch_size, 1, hidden_size).

            - **layer_present** (Tuple) - A tuple of the Tensor of the projected key and value vector with
                ((batch_size, num_heads, size_per_head, tgt_seq_length),
                (batch_size, num_heads, tgt_seq_length, size_per_head)).
    """

    def __init__(self,
                 batch_size,
                 seq_length,
                 dim: int = 512,
                 n_heads: int = 8,
                 n_kv_heads: Optional[int] = None,
                 compute_dtype=mstype.float16,
                 softmax_compute_dtype=mstype.float32,
                 rotary_dtype=mstype.float32,
                 param_init_type=mstype.float32,
                 use_past=False,
                 use_flash_attention=False,
                 compute_in_2d=False,
                 use_past_shard=False,
                 parallel_config=TransformerOpParallelConfig()):
        super().__init__()
        self.seq_length = seq_length
        self.hidden_size = dim
        self.n_head = n_heads
        self.head_dim = dim // n_heads
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads
        self.n_rep = self.n_head // self.n_kv_head

        self.dtype = compute_dtype
        self.softmax_dtype = softmax_compute_dtype
        self.is_first_iteration = True
        self.use_past = use_past
        self.compute_in_2d = compute_in_2d
        self.use_flash_attention = use_flash_attention and FLASHATTENTION_VALID

        if self.hidden_size % self.n_head != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                             "of 'n_head', but got the hidden_size is {} and the n_head is {}."
                             .format(self.hidden_size, self.n_head))
        if self.n_kv_head % parallel_config.model_parallel != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'n_kv_head' must be a multiple of "
                             "'parallel_config.model_parallel', but got the n_kv_head is {} "
                             "and the parallel_config.model_parallel  is {}."
                             .format(self.n_kv_head, parallel_config.model_parallel))

        self.inv_norm_factor = Tensor(1.0 / math.sqrt(self.head_dim), dtype=compute_dtype)

        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.merger_head_transpose = P.Transpose()
        self.batch_matmul = P.BatchMatMul()
        self.batch_matmul_q_k = P.BatchMatMul(transpose_b=True)
        self.mul = P.Mul()
        self.add = P.Add()
        self.softmax = nn.Softmax().to_float(softmax_compute_dtype)
        self.cast = P.Cast()
        self.cast_attn = P.Cast()
        self.tile_kv = P.Tile()

        self.apply_rotary_emb = LlamaRotaryEmbedding(self.head_dim, rotary_dtype)
        self.wo = Linear(in_channels=self.hidden_size,
                         out_channels=self.hidden_size,
                         has_bias=True,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)
        self.wq = Linear(self.hidden_size,
                         self.hidden_size,
                         has_bias=True,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)
        self.wk = Linear(self.hidden_size,
                         self.n_kv_head * self.head_dim,
                         has_bias=True,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)
        self.wv = Linear(self.hidden_size,
                         self.n_kv_head * self.head_dim,
                         has_bias=True,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)

        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel

        self.transpose.shard(((dp, 1, mp, 1),))
        self.merger_head_transpose.shard(((dp, mp, 1, 1),))
        self.batch_matmul_q_k.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.batch_matmul.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.mul.shard(((dp, mp, 1, 1), ()))
        self.add.shard(((dp, 1, 1, 1), (dp, mp, 1, 1)))
        self.softmax.softmax.shard(((dp, mp, 1, 1),))
        self.tile_kv.shard(((dp * mp, 1, 1, 1),))

        self.apply_rotary_emb.shard((dp, mp, 1, 1))
        self.wq.shard(((dp, 1), (mp, 1)))
        self.wk.shard(((dp, 1), (mp, 1)))
        self.wv.shard(((dp, 1), (mp, 1)))
        self.wo.shard(((dp, mp), (1, mp)))
        if parallel_config.use_seq_parallel and self.is_first_iteration:
            self.wo.shard(((dp, mp), (1, mp)), out_strategy_matmul=((dp * mp, 1),))
        if parallel_config.recompute.select_recompute:
            self.apply_rotary_emb.recompute()
            self.tile_kv.recompute()
            self.batch_matmul_q_k.recompute()
            self.mul.recompute()
            self.add.recompute()
            self.cast_attn.recompute()
            self.softmax.softmax.recompute()
            self.batch_matmul.recompute()

        if self.use_flash_attention:
            self.flash_attention = FlashAttention(self.head_dim, dp=dp, mp=mp, next_block_num=0)
            self.flash_attention.shard(((dp, mp, 1, 1), (dp, mp, 1, 1), (dp, mp, 1, 1), (dp, 1, 1), ()))
            if parallel_config.recompute.select_recompute:
                self.flash_attention.recompute()

        if self.use_past:
            # operators used for state reuse
            seq_range = np.arange(seq_length).reshape(1, 1, -1)
            self.range = Tensor(np.tile(seq_range, (batch_size, 1, 1)), mstype.int32)
            self.expand_dims = P.ExpandDims().shard(((dp, 1, 1),))
            self.add_past = P.Add().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
            self.equal = P.Equal().shard(((dp, 1, 1), (dp, 1, 1)))
            self.less = P.Less().shard(((dp, 1, 1), (dp, 1, 1)))
            self.mul_past = P.Mul().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
            if use_past_shard:
                self.add_past.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
                self.mul_past.shard(((dp, mp, 1, 1), (dp, 1, 1, 1)))

    def construct(self, x: Tensor, freqs_cis: Tuple[Tensor, Tensor], mask=None,
                  key_past=None, value_past=None, batch_valid_length=None):
        """Forward process of the MultiHeadAttention"""
        ori_dtype = x.dtype
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        x = self.reshape(x, (-1, x.shape[-1]))
        # [bs * seq/1, hidden_dim]
        query = self.cast(self.wq(x), self.dtype)  # dp, 1 -> dp, mp
        key = self.cast(self.wk(x), self.dtype)  # dp, 1 -> dp, mp
        value = self.cast(self.wv(x), self.dtype)  # dp, 1 -> dp, mp
        query = self.reshape(query, (-1, self._get_seq_length_under_incremental(self.seq_length),
                                     self.n_head, self.head_dim))
        key = self.reshape(key, (-1, self._get_seq_length_under_incremental(self.seq_length),
                                 self.n_kv_head, self.head_dim))
        value = self.reshape(value, (-1, self._get_seq_length_under_incremental(self.seq_length),
                                     self.n_kv_head, self.head_dim))
        # [bs, seq/1, n_head/n_kv_head, head_dim]
        query = self.transpose(query, (0, 2, 1, 3))
        key = self.transpose(key, (0, 2, 1, 3))
        value = self.transpose(value, (0, 2, 1, 3))
        # [bs, n_head/n_kv_head, seq/1, head_dim]
        query, key = self.apply_rotary_emb(query, key, freqs_cis)  # dp, mp, 1, 1
        # kv cache: [bs, n_kv_head, 1, head_dim] -> [bs, n_kv_head, seq, head_dim]
        key_present = key
        value_present = value
        if self.use_past:
            # The first graph with the input size of (bs, seq_length)
            if self.is_first_iteration:
                # Get the valid input length without padding
                valid_length_vector = (
                    self.less(self.range, batch_valid_length.view(-1, 1, 1))).astype(self.dtype)
                # Cover the key and value numbers corresponding to the padding position
                key_present = self.mul_past(key, self.expand_dims(valid_length_vector, 3))
                value_present = self.mul_past(value, self.expand_dims(valid_length_vector, 3))
            # The second graph with the inpus size of (bs, 1)
            else:
                # Get the current token position index
                valid_length = batch_valid_length - 1
                valid_length = self.reshape(valid_length, (-1, 1, 1))
                valid_length_vector = (self.equal(self.range, valid_length)).astype(self.dtype)
                # Pad the key and value to seq_length with only the position index not zero
                current_key = self.mul_past(key, self.expand_dims(valid_length_vector, 3))
                current_value = self.mul_past(value, self.expand_dims(valid_length_vector, 3))
                # Concat the previous saved state and current state
                key = self.add_past(key_past, current_key)
                value = self.add_past(value_past, current_value)
                # Update key_present and value_present for state update
                key_present = key
                value_present = value

        layer_present = (key_present, value_present)
        # kv share: [bs, n_kv_head, seq, head_dim] -> [bs, n_head, seq, head_dim]
        key = self._repeat_kv(key, self.n_rep)
        value = self._repeat_kv(value, self.n_rep)
        # q, k, v: [bs, n_head, seq/1, head_dim], [bs, n_head, seq, head_dim], [bs, n_head, seq, head_dim]
        if self.use_flash_attention:
            attention = self.flash_attention(query, key, value, mask)
            attention = self._merge_heads(attention)
        else:
            attention = self._attn(query, key, value, mask)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        output = self.wo(attention)  # dp, mp -> dp, 1 / dp * mp, 1
        output = self.cast(output, ori_dtype)

        return output, layer_present

    def _repeat_kv(self, x, rep):
        if rep == 1:
            return x
        bs, n_kv_head, seqlen, head_dim = x.shape
        x = self.reshape(x, (bs * n_kv_head, 1, seqlen, head_dim))
        x = self.tile_kv(x, (1, rep, 1, 1))
        x = self.reshape(x, (bs, n_kv_head * rep, seqlen, head_dim))
        return x

    def _get_seq_length_under_incremental(self, length):
        r"""Return the length of the tensor.
            For the incremental prediction, the seq length for the input is 1.
        """
        if self.use_past and not self.is_first_iteration:
            return 1
        return length

    def _merge_heads(self, x):
        """
        convert a 4d input to a 2d or 3d output

        Inputs:
            x: input tensor

        Output:
            x_merge: the 2d output
        """
        # [bs, n_head, seq/1, head_dim]
        x = self.merger_head_transpose(x, (0, 2, 1, 3))  # dp,mp,1,1 -> dp,1,mp,1
        # [bs, seq/1, n_head, head_dim]
        x_shape = x.shape
        if self.compute_in_2d:
            # [bs * seq/1, hidden_dim]
            new_shape = (-1, x_shape[-2] * x_shape[-1])
        else:
            # [bs, seq/1, hidden_dim]
            new_shape = (x_shape[0], x_shape[1], -1)
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _attn(self, query, key, value, mask):
        """
        Get the weighted score along the seq_length

        Inputs:
            query: the query matrix
            key: the key matrix
            value: the value matrix
            mask: the attention mask adder matrix with shape (batch_size,
            1, seq_length, seq_length)
        Outputs:
            weighted_values: Tensor, the weighted sum scores
        """
        # q, k: [bs, n_head, seq/1, head_dim], [bs, n_head, seq, head_dim]
        score = self.batch_matmul_q_k(query, key)
        # score: [bs, n_head, seq/1, seq]
        score = self.mul(score, self.inv_norm_factor)
        score = self.add(mask, score)

        attention_probs = self.softmax(self.cast_attn(score, self.softmax_dtype))
        # score, v: [bs, n_head, seq/1, seq], [bs, n_head, seq, head_dim]
        weighted_values = self.batch_matmul(self.cast(attention_probs, self.dtype), value)
        # [bs, n_head, seq/1, head_dim]
        attention_merge = self._merge_heads(weighted_values)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        return attention_merge


class InternLMDecodeLayer(nn.Cell):
    r"""
        Transformer Layer. This is an implementation of the single layer of the transformer
        encoder layer, including multihead attention and feedward layer.

        Args:
            batch_size(int): The batch size of the input tensor when do increnmental prediction. Should be a positive
                value. When do training or prediction, the argument will not work and the user can just pass None to
                the argument.
            seq_length(int): The input sequence length.
            layer_id(int): The layer id of current transformer block layer.
            dim(int): The hidden size of the input.
            num_heads(int): The number of the heads.
            multiple_of(int): The SwiGLU hidden layer size multiple of large power of 2.
            norm_eps (float): The epsilon value of the denominator. Default 1e-5.
            compute_dtype(dtype.Number): The computation type of the layer.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            layernorm_compute_type(dtype.Number): The computation type of the norm.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            softmax_compute_type(dtype.Number): The computation type of the softmax in the attention.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            param_init_type(dtype.Number): The parameter initialization type of the module.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            use_past(bool): Use the past state to compute, used for incremental prediction. For example, if we have two
                words and want to generate the ten more words. We just need to compute the two words' state only once,
                and generate the next word one by one. When use_past is True, there are two steps to run the prediction.
                In the first step, set the is_first_iteration to be True by
                `model.add_flags_recursive(is_first_iteration=True)`, and pass the full inputs. Then, set the
                is_first_iteration to be False by `model.add_flags_recursive(is_first_iteration=False)`.
                At this moment, pass the single step's input tensor, and loop it. Default False.
            parallel_config(OpParallelConfig, MoEParallelConfig): The parallel configure. When MoE is applied,
                MoEParallelConfig is effective, otherwise OpParallelConfig is effective. Default `default_dpmp_config`,
                an instance of `OpParallelConfig` with default args.

        Inputs:
            - **x** (Tensor) - Float Tensor, shape should be [batch_size, seq_length, hidden_size] or
              [batch_size * seq_length, hidden_size], if the use_past is False or is_first_iteration=True. Otherwise,
              should be [batch_size, 1, hidden_size]
            - **freqs_cis** (Tuple) - The precompute freqs and mask for rotary position embedding used in attention.
            - **input_mask** (Tensor) - Float Tensor, If the use_past is False or is_first_iteration=True,
              the attention mask matrix should ba [batch_size, seq_length, seq_length], or None. None means there will
              be no mask in softmax computation. Otherwise, should be [batch_size, 1, hidden_size]
            - **init_reset** (Tensor) - A bool tensor with shape [1], used to clear the past key parameter and
              past value parameter used in the incremental prediction. Only valid when use_past is True. Default True.
            - **batch_valid_length** (Tensor) - Int32 tensor with shape [batch_size] the past calculated the index.
              Used for incremental prediction when the use_past is True. Default None.

        Outputs:
            Tuple, a tuple contains(`output`, `layer_present`).

            - **output** (Tensor) - The float tensor of the output of the layer with
              shape (batch_size, seq_length, hidden_size) or (batch_size * seq_length, hidden_size), if the use_past is
              False or is_first_iteration=True. Otherwise, it will be (batch_size, 1, hidden_size)

            - **layer_present** (Tuple) - A tuple of the Tensor of the projected key and value vector with
              ((batch_size, num_heads, size_per_head, seq_length),
              (batch_size, num_heads, seq_length, size_per_head)).

    """

    def __init__(self,
                 batch_size,
                 seq_length,
                 layer_id,
                 dim: int = 512,
                 n_heads: int = 8,
                 multiple_of: int = 256,
                 n_kv_heads: Optional[int] = None,
                 ffn_dim_multiplier: Optional[int] = None,
                 norm_eps: float = 1e-5,
                 compute_dtype=mstype.float16,
                 layernorm_compute_dtype=mstype.float32,
                 softmax_compute_dtype=mstype.float32,
                 rotary_dtype=mstype.float32,
                 param_init_type=mstype.float32,
                 use_past=False,
                 use_flash_attention=False,
                 compute_in_2d=False,
                 use_past_shard=False,
                 parallel_config=TransformerOpParallelConfig()):
        super().__init__()
        if batch_size or use_past:
            Validator.check_positive_int(batch_size)
        self.batch_size = batch_size

        self.compute_in_2d = compute_in_2d
        self.seq_length = seq_length
        self.layer_id = layer_id
        self.hidden_size = dim
        self.n_head = n_heads
        self.head_dim = self.hidden_size // self.n_head
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads

        self.dtype = compute_dtype
        self.is_first_iteration = True
        self.use_past = use_past
        self.compute_in_2d = compute_in_2d
        self.key_past = None
        self.value_past = None

        self.reshape = P.Reshape()
        self.add = P.Add()
        self.attention_norm = LlamaRMSNorm(self.hidden_size, norm_eps, compute_type=layernorm_compute_dtype)
        self.ffn_norm = LlamaRMSNorm(self.hidden_size, norm_eps, compute_type=layernorm_compute_dtype)
        self.attention = InternLMAttention(batch_size=batch_size,
                                           seq_length=seq_length,
                                           dim=dim,
                                           n_heads=n_heads,
                                           n_kv_heads=n_kv_heads,
                                           compute_dtype=compute_dtype,
                                           softmax_compute_dtype=softmax_compute_dtype,
                                           rotary_dtype=rotary_dtype,
                                           param_init_type=param_init_type,
                                           use_past=use_past,
                                           use_flash_attention=use_flash_attention,
                                           compute_in_2d=compute_in_2d,
                                           use_past_shard=use_past_shard,
                                           parallel_config=parallel_config)
        self.feed_forward = LlamaFeedForward(dim=self.hidden_size,
                                             hidden_dim=4 * self.hidden_size,
                                             multiple_of=multiple_of,
                                             ffn_dim_multiplier=ffn_dim_multiplier,
                                             compute_dtype=compute_dtype,
                                             param_init_type=param_init_type)

        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel

        self.feed_forward.shard(parallel_config)
        if self.compute_in_2d:
            self.add.shard(((dp, 1), (dp, 1)))
            self.attention_norm.shard((dp, 1))
            self.ffn_norm.shard((dp, 1))
        else:
            self.add.shard(((dp, 1, 1), (dp, 1, 1)))
            self.attention_norm.shard((dp, 1, 1))
            self.ffn_norm.shard((dp, 1, 1))
            self.feed_forward.mul.shard(((dp, 1, mp), (dp, 1, mp)))

        if parallel_config.use_seq_parallel and self.is_first_iteration:
            if self.compute_in_2d:
                self.add.shard(((dp * mp, 1), (dp * mp, 1)))
                self.attention_norm.shard((dp * mp, 1))
                self.ffn_norm.shard((dp * mp, 1))
            else:
                self.add.shard(((dp, mp, 1), (dp, mp, 1)))
                self.attention_norm.shard((dp, mp, 1))
                self.ffn_norm.shard((dp, mp, 1))
            self.feed_forward.w2.shard(((dp, mp), (1, mp)), out_strategy_matmul=((dp * mp, 1),))

        if self.use_past:
            kv_shape = (batch_size, self.n_kv_head, seq_length, self.head_dim)
            self.key_past = Parameter(Tensor(np.zeros(kv_shape), self.dtype), name="key_past")
            self.value_past = Parameter(Tensor(np.zeros(kv_shape), self.dtype), name="value_past")
            self.mul_past = P.Mul().shard(((dp, 1, 1, 1), (1,)))
            self.assign_past = P.Assign().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
            if use_past_shard:
                self.mul_past.shard(((dp, mp, 1, 1), (1,)))
                self.assign_past.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))

    def construct(self, x, freqs_cis, mask=None, init_reset=True, batch_valid_length=None):
        """ Forward of transformer block. """
        if self.use_past:
            if not isinstance(init_reset, Tensor):
                init_reset = Tensor([init_reset], mstype.bool_)
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((self.batch_size, 1), mstype.int32)
        self._check_input(x, freqs_cis, mask, init_reset, batch_valid_length)
        # [bs, seq/1, hidden_dim] (first) [bs * seq/1, hidden_dim] (others)
        if self.compute_in_2d and x.ndim != 2:
            x = self.reshape(x, (-1, x.shape[-1]))
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        input_x = self.attention_norm(x)

        key_reset = None
        value_reset = None
        if self.use_past and self.is_first_iteration:
            # reset states, init_reset True for reuse and False for reset
            self.assign_past(self.key_past, self.mul_past(self.key_past, self.cast(init_reset, self.dtype)))
            self.assign_past(self.value_past, self.mul_past(self.value_past, self.cast(init_reset, self.dtype)))
            key_reset = self.key_past
            value_reset = self.value_past
            # add dependency for desired execution order
            input_x = ops.depend(input_x, key_reset)
            input_x = ops.depend(input_x, value_reset)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        h, layer_present = self.attention(input_x, freqs_cis, mask,
                                          self.key_past, self.value_past, batch_valid_length)
        h = self.add(x, h)
        ffn_norm = self.ffn_norm(h)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        ffn_out = self.feed_forward(ffn_norm)

        value_update = None
        key_update = None
        if self.use_past:
            # current key and value
            key_present, value_present = layer_present
            # update key and value calculated this step
            self.assign_past(self.key_past, key_present)
            self.assign_past(self.value_past, value_present)
            key_update = self.key_past
            value_update = self.value_past
            # add dependency for desired execution order
            key_update = ops.depend(key_update, key_reset)
            value_update = ops.depend(value_update, value_reset)

        # add dependency for desired execution order
        ffn_out = ops.depend(ffn_out, value_update)
        ffn_out = ops.depend(ffn_out, key_update)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        out = self.add(h, ffn_out)
        return out, layer_present

    def _check_input(self, x, freqs_cis, mask, init_reset, batch_valid_length):
        r"""Check inputs"""
        _check_input_dtype(
            x.dtype, "x", [mstype.float32, mstype.float16], self.cls_name)
        freqs_cos, freqs_sin, swap_mask = freqs_cis
        _check_input_dtype(freqs_cos.dtype, "freqs_cos", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_dtype(freqs_sin.dtype, "freqs_sin", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_dtype(swap_mask.dtype, "swap_mask", [mstype.float32, mstype.float16], self.cls_name)
        if mask is not None:
            _check_input_dtype(mask.dtype, "input_mask", [mstype.float32, mstype.float16], self.cls_name)

        init_reset_is_tensor = isinstance(init_reset, Tensor)
        init_reset_is_default = init_reset is True
        batch_valid_length_is_tensor = isinstance(batch_valid_length, Tensor)
        batch_is_default = batch_valid_length is None
        _check_past_none_input_none(self.use_past, "init_reset", self.cls_name, True, init_reset_is_tensor,
                                    init_reset_is_default)
        _check_past_none_input_none(self.use_past, "batch_valid_length", self.cls_name, None,
                                    batch_valid_length_is_tensor, batch_is_default)

        if self.use_past:
            _check_input_dtype(init_reset.dtype, "init_reset", [mstype.bool_], self.cls_name)
            _check_input_dtype(batch_valid_length.dtype, "batch_valid_length", [mstype.int32], self.cls_name)
        return True
