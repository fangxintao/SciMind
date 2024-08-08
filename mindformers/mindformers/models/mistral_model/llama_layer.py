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
"""LLaMA Model Layers' APIs."""

from enum import Enum
import numpy as np
from mindspore import ops

from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore import nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.cell import Cell

try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore import log as logger
from mindspore.common.initializer import initializer
from mindspore.parallel._utils import _get_parallel_mode
from mindspore.context import ParallelMode
from mindformers.modules.layers import Linear, _check_input_dtype, _args_type_validator_check, _valid_value_checks

from mindformers.tools.logger import _LogActionOnce


class SeqExtendMethod(Enum):
    """Stores the acceptable string identifiers for seq length extend method"""
    PI = "PI"
    NTK = "NTK"
    NONE = "None"


class LlamaSiLU(Cell):
    r"""
    A self-defined SwiGlu.

        Inputs:
            - **x** (Tensor) - Tensor.

        Outputs:
            Tensor. x = x * sigmod(x).
    """
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.mul = P.Mul()

    def shard(self, strategy):
        self.sigmoid.sigmoid.shard(strategy)
        self.mul.shard((strategy[0], strategy[0]))

    def construct(self, x):
        return self.mul(x, self.sigmoid(x))


def get_swap_mask(head_dim):
    """Swap matrix"""
    zero_block = np.zeros((head_dim // 2, head_dim // 2), dtype=np.float32)
    id_block = np.identity(head_dim // 2, dtype=np.float32)
    return np.block([[zero_block, id_block], [-id_block, zero_block]])


def precompute_freqs_cis(
        dim: int,
        end: int,
        theta: float = 1000000.0,
        dtype=mstype.float32,
        # pretrain_seqlen=2048,
        pretrain_seqlen=32768,
        extend_method=SeqExtendMethod.NONE.value):
    """
    Precompute of freqs and mask for rotary embedding.
    """
    ratio = 1.
    if extend_method != SeqExtendMethod.NONE.value and end > pretrain_seqlen:
        ratio = end / pretrain_seqlen
    if extend_method == SeqExtendMethod.NTK.value:
        theta *= ratio
    freqs_base = np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) # (head_dim // 2, )
    freqs = 1.0 / (theta ** (freqs_base / dim)) # (head_dim // 2, )
    if extend_method == SeqExtendMethod.PI.value:
        t = np.arange(0, end / ratio, 1 / ratio).astype(np.float32)
    else:
        t = np.arange(0, end, 1).astype(np.float32)  # type: ignore # (seq_len,)
    freqs = np.outer(t, freqs)  # type: ignore (seq_len, head_dim // 2)
    emb = np.concatenate((freqs, freqs), axis=-1)
    freqs_cos = np.cos(emb) # (seq_len, head_dim)
    freqs_sin = np.sin(emb) # (seq_len, head_dim)
    freqs_cos = Tensor(freqs_cos, dtype=dtype)
    freqs_sin = Tensor(freqs_sin, dtype=dtype)

    swap_mask = get_swap_mask(dim)
    swap_mask = Tensor(swap_mask, dtype=dtype)

    return freqs_cos, freqs_sin, swap_mask


class RotaryEmbedding(Cell): # copy from moudle.layers.py
    r"""
    Rotary Position Embedding.

    Args:
            - **head_dim** (int): The dim of multi head attention.
            - **compute_dtype** (mstype): The compute type, default mstype.float16.
            - **use_rope_slice** (dict): - Choose using rope slice. Default False.

    Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

    Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(self, head_dim=128, compute_dtype=mstype.float32, use_rope_slice=False):
        super().__init__(auto_prefix=False)
        self.half_head_dim = head_dim // 2
        self.head_dim = head_dim
        self.dtype = compute_dtype
        self.use_rope_slice = use_rope_slice
        self.is_first_iteration = True
        self.add = P.Add()
        self.bmm_swap = P.BatchMatMul()
        self.mul = P.Mul()
        self.mul_inc = P.Mul()
        self.neg = P.Neg()
        self.slice = P.StridedSlice()
        self.concat = P.Concat(axis=-1)
        self.shape = P.Shape()

    def rotate_half(self, x, swap_mask):
        # [bs, n_head/n_kv_head, seq/1, head_dim], [head_dim, head_dim]
        x = self.bmm_swap(x, swap_mask)
        return x

    def slice_half(self, x):
        bs, n_head, seq, _ = self.shape(x)
        x1 = self.slice(x, (0, 0, 0, 0), (bs, n_head, seq, self.half_head_dim), (1, 1, 1, 1))
        x2 = self.slice(x, (0, 0, 0, self.half_head_dim), (bs, n_head, seq, self.head_dim), (1, 1, 1, 1))
        x = self.concat((self.neg(x2), x1))
        return x

    def construct(self, xq: Tensor, xk: Tensor, freqs_cis):
        """Forward of rotary position embedding."""
        original_type = xq.dtype
        xq = self.cast(xq, self.dtype)
        xk = self.cast(xk, self.dtype)
        # xq, xk: [bs, n_head/n_kv_head, seq/1, head_dim]
      
        freqs_cos, freqs_sin, swap_mask = freqs_cis
        mul = self.mul if self.is_first_iteration else self.mul_inc
        if self.use_rope_slice:
            xq_out = self.add(mul(xq, freqs_cos), mul(self.slice_half(xq), freqs_sin))
            xk_out = self.add(mul(xk, freqs_cos), mul(self.slice_half(xk), freqs_sin))
        else:
            xq_out = self.add(mul(xq, freqs_cos), mul(self.rotate_half(xq, swap_mask), freqs_sin))
            # [4, 32, 2048, 32] [1, 1, 2048, 128]
            xk_out = self.add(mul(xk, freqs_cos), mul(self.rotate_half(xk, swap_mask), freqs_sin))

        xq_out = self.cast(xq_out, original_type)
        xk_out = self.cast(xk_out, original_type)
        return xq_out, xk_out

    def shard(self, parallel_config):
        """sharding for rotary embedding"""
        # dp = parallel_config.data_parallel
        # mp = parallel_config.model_parallel
        dp = 2
        mp = 1
        strategy_in = (dp, mp, 1, 1)
        self.add.shard((strategy_in, strategy_in))
        self.bmm_swap.shard((strategy_in, (1, 1)))
        self.mul.shard((strategy_in, (1, 1)))
        self.mul_inc.shard((strategy_in, (strategy_in[0], 1, 1, 1)))
        self.neg.shard((strategy_in,))
        self.slice.shard((strategy_in,))
        self.concat.shard((strategy_in, strategy_in))
        
        
class LlamaRotaryEmbedding(Cell):
    r"""
    Rotary Position Embedding.

    Args:
            - **head_dim** (int): The dim of multi head attention.
            - **compute_dtype** (mstype): The compute type, default mstype.float16.
            - **parallel_config** (dict): - Parallel Config.
    Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

    Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(self, head_dim=128, compute_dtype=mstype.float32):
        super().__init__(auto_prefix=False)
        head_dim = 32
        self.head_dim = head_dim
        self.dtype = compute_dtype

        self.add = P.Add()
        self.bmm_swap = P.BatchMatMul()
        self.mul = P.Mul()

        self.cast = P.Cast()

    def rotate_half(self, x, swap_mask):
        # [bs, n_head/n_kv_head, seq/1, head_dim], [head_dim, head_dim]
        x = self.bmm_swap(x, swap_mask)
        return x

    def construct(self, xq: Tensor, xk: Tensor, freqs_cis):
        # print("run construct here")
        """Forward of rotary position embedding."""
        original_type = xq.dtype
        xq = self.cast(xq, self.dtype)
        xk = self.cast(xk, self.dtype)
        # xq, xk: [bs, n_head/n_kv_head, seq/1, head_dim]
        freqs_cos, freqs_sin, swap_mask = freqs_cis
        # fangxt change
        # freqs_cos_kv, freqs_sin_kv, swap_mask_kv = freqs_cis_kv
        xq_out = self.add(self.mul(xq, freqs_cos),
                          self.mul(self.rotate_half(xq, swap_mask), freqs_sin))
        # x.shape [4, 32, 2048, 32] and y.shape = [1, 1, 2048, 128]
        
        xk_out = self.add(self.mul(xk, freqs_cos),
                          self.mul(self.rotate_half(xk, swap_mask), freqs_sin))
        # xk_out = self.add(self.mul(xk, freqs_cos_kv),
        #           self.mul(self.rotate_half(xk, swap_mask_kv), freqs_sin_kv))
        # k: [1, 32, 2048, 32] and y.shape = [1, 1, 2048, 128]

        xq_out = self.cast(xq_out, original_type)
        xk_out = self.cast(xk_out, original_type)
        return xq_out, xk_out

    def shard(self, strategy_in):
        self.add.shard((strategy_in, strategy_in))
        self.bmm_swap.shard((strategy_in, (1, 1)))
        self.mul.shard((strategy_in, (strategy_in[0], 1, 1, 1)))


class LlamaEmbedding(Cell):
    """
    Embedding Layer.

    Args:
            - **vocab_size** (int): Size of the dictionary of embeddings.
            - **embedding_size** (int): The size of each embedding vector.
            - **param_init_type** (mstype): The param init type, default mstype.float32.
            - **parallel_config** (TransformerOpParallelConfig): The parallel config of network. Default
                `default_embedding_parallel_config`, an instance of `EmbeddingOpParallelConfig` with default args.
            - **param_init** (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the embedding_table.
                Refer to class `initializer` for the values of string when a string
                is specified. Default: 'normal'.
    Inputs:
            - **input_ids** (Tensor) - The tokenized inputs with datatype int32 with shape (batch_size, seq_length)

    Outputs:
            - **output** (Tensor) - The embedding vector for the input with shape (batch_size,
              seq_length, embedding_size).
    """

    @_LogActionOnce(m_logger=logger, key='Embedding',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(vocab_table_size=Validator.check_positive_int,
                                embedding_size=Validator.check_positive_int)
    def __init__(self, vocab_table_size, embedding_size, param_init_type=mstype.float32, param_init='normal'):
        super().__init__()
        self.vocab_table_size = vocab_table_size
        self.embedding_size = embedding_size
        self.embedding_weight = Parameter(
            initializer(param_init, [self.vocab_table_size, self.embedding_size], dtype=param_init_type),
            name='embedding_weight', parallel_optimizer=False)
        self.gather = P.Gather()

    def construct(self, input_ids):
        """Forward of vocab embedding."""
        _check_input_dtype(F.dtype(input_ids), "input_ids", [mstype.int32, mstype.int64], self.cls_name)
        output = self.gather(self.embedding_weight, input_ids, 0)
        return output

    def shard(self, parallel_config):
        """sharding for embedding"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        if parallel_config.vocab_emb_dp:
            self.gather.shard(((1, 1), (dp, 1)))
            logger.info(f"Using {dp} data parallel for the embedding lookup.")
        else:
            if self.vocab_table_size % mp != 0:
                raise ValueError(f"The vocab size of the embedding {self.vocab_table_size} must be a "
                                 f"multiple of parallel_config.model_parallel {mp}.")
            self.gather.shard(((mp, 1), (dp, 1)))
            logger.info(f"Using {dp} data parallel and {mp} "
                        f"model parallel for the embedding lookup.")


class LlamaRMSNorm(nn.Cell):
    r"""
    A self-defined RMSNorm operation using reduce mean.

        Args:
            dim (tuple): The shape of the input tensor
            eps (float): The epsilon value of the denominator. Default 1e-5.
            compute_type: The compute type.
        Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

        Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """
    def __init__(self, dim, eps=1e-6, compute_type=mstype.float32):
        super(LlamaRMSNorm, self).__init__()
        self.eps = eps
        self.weight = Parameter(initializer('ones', (dim,), dtype=mstype.float32), parallel_optimizer=False)
        self.square = P.Square()
        self.mean = P.ReduceMean(keep_dims=True)
        self.add = P.Add()
        self.rsqrt = P.Rsqrt()
        self.mul = P.Mul()
        self.mul2 = P.Mul()
        self.cast = P.Cast()
        self.compute_type = compute_type

    def _norm(self, x):
        norm_factor = self.square(x)
        norm_factor = self.mean(norm_factor, -1)
        norm_factor = self.add(norm_factor, self.eps)
        norm_factor = self.rsqrt(norm_factor)
        x = self.cast(x, mstype.float16)
        norm_factor = self.cast(norm_factor, mstype.float16)
        return self.mul(x, norm_factor)

    def construct(self, x):
        """Forward of RMSNorm."""
        original_type = x.dtype
        x = self.cast(x, self.compute_type)
        output = self._norm(x)
        output = self.cast(output, mstype.float16)
        weight = self.cast(self.weight, mstype.float16)
        output = self.mul2(output, weight)
        output = self.cast(output, original_type)
        return output

    def shard(self, strategy_in):
        """Parallel strategy configuratiuon interface."""
        self.square.shard((strategy_in,))
        self.mean.shard((strategy_in,))
        self.rsqrt.shard((strategy_in,))
        self.add.shard((strategy_in, ()))
        self.mul.shard((strategy_in, strategy_in))
        self.mul2.shard((strategy_in, (1,)))


class LlamaFeedForward(Cell):
    r"""
    LLaMA FeedForward.

    .. math::
            (xW_1 * xW_3)W_2

        Inputs:
            - **x** (Tensor) - should be `[batch, seq_length, hidden_size] or [batch * seq_length, hidden_size]`.
              Float tensor.

        Outputs:
            Tensor, the output of this layer after mapping. The shape is `[batch, seq_length, hidden_size] or
            [batch * seq_length, hidden_size]`.

        Raises:
            ValueError: `hidden_dim` is not a multiple of the model parallel way.
            ValueError: `dim` is not a multiple of the model parallel way.
    """

    @_LogActionOnce(m_logger=logger, key='FeedForward',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(dim=Validator.check_positive_int,
                                hidden_dim=Validator.check_positive_int,
                                multiple_of=Validator.check_positive_int,
                                compute_dtype=_valid_value_checks([mstype.float32, mstype.float16],
                                                                  "FeedForward"),
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16],
                                                                    "FeedForward"))
    def __init__(self, dim,
                 hidden_dim,
                 multiple_of,
                 hidden_act=LlamaSiLU,
                 ffn_dim_multiplier=None,
                 compute_dtype=mstype.float16,
                 param_init_type=mstype.float32,
                 layer_id=None):
        super().__init__()
        
        if hidden_act is None or not (isinstance(hidden_act, str) or issubclass(hidden_act, nn.Cell)):
            raise TypeError(f"For FeedForward cell, the hidden_act should str type or nn.Cell type, "
                            f"but got {hidden_act}.")

        if ffn_dim_multiplier is not None:
            hidden_dim = int((ffn_dim_multiplier + 0.01) * hidden_dim)
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * \
            ((hidden_dim + multiple_of - 1) // multiple_of)

        self.dtype = compute_dtype
        self.hidden_act = hidden_act
        
        self.dim = dim
        hidden_dim = 14336
        self.hidden_dim = hidden_dim

        self.mul = P.Mul()
        self.cast = P.Cast()
        self.w1 = Linear(in_channels=dim,
                         out_channels=hidden_dim,
                         activation=hidden_act,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)

        self.w2 = Linear(in_channels=hidden_dim,
                         out_channels=dim,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)

        self.w3 = Linear(in_channels=dim,
                         out_channels=hidden_dim,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)
        
        self.mul2 = P.Mul()
        self.matmul = P.MatMul()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.topk = P.TopK()
        self.gather = P.Gather()
        self.add = P.Add()
        
        
        patterns = np.load(f"/home/ma-user/work/r0.8_fangxt/param/patterns_{layer_id}.npy")
        self.patterns = Tensor(patterns, dtype=mstype.float16)
        
        self.k_molecular = 16
        self.k_text = 16
        logger.info("&&&&_&&&" * 200)

    def construct(self, x, molecular_mask):
        """Forward process of the FeedForward"""
        _check_input_dtype(F.dtype(x), "x", [mstype.float32, mstype.float16], self.cls_name)
        x = self.cast(x, self.dtype)
        # [bs, seq, hidden_dim] or [bs * seq, hidden_dim]
        gate = self.w1(x) # dp,1 -> dp, mp
        hidden = self.w3(x) # dp,1 -> dp, mp
        hidden = self.mul(hidden, gate) # dp,mp -> dp, mp
        output = self.w2(hidden) # dp,mp -> dp, 1
        return output

    # def construct(self, x, molecular_mask):
    #     """Forward process of the FeedForward"""
    #     _check_input_dtype(F.dtype(x), "x", [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
#         x = self.cast(x, self.dtype)
#         # print(x.shape)
#         gate = self.w1(x) # dp,1 -> dp, mp
#         hidden = self.w3(x) # dp,1 -> dp, mp
#         hidden_states = self.mul(hidden, gate) # dp,mp -> dp, mp
        
#         # [1, 4096, 11008] [1, 1, 11008]
        
#         molecular_mask = self.cast(molecular_mask, mstype.bool_)
#         new_molecular_mask = ops.unsqueeze(molecular_mask.copy(), -1)
        
#         other1 = ops.ones(new_molecular_mask.shape, mstype.int32)
#         new_reversal_mask = ops.bitwise_xor(new_molecular_mask, other1)

#         molecular_mask = self.reshape(molecular_mask, (-1, 1))
#         other2 = ops.ones(molecular_mask.shape, mstype.int32)
#         reversal_mask = ops.bitwise_xor(molecular_mask, other2)
#         # reversal_mask = molecular_mask
#         # if self.patterns is not None:
            
#         k_molecular = self.k_molecular
#         k_text = self.k_text
#         bsz, seq_len, hidden_size = hidden_states.shape

#         # hidden_states_relu = hidden_states.copy()
#         hidden_states_relu = hidden_states
#         hidden_states_relu = ops.stop_gradient(hidden_states_relu)
#         hidden_states_relu = self.reshape(hidden_states_relu, (-1, hidden_size))
#         score = self.matmul(hidden_states_relu, self.transpose(self.patterns, (1, 0)))

#         molecular_score = self.mul2(score, molecular_mask)
#         text_score = self.mul2(score, reversal_mask)
#         labels_topk_molecular = self.topk(molecular_score, k_molecular)[1]
#         labels_topk_molecular = self.reshape(labels_topk_molecular, (bsz, seq_len, k_molecular))

#         labels_topk_text = self.topk(text_score, k_text)[1]
#         labels_topk_text = self.reshape(labels_topk_text, (bsz, seq_len, k_text))

#         patterns = self.cast(self.patterns, mstype.int32)
#         cur_mask_molecular = self.gather(patterns, labels_topk_molecular, 0).sum(-2)
#         cur_mask_text = self.gather(patterns, labels_topk_text, 0).sum(-2)

#         cur_mask = self.add(self.mul2(cur_mask_molecular, new_molecular_mask), self.mul2(cur_mask_text, new_reversal_mask))
#         cur_mask = cur_mask > 0
#         cur_mask = self.cast(cur_mask, hidden_states.dtype)
#         hidden_states_new = self.mul(hidden_states, cur_mask)
#         output = self.w2(hidden_states_new)
#         return output

#     def construct(self, x, molecular_mask):
#         """Forward process of the FeedForward"""
#         _check_input_dtype(F.dtype(x), "x", [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
#         x = self.cast(x, self.dtype)
#         # [bs, seq, hidden_dim] or [bs * seq, hidden_dim]
#         # origin code
#         # gate = self.w1(x) # dp,1 -> dp, mp
#         # hidden = self.w3(x) # dp,1 -> dp, mp
#         # hidden = self.mul(hidden, gate) # dp,mp -> dp, mp
#         # output = self.w2(hidden) # dp,mp -> dp, 1

#         gate = self.w1(x) # dp,1 -> dp, mp
#         hidden = self.w3(x) # dp,1 -> dp, mp
#         hidden_states = self.mul(hidden, gate) # dp,mp -> dp, mp
        
#         # [1, 4096, 11008] [1, 1, 11008]
        
#         molecular_mask = self.cast(molecular_mask, mstype.bool_)
#         molecular_mask = self.reshape(molecular_mask, (-1, 1))
#         reversal_mask = ~molecular_mask
#         # if self.patterns is not None:
            
#         k_molecular = self.k_molecular
#         k_text = self.k_text
#         bsz, seq_len, hidden_size = hidden_states.shape

#         # hidden_states_relu = hidden_states.copy()
#         hidden_states_relu = hidden_states
#         hidden_states_relu = ops.stop_gradient(hidden_states_relu)
#         hidden_states_relu = self.reshape(hidden_states_relu, (-1, hidden_size))
#         score = self.matmul(hidden_states_relu, self.transpose(self.patterns, (1, 0)))

#         molecular_score = self.mul2(score, molecular_mask)
#         text_score = self.mul2(score, reversal_mask)
#         labels_topk_molecular = self.topk(molecular_score, k_molecular)[1]
#         labels_topk_molecular = self.reshape(labels_topk_molecular, (bsz, seq_len, k_molecular))

#         labels_topk_text = self.topk(text_score, k_text)[1]
#         labels_topk_text = self.reshape(labels_topk_text, (bsz, seq_len, k_text))

#         patterns = self.cast(self.patterns, mstype.int32)
#         cur_mask_molecular = self.gather(patterns, labels_topk_molecular, 0).sum(-2)
#         cur_mask_text = self.gather(patterns, labels_topk_text, 0).sum(-2)
#         cur_mask = self.add(cur_mask_molecular, cur_mask_text)
#         cur_mask = cur_mask > 0
#         cur_mask = self.cast(cur_mask, hidden_states.dtype)
#         hidden_states_new = self.mul(hidden_states, cur_mask)
#         # print("hidden_states",(hidden_states_new- hidden_states).abs().sum())
        
#         output = self.w2(hidden_states_new)
#         return output

    def shard(self, parallel_config):
        """sharding for feedforward"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        if self.hidden_dim % mp != 0:
            raise ValueError("For 'FeedForward', the class variable 'hidden_dim' must be a multiple of the"
                             "num of model parallel, but got the hidden_dim is {} and the num of model "
                             "parallel is {}.".format(self.hidden_dim, mp))
        if self.dim % mp != 0:
            raise ValueError("For 'FeedForward', the class variable 'dim' must be a multiple of the num of "
                             "model parallel, but got the dim is {} and the num of model parallel is {}."
                             .format(self.dim, mp))
        self.w1.shard(((dp, 1), (mp, 1)), strategy_activation=((dp, mp),))
        self.w1.activation.shard(((dp, mp),))
        self.w2.shard(((dp, mp), (1, mp)))
        self.w3.shard(((dp, 1), (mp, 1)))
        self.mul.shard(((dp, mp), (dp, mp)))