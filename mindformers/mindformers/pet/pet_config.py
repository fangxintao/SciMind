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
Note: The config module of Parameter Efficient Tuning module.
"""
from mindformers.models.base_config import BaseConfig
from mindformers.models.utils import convert_mstype
from mindformers.pet.constants import PetType


__all__ = ['LoraConfig']


class PetConfig(BaseConfig):
    """
    Pet model config class which defines the tuning algorithm and pretrained for tuning.

    Args:
        pet_type: (`str`, *optional*, defaults to None): The Pet method type.
    """
    def __init__(self,
                 pet_type: str = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.pet_type = pet_type


class LoraConfig(PetConfig):
    """
    Lora tuning algorithm config.

    Args:
        lora_rank (`int`, *optional*, defaults to 8):
            The number of rows(columns) in LoRA matrices.
        lora_alpha (`int`, *optional*, defaults to 16):
            A constant in lora_rank.
        lora_dropout (`float`, *optional*, defaults to 0.01):
            The dropout rate, greater equal than 0 and less than 1.
        lora_a_init (`str`, *optional*, defaults to 'normal'):
            The initialization strategy of LoRA A matrix.
            Refers to (https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.common.initializer.html)
        lora_b_init (`str`, *optional*, defaults to 'zero'):
            The initialization strategy of LoRA B matrix.
            Refers to (https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.common.initializer.html)
        param_init_type (`str`, *optional*, defaults to 'float16'):
            The type of data in initialized tensor.
        compute_dtype (`str`, *optional*, defaults to 'float16'):
            The compute type of data.
        target_modules (`str`, *optional*, defaults None):
            The Layers that require replacement with LoRa algorithm.
        exclude_layers (`str`, *optional*, defaults None):
            The layers that do not require replacement with the LoRa algorithm.

    Returns:
        Class, LoraConfig.
    """
    def __init__(self,
                 lora_rank: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.01,
                 lora_a_init: str = 'normal',
                 lora_b_init: str = 'zero',
                 param_init_type: str = 'float16',
                 compute_dtype: str = 'float16',
                 target_modules: str = None,
                 exclude_layers: str = None,
                 **kwargs):
        super().__init__(pet_type=PetType.LORA.value, **kwargs)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_a_init = lora_a_init
        self.lora_b_init = lora_b_init
        self.param_init_type = convert_mstype(param_init_type)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.target_modules = target_modules
        self.exclude_layers = exclude_layers
