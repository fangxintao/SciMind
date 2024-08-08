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

"""TextGenerationPipeline"""
import os.path
from typing import Optional, Union

import mindspore
from mindspore import Model, Tensor

from ..auto_class import AutoConfig, AutoModel, AutoProcessor
from ..mindformer_book import MindFormerBook
from ..models import BaseModel, BaseTokenizer
from ..tools.register import MindFormerModuleType, MindFormerRegister
from .base_pipeline import BasePipeline

__all__ = ['TextGenerationPipeline']


def _setup_support_list(support_model_list):
    support_list = []
    for support_model in support_model_list:
        support_list.extend(MindFormerBook.get_model_support_list().get(support_model))
    return support_list


@MindFormerRegister.register(MindFormerModuleType.PIPELINE, alias="text_generation")
class TextGenerationPipeline(BasePipeline):
    r"""Pipeline for Text Generation

    Args:
        model (Union[str, BaseModel]):
            The model used to perform task, the input could be a supported model name, or a model instance
            inherited from BaseModel.
        tokenizer (Optional[BaseTokenizer]):
            A tokenizer (None or Tokenizer) for text processing.
        **kwargs:
            Specific parametrization of `generate_config` and/or additional model-specific kwargs that will be
            forwarded to the `forward` function of the model. Supported `generate_config` keywords can be
            checked in [`GenerationConfig`]'s documentation. Mainly used Keywords are shown below:

            max_length(int): The maximum length the generated tokens can have. Corresponds to the length of
                the input prompt + `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
            max_new_tokens (int): The maximum numbers of tokens to generate, ignoring the number of
                tokens in the prompt.
            do_sample(bool): Whether to do sampling on the candidate ids.
                If set True it will be enabled, and set it to be False to disable the sampling,
                equivalent to topk 1.
                If set None, it follows the setting in the configureation in the model.
            top_k(int): Determine the topK numbers token id as candidate. This should be a positive number.
                If set None, it follows the setting in the configureation in the model.
            top_p(float): The accumulation probability of the candidate token ids below the top_p
                will be select as the condaite ids. The valid value of top_p is between (0, 1]. If the value
                is larger than 1, top_K algorithm will be enabled. If set None, it follows the setting in the
                configureation in the model.
            eos_token_id(int): The end of sentence token id. If set None, it follows the setting in the
                configureation in the model.
            pad_token_id(int): The pad token id. If set None, it follows the setting in the configureation
                in the model.
            repetition_penalty(float): The penalty factor of the frequency that generated words. The If set 1,
                the repetition_penalty will not be enabled. If set None, it follows the setting in the
                configureation in the model. Default None.

    Raises:
        TypeError:
            If input model and tokenizer's types are not corrected.
        ValueError:
            If the input model is not in support list.

    Examples:
        >>> from mindformers.pipeline import TextGenerationPipeline
        >>> text_generate = TextGenerationPipeline("gpt2")
        >>> output = text_generate("I love Beijing, because ")
    """
    _support_list = MindFormerBook.get_pipeline_support_task_list()['text_generation'].keys()
    _model_build_kwargs = ["batch_size", "use_past", "seq_length"]
    return_name = 'text_generation'

    def __init__(self, model: Union[str, BaseModel, Model],
                 tokenizer: Optional[BaseTokenizer] = None,
                 **kwargs):
        batch_size = kwargs.get("batch_size", None)
        if isinstance(model, str):
            if model in self._support_list or os.path.isdir(model):
                if tokenizer is None:
                    tokenizer = AutoProcessor.from_pretrained(model).tokenizer
                # build model using parameters
                model_config = AutoConfig.from_pretrained(model)
                for build_arg in self._model_build_kwargs:
                    model_config[build_arg] = kwargs.pop(build_arg, \
                        model_config.get(build_arg, None))
                model = AutoModel.from_config(model_config)
            else:
                raise ValueError(f"{model} is not supported by {self.__class__.__name__},"
                                 f"please selected from {self._support_list}.")

        if not isinstance(model, (BaseModel, Model)):
            raise TypeError(f"model should be inherited from BaseModel or Model, but got type {type(model)}.")

        if tokenizer is None:
            raise ValueError(f"{self.__class__.__name__}"
                             " requires for a tokenizer.")

        super().__init__(model, tokenizer, **kwargs)
        self._batch_size = batch_size

    def _sanitize_parameters(self, **pipeline_parameters):
        r"""Sanitize Parameters

        Args:
            pipeline_parameters (Optional[dict]):
                The parameter dict to be parsed.
        """
        preprocess_keys = ['keys', 'add_special_tokens']
        preprocess_params = {}
        for item in preprocess_keys:
            if item in pipeline_parameters:
                preprocess_params[item] = pipeline_parameters.pop(item)

        postprocess_params = {}

        # all other pipeline_parameters are passed to text generator to handle
        forward_kwargs = pipeline_parameters

        return preprocess_params, forward_kwargs, postprocess_params

    def preprocess(self, inputs: Union[str, dict, Tensor],
                   **preprocess_params):
        r"""The Preprocess For Translation

        Args:
            inputs (Union[str, dict, Tensor]):
                The text to be classified.
            preprocess_params (dict):
                The parameter dict for preprocess.

        Return:
            Processed text.
        """
        add_special_tokens = preprocess_params.get('add_special_tokens', True)
        if isinstance(inputs, dict):
            keys = preprocess_params.get('keys', None)
            default_src_language_name = 'text'
            feature_name = keys.get('src_language', default_src_language_name) if keys else default_src_language_name

            inputs = inputs[feature_name]
            if isinstance(inputs, mindspore.Tensor):
                inputs = inputs.asnumpy().tolist()
        # for batch inputs, pad to longest
        input_ids = self.tokenizer(inputs,
                                   return_tensors=None,
                                   add_special_tokens=add_special_tokens,
                                   padding=True)["input_ids"]
        return {"input_ids": input_ids}

    def forward(self, model_inputs: dict,
                **forward_params):
        r"""The Forward Process of Model

        Args:
            model_inputs (dict):
                The output of preprocess.
            forward_params (dict):
                The parameter dict for model forward.
        """
        forward_params.pop("None", None)
        input_ids = model_inputs["input_ids"]
        output_ids = self.network.generate(input_ids, **forward_params)
        return {"output_ids": output_ids}

    def postprocess(self, model_outputs: dict,
                    **postprocess_params):
        r"""Postprocess

        Args:
            model_outputs (dict):
                Outputs of forward process.

        Return:
            Translation results.
        """
        outputs = self.tokenizer.decode(model_outputs["output_ids"], skip_special_tokens=True)
        return [{self.return_name + '_text': outputs}]
