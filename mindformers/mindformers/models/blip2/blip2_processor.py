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
BLIP2Processor
"""

from typing import Optional, Union, List
import numpy as np
import PIL
import PIL.Image

import mindspore as ms

from mindformers.mindformer_book import MindFormerBook
from mindformers.models.base_processor import BaseProcessor, BaseImageProcessor
from mindformers.dataset.transforms.vision_transforms import (
    BatchPILize,
    BatchResize,
    BatchToTensor,
    BatchNormalize
)
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class Blip2ImageProcessor(BaseImageProcessor):
    """
    Blip2ImageProcessor.

    Args:
        image_size (int): The target size.

    Examples:
        >>> from mindformers import Blip2ImageProcessor
        >>> from mindformers.tools.image_tools import load_image
        >>> processor = Blip2ImageProcessor(image_size=224)
        >>> image = load_image("https://ascend-repo-modelzoo.obs.cn-east-2."
            "myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
        >>> processor(image)
            Tensor(shape=[1, 3, 224, 224], dtype=Float32, value=
            [[[[-1.55868769e+00, -1.52949083e+00, ... -1.48569560e+00, -1.48569560e+00],
            [-1.54408932e+00, -1.52949083e+00, ... -1.50029397e+00, -1.50029397e+00],
            [-1.52949083e+00, -1.52949083e+00, ... -1.50029397e+00, -1.50029397e+00],
            ...
            [-1.38067937e+00, -1.48021984e+00, ... -1.30957901e+00, -1.40911949e+00],
            [-1.46599972e+00, -1.43755960e+00, ... -1.48021984e+00, -1.43755960e+00],
            [-1.40911949e+00, -1.28113890e+00, ... -1.48021984e+00, -1.43755960e+00]]]])
    """

    def __init__(self,
                 image_size: Optional[int] = 224,
                 interpolation: Optional[str] = 'bicubic',
                 mean=None,
                 std=None,
                 is_hwc=False):
        self.pilize = BatchPILize()
        super(Blip2ImageProcessor, self).__init__()
        if isinstance(image_size, int):
            image_size = (image_size,) * 2
        self.resize = BatchResize(image_size, interpolation=interpolation)
        self.to_tensor = BatchToTensor()
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)
        self.normalize = BatchNormalize(mean, std, is_hwc)

    def preprocess(self, images: Union[ms.Tensor, PIL.Image.Image,
                                       np.ndarray, List[PIL.Image.Image]], **kwargs):
        r"""
        Preprocess Required By Base Processor.

        Args:
            images (ms.Tensor, PIL.Image, numpy.array, List[PIL.Image]): A batch of images.

        Return:
            A 4-rank tensor for a batch of images.
        """
        images = self.pilize(images)
        images = self.resize(images)
        images = self.to_tensor(images)
        images = self.normalize(images)

        kwargs.pop("other", None)
        if isinstance(images, list):
            return ms.Tensor(np.row_stack([np.expand_dims(item, axis=0) for item in images]))
        if len(images.shape) == 4:
            return ms.Tensor(images)
        return ms.Tensor(np.expand_dims(images, axis=0))

    def _bhwc_check(self, image_batch: Union[ms.Tensor, PIL.Image.Image,
                                             np.ndarray, List[PIL.Image.Image]]):
        r"""Bhwc_check"""
        if isinstance(image_batch, np.ndarray):
            if image_batch.shape[-1] == 3:
                return True
        if isinstance(image_batch, ms.Tensor):
            if image_batch.asnumpy().shape[-1] == 3:
                return True
        if isinstance(image_batch, (list, PIL.Image.Image)):
            return True
        return False


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class Blip2Processor(BaseProcessor):
    r"""Blip2 Processor,
    consists of a feature extractor (BaseFeatureEXtractor) for image input,
    and a tokenizer (BaseTokenizer) for text input.

    Args:
        image_processor (BaseImageProcessor): Used for process image data.
        tokenizer (BaseTokenizer): Used for process text data.
        max_length (Optional[int]): The length of text tokens.
        padding (Optional[str]): The padding strategy of tokenizer, [None, "max_length"].
        return_tensors (Optional[str]): The type of returned tensors for tokenizer, [None, "ms"].

    Examples:
        >>> from mindformers import Blip2Processor
        >>> from mindformers.tools.image_tools import load_image
        >>> image = load_image("https://ascend-repo-modelzoo.obs.cn-east-2."
        ...  "myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
        >>> text = ["a boy", "a girl"]
        >>> Blip2Processor.show_support_list()
        INFO - support list of Blip2Processor is:
        INFO -    ['blip2_stage1_vit_g', 'blip2_stage1_classification']
        INFO - -------------------------------------
        >>> processor = Blip2Processor.from_pretrained('blip2_stage1_vit_g')
        INFO - processor built successfully!
        >>> processor(image, text)
        {'image': Tensor(shape=[1, 3, 224, 224], dtype=Float32, value=
    [[[[-1.55868769e+00, -1.52949083e+00, -1.55868769e+00 ... -1.48569560e+00, -1.48569560e+00],
       [-1.54408932e+00, -1.52949083e+00, -1.54408932e+00 ... -1.50029397e+00, -1.50029397e+00],
       [-1.52949083e+00, -1.52949083e+00, -1.52949083e+00 ... -1.50029397e+00, -1.50029397e+00],
       ...
       [-1.38067937e+00, -1.48021984e+00, -1.38067937e+00 ... -1.30957901e+00, -1.40911949e+00],
       [-1.46599972e+00, -1.43755960e+00, -1.26691878e+00 ... -1.48021984e+00, -1.43755960e+00],
       [-1.40911949e+00, -1.28113890e+00, -1.30957901e+00 ... -1.48021984e+00, -1.43755960e+00]
       ]]]),
       'text': Tensor(shape=[2, 32], dtype=Int32, value=
       [[ 101, 1037, 2879 ...    0,    0,    0],
       [ 101, 1037, 2611 ...    0,    0,    0]])}
    """
    _support_list = MindFormerBook.get_processor_support_list()['blip2']

    def __init__(self, image_processor, tokenizer,
                 max_length=32, padding='max_length', return_tensors='ms'):
        super(Blip2Processor, self).__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_length=max_length,
            padding=padding,
            return_tensors=return_tensors)
