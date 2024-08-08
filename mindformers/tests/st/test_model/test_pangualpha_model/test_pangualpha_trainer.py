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
Test module for testing the pangualpha interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_llm_model/test_pangualpha_trainer.py
"""
import numpy as np
import pytest

import mindspore as ms

from mindspore.dataset import GeneratorDataset
from mindformers.models.pangualpha.pangualpha import PanguAlphaHeadModel
from mindformers.models.pangualpha.pangualpha_config import PanguAlphaConfig
from mindformers import Trainer, TrainingArguments

ms.set_context(mode=0)


def generator_train():
    """train dataset generator"""
    seq_len = 129
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    position_ids = np.ones((seq_len)).astype(np.int32)
    attention_mask = np.ones((seq_len, seq_len)).astype(np.int32)
    train_data = (input_ids, position_ids, attention_mask)
    for _ in range(16):
        yield train_data


def generator_eval():
    """eval dataset generator"""
    seq_len = 128
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    # input_mask = np.ones_like(input_ids)
    train_data = (input_ids)
    for _ in range(16):
        yield train_data


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
class TestPanguAlphaTrainerMethod:
    """A test class for testing pipeline."""

    def setup_method(self):
        """init task trainer."""
        args = TrainingArguments(batch_size=4, num_train_epochs=1)
        train_dataset = GeneratorDataset(generator_train, column_names=["input_ids", "position_id", "attention_mask"])
        eval_dataset = GeneratorDataset(generator_eval, column_names=["input_ids"])
        train_dataset = train_dataset.batch(batch_size=4)
        eval_dataset = eval_dataset.batch(batch_size=4)

        model_config = PanguAlphaConfig(num_layers=2,
                                        hidden_size=128,
                                        ffn_hidden_size=128*4,
                                        num_heads=2,
                                        seq_length=128)
        model = PanguAlphaHeadModel(model_config)

        self.task_trainer = Trainer(task='text_generation',
                                    model=model,
                                    args=args,
                                    train_dataset=train_dataset,
                                    eval_dataset=eval_dataset)

    def test_train(self):
        """
        Feature: Trainer.train()
        Description: Test trainer for train.
        Expectation: TypeError, ValueError, RuntimeError
        """
        self.task_trainer.config.runner_config.epochs = 1
        self.task_trainer.train()

    def test_eval(self):
        """
        Feature: Trainer.evaluate()
        Description: Test trainer for evaluate.
        Expectation: TypeError, ValueError, RuntimeError
        """
        self.task_trainer.model.set_train(False)
        self.task_trainer.evaluate()

    def test_predict(self):
        """
        Feature: Trainer.predict()
        Description: Test trainer for predict.
        Expectation: TypeError, ValueError, RuntimeError
        """
        self.task_trainer.predict(input_data="今天天气如何？", max_length=20, repetition_penalty=1, top_k=3, top_p=1)

    def test_finetune(self):
        """
        Feature: Trainer.finetune()
        Description: Test trainer for finetune.
        Expectation: TypeError, ValueError, RuntimeError
        """
        self.task_trainer.config.runner_config.epochs = 1
        self.task_trainer.finetune(finetune_checkpoint=True)
