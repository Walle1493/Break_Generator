# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """


import csv
import glob
import json
import logging
import os
from typing import List

import tqdm

import torch
from torch.utils.data import TensorDataset
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question, label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.label = label


class InputFeatures(object):
    def __init__(self, example_id, input_ids, attention_mask, token_type_ids, label):
        self.example_id = example_id
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class BinaryProcessor(DataProcessor):
    """Processor for the Musique data set."""

    def get_train_examples(self, data_dir, filename):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        dataset = os.path.join(data_dir, filename)
        return self._create_examples(dataset, "train")

    def get_dev_examples(self, data_dir, filename):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        dataset = os.path.join(data_dir, filename)
        return self._create_examples(dataset, "dev")

    def get_test_examples(self, data_dir, filename):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        dataset = os.path.join(data_dir, filename)
        return self._create_examples(dataset, "test")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        with open(dataset) as f:
            lines = json.load(f)
        examples = []
        for (_, data_raw) in enumerate(lines):
            musique_id = "%s-%s" % (set_type, data_raw["id"])
            question = data_raw["question"]
            qtype = data_raw["id"].split("_")[0]
            # binary classification
            if qtype in ["2hop", "3hop1", "4hop1"]:
                label = 0
            elif qtype in ["3hop2", "4hop2", "4hop3"]:
                label = 1
            examples.append(
                InputExample(
                    example_id=musique_id,
                    question=question,
                    label=label,
                )
            )
        return examples


class SixProcessor(DataProcessor):
    """Processor for the Musique data set."""

    def get_train_examples(self, data_dir, filename):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        dataset = os.path.join(data_dir, filename)
        return self._create_examples(dataset, "train")

    def get_dev_examples(self, data_dir, filename):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        dataset = os.path.join(data_dir, filename)
        return self._create_examples(dataset, "dev")

    def get_test_examples(self, data_dir, filename):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        dataset = os.path.join(data_dir, filename)
        return self._create_examples(dataset, "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3, 4, 5]

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        with open(dataset) as f:
            lines = json.load(f)
        examples = []
        for (_, data_raw) in enumerate(lines):
            musique_id = "%s-%s" % (set_type, data_raw["id"])
            question = data_raw["question"]
            qtype = data_raw["id"].split("_")[0]
            # six classification
            if qtype == "2hop":
                label = 0
            elif qtype == "3hop1":
                label = 1
            elif qtype == "3hop2":
                label = 2
            elif qtype == "4hop1":
                label = 3
            elif qtype == "4hop2":
                label = 4
            elif qtype == "4hop3":
                label = 5
            examples.append(
                InputExample(
                    example_id=musique_id,
                    question=question,
                    label=label,
                )
            )
        return examples


class WikiProcessor(DataProcessor):
    """Processor for the 2WikiMultiHopQA data set."""
    pass


class HotpotqaProcessor(DataProcessor):
    """Processor for the Hotpotqa data set."""
    pass


class RaceProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        high = os.path.join(data_dir, "train/high")
        middle = os.path.join(data_dir, "train/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        high = os.path.join(data_dir, "dev/high")
        middle = os.path.join(data_dir, "dev/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        high = os.path.join(data_dir, "test/high")
        middle = os.path.join(data_dir, "test/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_txt(self, input_dir):
        lines = []
        files = glob.glob(input_dir + "/*txt")
        for file in tqdm.tqdm(files, desc="read files"):
            with open(file, "r", encoding="utf-8") as fin:
                data_raw = json.load(fin)
                data_raw["race_id"] = file
                lines.append(data_raw)
        return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data_raw) in enumerate(lines):
            race_id = "%s-%s" % (set_type, data_raw["race_id"])
            article = data_raw["article"]
            for i in range(len(data_raw["answers"])):
                truth = str(ord(data_raw["answers"][i]) - ord("A"))
                question = data_raw["questions"][i]
                options = data_raw["options"][i]

                examples.append(
                    InputExample(
                        example_id=race_id,
                        question=question,
                        contexts=[article, article, article, article],  # this is not efficient but convenient
                        endings=[options[0], options[1], options[2], options[3]],
                        label=truth,
                    )
                )
        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
    
        inputs = tokenizer.encode_plus(example.question, add_special_tokens=True, max_length=max_length,)
        if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
            logger.info(
                "Attention! you are cropping tokens (swag task is ok). "
                "If you are training ARC and RACE and you are poping question + options,"
                "you need to try to use a bigger max seq length!"
            )

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length

        label = example.label

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("example_id: {}".format(example.example_id))
            logger.info("input_ids: {}".format(" ".join(map(str, input_ids))))
            logger.info("attention_mask: {}".format(" ".join(map(str, attention_mask))))
            logger.info("token_type_ids: {}".format(" ".join(map(str, token_type_ids))))
            logger.info("label: {}".format(label))

        features.append(InputFeatures(
            example_id=example.example_id, 
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids, 
            label=label,
        ))

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([feature.input_ids for feature in features], dtype=torch.long)
    all_attention_masks = torch.tensor([feature.attention_mask for feature in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([feature.token_type_ids for feature in features], dtype=torch.long)
    all_label_ids = torch.tensor([feature.label for feature in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_label_ids)

    return dataset
