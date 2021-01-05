import logging
import os
from multiprocessing import Pool, cpu_count
from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid: int, text_a: str, text_b: str = None, labels: List[str] = None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels: (Optional) string[]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


class InputFeatures(object):
    def __init__(self, input_ids: List[int], input_mask: List[int], segment_ids: List[int], label_ids: List[int]):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, truncate_mode: str):
        self.truncate_mode = truncate_mode

    def get_train_examples(self, data_dir: str):
        """Gets a collection of `InputExample`s for the train set."""
        data_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        return self._create_examples(data_df)

    def get_test_examples(self, data_dir: str):
        """Gets a collection of `InputExample`s for the dev set."""
        data_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
        return self._create_examples(data_df)

    def get_labels(self):
        """Gets the list of labels for this data set."""
        try:
            return self.labels
        except AttributeError:
            logging.warning("Attribute 'labels' was not provided.")
            return None

    def _create_examples(self, df: DataFrame):
        raise NotImplementedError

    @staticmethod
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def truncate(self, tokens_a: List[str], max_seq_length: int):
        '''Implementing a head + tail truncation as `How to Fine-Tune BERT for Text Classification?`
        has shown model's improvements using such a technique.'''
        if self.truncate_mode == "head_tail" and max_seq_length > 128 + 1:
            return tokens_a[:(128 - 1)] + tokens_a[-(max_seq_length - 1 - 128):]
        return tokens_a[:(max_seq_length - 2)]

    def convert_examples_to_features(self, examples, label_list, max_seq_length,
                                     tokenizer, output_mode,
                                     cls_token_at_end=False, pad_on_left=False,
                                     cls_token='[CLS]', sep_token='[SEP]',
                                     cls_token_segment_id=1, pad_token_segment_id=0):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """

        if output_mode == "multi-label-classification":
            label_map = None
        else:
            label_map = {label: i for i, label in enumerate(label_list)}

        examples = [(example, label_map, max_seq_length, tokenizer, output_mode, cls_token_at_end, cls_token, sep_token,
                     cls_token_segment_id, pad_on_left, pad_token_segment_id) for example in examples]
        process_count = cpu_count() - 2

        with Pool(process_count) as p:
            features = list(tqdm(p.imap(self.convert_example_to_feature, examples, chunksize=100), total=len(
                examples)))

        return features

    def convert_example_to_feature(self, example_row, pad_token=0,
                                   sequence_a_segment_id=0, sequence_b_segment_id=1,
                                   mask_padding_with_zero=True):
        example, label_map, max_seq_length, tokenizer, output_mode, cls_token_at_end, cls_token, sep_token, \
        cls_token_segment_id, pad_on_left, pad_token_segment_id = example_row
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = self.truncate(tokens_a, max_seq_length=max_seq_length)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.labels]
            assert label_id >= 0
            assert label_id < len(label_map)
        elif output_mode == "multi-label-classification":
            label_id = np.array(example.labels, dtype=bool)
            assert len(label_id) == len(self.labels)
        else:
            raise ValueError("Unsupported 'output_mode'.")

        return InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             label_ids=label_id)


class MultiLabelProcessor(DataProcessor):
    """Processor for multi-label classification problems."""

    def __init__(self, labels: List[str], truncate_mode: str):
        super().__init__(truncate_mode)
        self.labels = labels

    def _create_examples(self, df: DataFrame, labels_available: bool = True):
        """Creates examples for the training and test sets."""
        logging.info("Creating {} examples.".format(len(df)))
        logging.info("Sample row: {}".format(df.iloc[0]))
        examples = []
        for row in df.values:
            guid = row[0]
            text_a = row[1]
            if labels_available:
                labels = row[2:]
            else:
                labels = []
            examples.append(
                InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples


class MultiClassProcessor(DataProcessor):
    """Processor for multi-class classification problems."""

    def __init__(self, labels: List[str], truncate_mode: str):
        super().__init__(truncate_mode)
        self.labels = labels

    def _create_examples(self, df: DataFrame, labels_available: bool = True):
        """Creates examples for the training and dev sets."""
        logging.info("Creating {} examples.".format(len(df)))
        logging.info("Sample row: {}".format(df.iloc[0]))
        examples = []
        for i, row in enumerate(df.values):
            guid = i
            text_a = row[0]
            if labels_available:
                label = row[1]
            else:
                label = None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, labels=label))
        return examples
