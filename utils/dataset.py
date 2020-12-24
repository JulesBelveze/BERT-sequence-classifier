import logging
import os

import torch
from torch.utils.data import TensorDataset


class Dataset(object):
    def __init__(self, task, tokenizer, processor, labels, truncate_mode):
        self.task = task
        self.tokenizer = tokenizer
        self.processor = processor(labels, truncate_mode)
        self.labels = labels

    def load_and_cache_examples(self, train: bool, **kwargs):
        output_mode = kwargs['output_mode']

        mode = 'train' if train else 'test'
        cached_features_file = os.path.join(kwargs['data_dir'],
                                            f"cached_{mode}_{kwargs['model_name']}_{kwargs['max_seq_length']}_"
                                            f"{kwargs['task_name']}")

        if os.path.exists(cached_features_file) and not kwargs['reprocess_input_data']:
            logging.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)

        else:
            logging.info("Creating features from dataset file at %s", kwargs['data_dir'])
            label_list = self.processor.get_labels()
            examples = self.processor.get_test_examples(kwargs['data_dir']) if mode == "test" else \
                self.processor.get_train_examples(kwargs['data_dir'])

            features = self.processor.convert_examples_to_features(
                examples=examples,
                label_list=label_list,
                max_seq_length=kwargs['max_seq_length'],
                tokenizer=self.tokenizer,
                output_mode=output_mode,
                cls_token_at_end=bool(kwargs['model_type'] in ['xlnet']),  # xlnet has a cls token at the end
                cls_token=self.tokenizer.cls_token,
                sep_token=self.tokenizer.sep_token,
                cls_token_segment_id=2 if kwargs['model_type'] in ['xlnet'] else 0,
                pad_on_left=bool(kwargs['model_type'] in ['xlnet']),  # pad on the left for xlnet
                pad_token_segment_id=4 if kwargs['model_type'] in ['xlnet'] else 0
            )

            logging.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        elif output_mode == "multi-label-classification":
            all_label_ids = torch.stack([torch.tensor(f.label_ids) for f in features])
        else:
            raise ValueError(f"The following mode {output_mode} is not handled by the model. Please choose "
                             f"'classification' or 'multi-label-classification'.")

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return dataset

    def __repr__(self):
        return f"<Dataset(evaluate={self.evaluate})>"
