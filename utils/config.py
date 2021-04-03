import torch
from transformers import (BertConfig, BertTokenizer,
                          DistilBertConfig, DistilBertTokenizer,
                          RobertaConfig, RobertaTokenizer,
                          XLMRobertaConfig, XLMRobertaTokenizer)

from models import BertForMultiLabelSequenceClassification, BertWithWeightedLoss, \
    DistilBertForMultiLabelSequenceClassification, DistilBertWithWeightedLoss, \
    RobertaForMultiLabelSequenceClassification, RobertaWithWeightedLoss, \
    XLMRobertaForMultiLabelSequenceClassification, XLMRobertaWithWeightedLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_CLASSES = {
    'bert-multi-label': (BertConfig, BertForMultiLabelSequenceClassification, BertTokenizer),
    'distilbert-multi-label': (DistilBertConfig, DistilBertForMultiLabelSequenceClassification, DistilBertTokenizer),
    'roberta-multi-label': (RobertaConfig, RobertaForMultiLabelSequenceClassification, RobertaTokenizer),
    'xlm-roberta-multi-label': (XLMRobertaConfig, XLMRobertaForMultiLabelSequenceClassification, XLMRobertaTokenizer),
    'bert-weighted': (BertConfig, BertWithWeightedLoss, BertTokenizer),
    'distilbert-weighted': (DistilBertConfig, DistilBertWithWeightedLoss, DistilBertTokenizer),
    'roberta-weighted': (RobertaConfig, RobertaWithWeightedLoss, RobertaTokenizer),
    'xlm-roberta-weighted': (XLMRobertaConfig, XLMRobertaWithWeightedLoss, XLMRobertaTokenizer)
}


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set(self, key, value):
        self[key] = value
        setattr(self, key, value)

    def update(self, other=None, **kwargs):
        if other is not None:
            for key, value in other.items():
                self.set(key, value)

        for key, value in kwargs.items():
            self.set(key, value)

    @property
    def model_config(self):
        try:
            return MODEL_CLASSES[self.model_type][0]
        except KeyError:
            raise KeyError("'model_type' parameter is not defined.")

    @property
    def model_class(self):
        try:
            return MODEL_CLASSES[self.model_type][1]
        except KeyError:
            raise KeyError("'model_type' parameter is not defined.")

    @property
    def tokenizer_class(self):
        try:
            return MODEL_CLASSES[self.model_type][2]
        except KeyError:
            raise KeyError("'model_type' parameter is not defined.")


multi_label_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
multi_class_labels = ["VLDB", "ISCAS", "SIGGRAPH", "INFOCOM", "WWW"]

# check https://huggingface.co/transformers/pretrained_models.html?highlight=pretrained for
# the list of pretrained models.

config = Config(
    data_dir='data/',
    model_type='roberta-multi-label',
    model_name='roberta-base',
    tokenizer_name='roberta-base',
    task_name='multi-label',
    output_dir='outputs/',
    cache_dir='cache/',
    fp16=False,
    fp16_opt_level='O1',
    n_gpu=torch.cuda.device_count(),
    max_seq_length=512,
    hidden_dropout_prob=.2,
    output_mode='multi-label-classification',
    train_batch_size=8,
    eval_batch_size=8,
    num_labels=6,
    labels=multi_label_labels,
    class_weights=[0.57041252, 0.97094431, 1.3147541, 1.57874016, 1.21515152],
    use_class_weights=True,
    pos_weight=[10.433569, 100.044514, 18.886377, 333.830544, 20.257839, 113.573665],
    use_pos_weight=True,
    truncate_mode="head_tail",

    gradient_accumulation_steps=1,
    num_train_epochs=5,
    weight_decay=0,
    learning_rate=1e-5,
    adam_epsilon=1e-8,
    warmup_ratio=0.06,
    warmup_steps=False,
    max_grad_norm=0.25,

    logging_steps=200,
    evaluate_during_training=True,
    save_steps=20000,
    eval_all_checkpoints=True,
    get_mismatched=True,

    model_path=None,

    notes='Using toxicity dataset'
)
