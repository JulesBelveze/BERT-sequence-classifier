import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import DistilBertForSequenceClassification
from utils import LabelSmoothingLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DistilBertWithWeightedLoss(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.loss_fct = config.loss_fct
        self.class_weights = torch.tensor(config.class_weights).to(device) if config.class_weights else None

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels=None):
        """"""
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        if labels is not None:
            loss_fct = {
                None: CrossEntropyLoss(),
                "lsl": LabelSmoothingLoss(classes=self.num_labels, smoothing=.2),
                "distrib": CrossEntropyLoss(weight=self.class_weights.to(device) if self.class_weights else None),
                "batch": CrossEntropyLoss(weight=self.get_weights(labels, self.num_labels).to(device))
            }[self.loss_fct]
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + distilbert_output[1:]
        return ((loss,) + output) if loss is not None else output

    @staticmethod
    def get_weights(target, n_labels):
        """Get class weights per batch"""
        weights = n_labels * [1]
        count_labels, count = torch.unique(target, return_counts=True)
        count_labels = count_labels.cpu().data.numpy().tolist()
        count = count.cpu().data.numpy().tolist()
        labels_size = target.size(0)
        for i, label in enumerate(count_labels):
            weights[label] = labels_size / (len(count) * count[i])
        return torch.from_numpy(np.array(weights)).float()
