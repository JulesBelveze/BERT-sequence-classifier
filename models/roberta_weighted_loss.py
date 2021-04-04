import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification
from transformers.models.xlm_roberta import XLMRobertaConfig
from utils import LabelSmoothingLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RobertaWithWeightedLoss(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.loss_fct = config.loss_fct
        self.class_weights = torch.tensor(config.class_weights).to(device) if config.class_weights else None

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None):
        """"""
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = {
                None: CrossEntropyLoss(),
                "lsl": LabelSmoothingLoss(classes=self.num_labels, smoothing=.2),
                "distrib": CrossEntropyLoss(weight=self.class_weights.to(device) if self.class_weights else None),
                "batch": CrossEntropyLoss(weight=self.get_weights(labels, self.num_labels).to(device))
            }[self.loss_fct]
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

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


class XLMRobertaWithWeightedLoss(RobertaWithWeightedLoss):
    config_class = XLMRobertaConfig
