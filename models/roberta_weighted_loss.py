import torch
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification
from transformers.models.xlm_roberta import XLMRobertaConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RobertaWithWeightedLoss(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.class_weights = torch.tensor(config.class_weights).to(device) if config.use_class_weights else None

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
            loss_fct = CrossEntropyLoss(weight=self.class_weights) if self.class_weights is not None \
                else CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class XLMRobertaWithWeightedLoss(RobertaWithWeightedLoss):
    config_class = XLMRobertaConfig
