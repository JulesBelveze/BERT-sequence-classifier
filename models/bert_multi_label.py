import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers import BertModel, BertPreTrainedModel


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """
    Bert model adapted for multi-label sequence classification.
    Note that for imbalance problems will also provide an extra parameter to add inside
    the loss function to integrate the classes distribution.
    """

    def __init__(self, config):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.pos_weight = config["pos_weight"] if config["use_pos_weight"] else None

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, labels=None):
        """
        :param input_ids:
        :param attention_mask:
        :param token_type_ids:
        :param position_ids:
        :param head_mask:
        :param labels:
        :return:
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss, ) + outputs

        return outputs

    def freeze_bert_encoder(self):
        """Freeze BERT layers"""
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        """Unfreeze BERT layers"""
        for param in self.bert.parameters():
            param.requires_grad = True
