import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers import DistilBertModel, DistilBertPreTrainedModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DistilBertForMultiLabelSequenceClassification(DistilBertPreTrainedModel):
    """
    DistilBert model adapted for multi-label sequence classification.
    Note that for imbalance problems will also provide an extra parameter to add inside
    the loss function to integrate the classes distribution.
    """

    def __init__(self, config):
        super(DistilBertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.pos_weight = torch.Tensor(config.pos_weight).to(device) if config.use_pos_weight else None

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels=None):
        """
        :param input_ids: sentence or sentences represented as tokens
        :param attention_mask: tells the model which tokens in the input_ids are words and which are padding.
                               1 indicates a token and 0 indicates padding.
        :param head_mask: mask to nullify selected heads of the self-attention modules
        :param inputs_embeds: Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an
                              embedded representation. This is useful if you want more control over how to convert
                              :obj:`input_ids` indices into associated vectors than the model's internal embedding
                              lookup matrix.
        :param labels: target for each input
        :return:
        """
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)

        outputs = (logits,) + distilbert_output[1:]
        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    def freeze_bert_encoder(self):
        """Freeze DistilBERT layers"""
        for param in self.distilbert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        """Unfreeze DistilBERT layers"""
        for param in self.distilbert.parameters():
            param.requires_grad = True
