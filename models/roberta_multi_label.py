import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers import RobertaForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
from transformers.models.xlm_roberta import XLMRobertaConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RobertaForMultiLabelSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config, use_rch=False):
        super(RobertaForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if use_rch else \
            RobertaClassificationHead(config)
        self.pos_weight = torch.Tensor(config.pos_weight).to(device) if config.use_pos_weight else None

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None):
        """
        :param input_ids: sentence or sentences represented as tokens
        :param attention_mask: tells the model which tokens in the input_ids are words and which are padding.
                               1 indicates a token and 0 indicates padding.
        :param token_type_ids: used when there are two sentences that need to be part of the input. It indicate which
                               tokens are part of sentence1 and which are part of sentence2.
        :param position_ids: indices of positions of each input sequence tokens in the position embeddings. Selected
                             in the range ``[0, config.max_position_embeddings - 1]
        :param head_mask: mask to nullify selected heads of the self-attention modules
        :param labels: target for each input
        :param output_attentions: whether or not to return the attentions tensors of all attention layers
        :param output_hidden_states: whether or not to return the hidden states of all layers.
        :return:
        """
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    def freeze_bert_encoder(self):
        """Freeze BERT layers"""
        for param in self.roberta.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        """Unfreeze BERT layers"""
        for param in self.roberta.parameters():
            param.requires_grad = True


class XLMRobertaForMultiLabelSequenceClassification(RobertaForMultiLabelSequenceClassification):
    config_class = XLMRobertaConfig
