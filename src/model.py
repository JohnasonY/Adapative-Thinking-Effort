"""DistilBERT encoder with a five-output regression head."""

import torch
from torch import nn
from transformers import DistilBertConfig, DistilBertModel

DEFAULT_ENCODER = "distilbert/distilbert-base-uncased"


class DifficultyModel(nn.Module):
    def __init__(self, encoder_name=DEFAULT_ENCODER, config=None):
        super().__init__()
        self.encoder = (DistilBertModel(DistilBertConfig.from_dict(config))
                        if config is not None else DistilBertModel.from_pretrained(encoder_name))
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.encoder.config.dim, 5))

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return 1.0 + 4.0 * torch.sigmoid(self.head(hidden[:, 0]))
