# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, logging
logging.set_verbosity_error()  # 忽略 bert 警告


class BertEmbedding(nn.Module):

    def __init__(self, model, n_layers, n_out, requires_grad=False):
        super(BertEmbedding, self).__init__()

        self.config = AutoConfig.from_pretrained(model)
        self.bert = AutoModel.from_pretrained(model,
                                              config=self.config)
        # self.bert = self.bert.requires_grad_(requires_grad)
        self.n_layers = n_layers
        self.n_out = n_out
        self.requires_grad = requires_grad
        self.hidden_size = self.bert.config.hidden_size

        if self.hidden_size != n_out:
            self.projection = nn.Linear(self.hidden_size, n_out, False)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"n_layers={self.n_layers}, n_out={self.n_out}"
        if self.requires_grad:
            s += f", requires_grad={self.requires_grad}"
        s += ')'

        return s

    # def forward(self, feed_dict):
    #     if not self.requires_grad:
    #         self.bert.eval()
    #     embed = self.bert(**feed_dict)
    #     embed = embed.last_hidden_state
    #
    #     if hasattr(self, 'projection'):
    #         embed = self.projection(embed)
    #
    #     return embed

    def forward(self, subwords, bert_mask):
        if not self.requires_grad:
            self.bert.eval()
        embed = self.bert(subwords, attention_mask=bert_mask).last_hidden_state

        if hasattr(self, 'projection'):
            embed = self.projection(embed)

        return embed
