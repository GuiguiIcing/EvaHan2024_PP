# -*- coding: utf-8 -*-
# @Time    : 2024/1/18 18:16

import torch
import torch.nn as nn
from parsering.modules import BertEmbedding, BiLSTM, MLP, CRF
from parsering.modules.dropout import IndependentDropout, SharedDropout

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class roberta_bilstm_crf(nn.Module):
    def __init__(self, args):
        super(roberta_bilstm_crf, self).__init__()

        self.args = args
        self.pretrained = False
        # the embedding layer
        self.char_embed = nn.Embedding(num_embeddings=args.n_chars,
                                       embedding_dim=args.n_embed)
        n_lstm_input = args.n_embed  # 100

        self.feat_embed = BertEmbedding(model=args.base_model,
                                        n_layers=args.n_bert_layers,
                                        n_out=args.n_feat_embed)
        n_lstm_input += args.n_feat_embed

        self.embed_dropout = IndependentDropout(p=args.embed_dropout)

        # the lstm layer
        self.lstm = BiLSTM(input_size=n_lstm_input,
                           hidden_size=args.n_lstm_hidden,
                           num_layers=args.n_lstm_layers,
                           dropout=args.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=args.lstm_dropout)

        # the MLP layers
        self.mlp = MLP(n_in=args.n_lstm_hidden*2,
                       n_out=args.n_labels)

        self.crf = CRF(n_labels=args.n_labels)

        self.pad_index = args.pad_index
        self.unk_index = args.unk_index

    def forward(self, feed_dict, target=None, do_predict=False):
        chars = feed_dict["chars"]
        char_embed = self.char_embed(chars)

        batch_size, seq_len = feed_dict['bert'][0].shape

        # get outputs from embedding layers
        # char_embed = self.char_embed(ext_chars)
        mask = feed_dict['bert'][1]
        lens = mask.sum(dim=1).cpu()

        # feats = feed_dict
        feat_embed = self.feat_embed(*feed_dict['bert'])

        char_embed, feat_embed = self.embed_dropout(char_embed, feat_embed)
        feat_embed = torch.cat((char_embed, feat_embed), dim=-1)
        # embed: (B, L, D)

        x = pack_padded_sequence(feat_embed, lens, True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)

        x = self.mlp(x)
        mask = feed_dict['crf_mask']
        mask = mask[:, 1: -1]
        x = x[:, 1: -1]

        ret = {}
        if target is not None:
            loss = self.crf(x, target, mask)
            ret['loss'] = loss
        if do_predict:
            predict_labels = self.crf.viterbi(x, mask)
            ret['predict'] = predict_labels

        return ret

    @classmethod
    def load(cls, path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(path, map_location=device)
        model = cls(state['args'])
        # model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(device)

        return model

    def save(self, path):
        state_dict, pretrained = self.state_dict(), None
        if self.pretrained:
            pretrained = {'embed': state_dict.pop('char_pretrained.weight')}
            if hasattr(self, 'bi_pretrained'):
                pretrained.update(
                    {'bi_embed': state_dict.pop('bi_pretrained.weight')})
            if hasattr(self, 'tri_pretrained'):
                pretrained.update(
                    {'tri_embed': state_dict.pop('tri_pretrained.weight')})
        state = {
            'args': self.args,
            'state_dict': state_dict,
            'pretrained': pretrained
        }
        torch.save(state, path)


