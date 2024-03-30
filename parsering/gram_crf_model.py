# -*- coding: utf-8 -*-
# @ModuleName: crf_2_model
# @Function:
# @Author: Wxb
# @Time: 2024/2/25 15:21
import torch
import torch.nn as nn
from parsering.modules import MLP, BertEmbedding, BiLSTM, Biaffine, CRF
from parsering.modules.dropout import IndependentDropout, SharedDropout
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence)


class bigram_bert_model(nn.Module):
    def __init__(self, args):
        super(bigram_bert_model, self).__init__()

        self.args = args
        self.pretrained = False
        # the embedding layer
        self.char_embed = nn.Embedding(num_embeddings=args.n_chars,
                                       embedding_dim=args.n_feat_embed)
        n_lstm_input = args.n_feat_embed  # 100
        # self.bigram_embed = nn.Embedding(num_embeddings=args.n_bigrams,
        #                                  embedding_dim=args.n_feat_embed)
        # n_lstm_input += args.n_feat_embed  # 100
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
        self.mlp = MLP(n_in=args.n_lstm_hidden * 2,
                       n_out=args.n_labels)

        self.crf = CRF(n_labels=args.n_labels)

        # the MLP layers
        self.mlp2 = MLP(n_in=args.n_lstm_hidden * 2,
                        n_out=args.n_stop_labels)

        self.crf2 = CRF(n_labels=args.n_stop_labels)

        self.pad_index = args.pad_index
        self.unk_index = args.unk_index

    def forward(self, feed_dict, target1=None, target2=None, do_predict=False):
        chars = feed_dict["chars"]
        # get outputs from embedding layers
        char_embed = self.char_embed(chars)
        # bigram = feed_dict["bigram"]
        # bigram_embed = self.bigram_embed(bigram)

        batch_size, seq_len = feed_dict['bert'][0].shape

        mask = feed_dict['bert'][1]
        lens = mask.sum(dim=1).cpu()
        feat_embed = self.feat_embed(*feed_dict['bert'])
        # char_embed, bigram_embed, feat_embed = self.embed_dropout(char_embed, bigram_embed, feat_embed)
        # feat_embed = torch.cat((char_embed, bigram_embed, feat_embed), dim=-1)

        char_embed, feat_embed = self.embed_dropout(char_embed, feat_embed)
        feat_embed = torch.cat((char_embed, feat_embed), dim=-1)
        # feat_embed: (B, L, D)

        x = pack_padded_sequence(feat_embed, lens, True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)

        x1 = self.mlp(x)
        mask = feed_dict['crf_mask']
        non_stop = self.forward_crf(x1, mask, target1, do_predict, ind=0)

        x2 = self.mlp2(x)
        stop = self.forward_crf(x2, mask, target2, do_predict, ind=1)

        return stop, non_stop

    def forward_crf(self, x, mask, target=None, do_predict=False, ind=0):
        mask = mask[:, 1: -1]
        x = x[:, 1: -1]

        # print('mask', mask.shape)
        # print('x', x.shape)
        # print('target', target.shape)

        ret = {}
        if target is not None:
            if not ind:
                loss = self.crf(x, target, mask)
            else:
                loss = self.crf2(x, target, mask)
            ret['loss'] = loss
        if do_predict:
            if not ind:
                predict_labels = self.crf.viterbi(x, mask)
            else:
                predict_labels = self.crf2.viterbi(x, mask)
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
