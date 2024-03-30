# -*- coding: utf-8 -*-
# @Time    : 2024/2/27 20:07
# @Author  : wxb
# @File    : cmd_bigram.py

import os
import sys
from typing import Any
from copy import deepcopy

from ..BasePlusModel import roberta_bilstm_crf

from ..utils.metric import PosMetric

import torch
import torch.nn as nn


class CMD(object):

    def __call__(self, args) -> Any:
        self.args = args
        if not os.path.exists(args.file):
            os.mkdir(args.file)

        self.model_check = args.base_model

        self.model_cl = roberta_bilstm_crf

        args.update({
            'model_check': self.model_check,
            'model_cl': self.model_cl,
        })

        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)

    def train(self, loader):
        self.model.train()
        torch.set_grad_enabled(True)
        for data in loader:
            ((chars, bi_chars, bert_input, attention_mask, mask), tags) = data
            self.optimizer.zero_grad()

            feed_dict = {'chars': chars,
                         'bert': [bert_input, attention_mask],
                         'crf_mask': mask}

            ret = self.model(feed_dict, tags)
            loss = ret['loss']
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.args.clip)

            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, loader):
        print('evaluate...')
        self.model.eval()
        total_loss, metric_pos = 0, PosMetric()
        total_re, total_num = 0, 0

        for data in loader:
            ((chars, bi_chars, bert_input, attention_mask, mask), tags) = data
            self.optimizer.zero_grad()

            feed_dict = {'chars': chars,
                         'bert': [bert_input, attention_mask],
                         'crf_mask': mask}
            # do_predict=True
            ret = self.model(feed_dict, tags, do_predict=True)
            loss = ret['loss']

            total_loss += loss.item()

            pred = ret['predict']
            # 评估指标计算
            metric_pos(pred, tags, mask.sum(dim=-1))

            total_num += mask.sum()

        total_loss /= len(loader)

        return total_loss, metric_pos

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()

        chars_preds = []
        lens = []
        total_re, total_num = 0, 0
        for data in loader:
            chars, bi_chars, bert_input, attention_mask, mask, str_chars = data

            feed_dict = {'chars': chars,
                         'bert': [bert_input, attention_mask],
                         'crf_mask': mask}

            ret = self.model(feed_dict, do_predict=True)
            for char_line, punc in zip(str_chars, ret['predict']):
                chars_preds.append((char_line, punc))

            lens.append(mask.sum(dim=-1))
            total_num += mask.sum()
        print("Numbers of total chars", total_num)
        return chars_preds, torch.cat(lens)

