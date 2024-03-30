# -*- coding: utf-8 -*-
# @Time    : 2024/1/22 17:27
# @Author  : wxb
# @File    : predict_single.sh.py

import os
from datetime import datetime

from .cmd_single import CMD
from ..utils.load_pred_single import Load_pred

from torch.utils.data import DataLoader


class Predict_single(CMD):
    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Evaluate a trained model.'
        )
        subparser.add_argument('--data', default='data/ptb_pos',
                               help='path to train file')
        subparser.add_argument('--pred_data', default='../dataset/EvaHan2024_testset.txt',
                               help='path to train file')
        subparser.add_argument('--pred_path', default='TestPredict/result.txt',
                               help='path to save the predict result.')
        return subparser

    def __call__(self, args):
        super(Predict_single, self).__call__(args)
        print('Load the dataset.')
        start = datetime.now()

        loader = Load_pred(args)
        sliding_ids = loader.sliding_ids
        enters = loader.enters
        print('enter:', enters)
        collate_fn = loader.collate_fn_bigram_pred

        test = DataLoader(loader.test, batch_size=args.batch_size,
                          shuffle=False, collate_fn=collate_fn)

        print('Load the model.')
        self.model = self.model_cl.load(args.save_model)
        print(self.model)

        chars_preds, lens = self.predict(test)
        tokens_punc = []
        for i in range(len(lens)):
            char, punc = chars_preds[i]
            tokens = loader.back_2_sentence_last(char, punc, lens[i])

            tokens_punc.append(tokens)
        
        for each in sliding_ids[::-1]:
            tokens_punc[each[0]] = loader.merge(tokens_punc[each[0]: each[1]])
            tokens_punc[each[0]+1:] = tokens_punc[each[1]:]

        with open(args.pred_path, mode='w', encoding='utf-8') as f:
            length = len(tokens_punc)
            for i in range(length):
                f.write("".join(tokens_punc[i]))
                if i != length - 1:
                    f.write('\n')
                if temp := enters.get(i):
                    for _ in range(temp):
                        f.write('\n')

            if 'zz' in args.pred_path:
                f.write('\n')

        print(f'{datetime.now() - start}s elapsed.')
        print(f'Predict result save in {args.pred_path}')
        print('Finish.')
