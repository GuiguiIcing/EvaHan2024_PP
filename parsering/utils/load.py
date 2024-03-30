# -*- coding: utf-8 -*-
# @Time    : 2024/1/17 15:34

import sys
import random
import json
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from collections import Counter

from ..utils.common import punctuation, tag_before, tag_after
from ..utils.common import pad, bos, eos


def check_non_stop(line):
    temp = []
    try:
        for each in line:
            if each in tag_before:
                temp.append(each)
            elif each in tag_after:
                temp.pop()
    except:
        return False
    if temp:
        return False
    return True


class Load:
    def __init__(self, args):
        self.args = args

        self.stop = {"。", '，', '？', '！', '、', '：', '；'}
        self.non_stop = {'“': 'Q_SY', '”': 'H_SY', '‘': 'Q_DY',
                         '’': 'H_DY', '《': 'Q_S', '》': 'H_S'}

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_check)
        self.tokenizer.pad_token = pad
        self.tokenizer.bos_token = bos
        self.tokenizer.eos_token = eos

        self.labels = set()
        self.stop_labels = set()
        self.chars = set()
        self.count = Counter()

        self.data_ls = []
        self.llm_train_ls = []
        self.llm_dev_ls = []
        # self.read_file()
        self.read_file_twice()

        self.labels_dic = {key: index for index, key in enumerate(sorted(self.labels))}
        self.id2labels = {index: key for index, key in enumerate(sorted(self.labels))}
        self.stop_labels_dic = {key: index for index, key in enumerate(sorted(self.stop_labels))}
        self.id2stop_labels = {index: key for index, key in enumerate(sorted(self.stop_labels))}

        print(self.labels_dic)
        print(self.stop_labels_dic)

        self.chars2ids = {}
        self.id2chars = {}
        self.bichars2ids = {}

        # self.save()
        length = len(self.data_ls)
        print('original length + llm:', length)
        train_len = int(0.95 * length)
        llm_train_len, llm_dev_len = len(self.llm_train_ls), len(self.llm_dev_ls)

        self.data_ls = self.data_ls + self.llm_train_ls + self.llm_dev_ls
        self.punc_to_ids()
        self.llm_train_ls, self.llm_dev_ls = (self.data_ls[length: length + llm_train_len],
                                              self.data_ls[length + llm_train_len:])
        self.data_ls = self.data_ls[: length]

        args.update({
            'pad_index': self.tokenizer.pad_token_id,
            'unk_index': self.tokenizer.unk_token_id,
            'bos_index': self.tokenizer.bos_token_id,
            'eos_index': self.tokenizer.eos_token_id,
            "n_labels": len(self.labels_dic) + 1,
            "n_stop_labels": len(self.stop_labels_dic) + 1,
            "n_chars": len(self.chars2ids),
            "n_bigrams": len(self.bichars2ids),
        })
        # print(self.stop_labels_dic)
        self.pad_id = args.n_labels - 1
        self.pad_id_stop = args.n_stop_labels - 1

        random.shuffle(self.data_ls)
        self.train = self.data_ls[: train_len] + self.llm_train_ls
        self.validation = self.data_ls[train_len:] + self.llm_dev_ls
        print(f'train data length = {len(self.train)}, dev length = {len(self.validation)}')

    def read_file(self):  # 数据清洗
        f = open(self.args.data, mode='r', encoding='utf-8')
        lines = [line.strip() for line in f]
        f.close()

        new_lines = []
        # 1.st clean the data (把出现在开头的不合法符号放到上一行)
        for i, line in enumerate(lines):
            while line and line[0] in punctuation and line[0] not in tag_before:
                new_lines[-1] += line[0]
                line = line[1:]
            if line:
                for each in punctuation:
                    if each in line:  # 检查是否有标点在句子里
                        new_lines.append(line)
                        break

        # 2.st process the data
        self.data_ls = [self.signal_tag(list(line))
                        for line in new_lines]

    def read_file_twice(self):
        f = open(self.args.data, mode='r', encoding='utf-8')
        lines = [line.strip() for line in f]
        f.close()

        new_lines = []
        no_punc_lines = []
        for i, line in enumerate(lines):
            while line and line[0] in punctuation and line[0] not in tag_before:
                new_lines[-1] += line[0]
                line = line[1:]
            if line:
                for each in punctuation:
                    if each in line:  # 检查是否有标点在句子里
                        if check_non_stop(line):
                            line = line.replace("‘‘", "‘")
                            line = line.replace("’’", "’")
                            line = line.replace("““", "“")
                            line = line.replace("””", "”")
                            line = line.replace("《《", "《")
                            line = line.replace("》》", "》")
                            line = line.replace("：：", "：")
                            new_lines.append(line)
                        break
                    else:
                        no_punc_lines.append(line)
                        break
        print('no_punc_lines', len(no_punc_lines))
        print('punc_lines', len(new_lines))

        f = open('dataset/train.json', mode='r')
        self.llm_train_ls = [self.double_tag(list(line))
                             for line in json.load(f)]
        f.close()
        f = open('dataset/dev.json', mode='r')
        self.llm_dev_ls = [self.double_tag(list(line))
                           for line in json.load(f)]
        f.close()
        # 2.nd process the data
        self.data_ls = [self.double_tag(list(line))
                        for line in new_lines + no_punc_lines[: int(0.5 * len(new_lines))]]

        # 3.rd get the count of tags
        [self.signal_tag(list(line)) for line in new_lines]
        print(self.count)

    @staticmethod
    def sliding_window(line, window=100, max_len=510):
        if (t := len(line)) <= max_len:
            return [line]
        res, pre, now = [], 0, 0
        while now < t:
            now = pre + max_len
            res.append(line[pre: now])
            pre = now - window
        return res

    def punc_to_ids(self):
        bichars = set()
        for words, _, _ in self.data_ls:
            bichars.update(set(["".join(words[i: i + 2]) for i in range(len(words) - 1)]))

        self.bichars2ids = {key: index for index, key in enumerate(sorted(bichars))}
        bos_index = len(bichars)
        eos_index, pad_index = bos_index + 1, bos_index + 2
        self.bichars2ids[bos] = bos_index
        self.bichars2ids[eos] = eos_index
        self.bichars2ids[pad] = pad_index

        self.chars2ids = {key: index for index, key in enumerate(sorted(self.chars))}
        bos_index = len(self.chars)
        eos_index, pad_index = bos_index + 1, bos_index + 2
        self.chars2ids[bos], self.chars2ids[eos], self.chars2ids[pad] = \
            bos_index, eos_index, pad_index
        self.id2chars = {self.chars2ids[key]: key for key in self.chars2ids}

        new_dataset = []
        for words, tags, segs in self.data_ls:
            bert_input = self.tokenizer.encode("".join(words),
                                               truncation=True,
                                               max_length=512)
            bert_input = torch.tensor(bert_input)
            length = len(bert_input)

            words_index = torch.tensor([self.chars2ids[e] for e in [bos] + words[:length - 2] + [eos]])
            bi_words = [bos] + ["".join(words[i: i + 2]) for i in range(length - 3)] + [eos, pad]
            bi_words_index = torch.tensor([self.bichars2ids[e] for e in bi_words])

            attention_mask = torch.ones(length).gt(0)
            mask = torch.tensor([0] + [1] * (length - 2) + [0])
            words = (words_index, bi_words_index, bert_input, attention_mask, mask)

            tags = torch.tensor([self.labels_dic[e] for e in tags[:length - 2]])
            # segs = [(i, j, self.stop_labels_dic[e]) for i, j, e in segs]
            segs = torch.tensor([self.stop_labels_dic[e] for e in segs[: length - 2]])  # todo:
            line = (*words, tags, segs)

            new_dataset.append(line)
        self.data_ls = new_dataset

    def save(self):
        f = open('siku.train.txt', mode='w', encoding='utf-8')
        a = int(len(self.data_ls) * 0.9)
        for words, non_stop, tags in self.data_ls[: a]:
            if len(words) <= 510:
                for w, n, t in zip(words, non_stop, tags):
                    f.write(w + '\t' + n + '\t' + t + '\n')
                f.write('\n')
        f.close()

        f = open('siku.dev.txt', mode='w', encoding='utf-8')
        for words, non_stop, tags in self.data_ls[a:]:
            if len(words) <= 510:
                for w, n, t in zip(words, non_stop, tags):  # tags 不是三元组
                    f.write(w + '\t' + n + '\t' + t + '\n')
                f.write('\n')
        f.close()

        print('Save successfully.')

        # x, n = 0, 0
        # for each in self.data_ls:
        #     if len(each[0]) >= 510:
        #         x += 1
        #         # print(each)
        #     else:
        #         for i, j, punc in each[-1]:   # tags 是三元组去测试最大长度。
        #             n = j - i if n <= j - i else n
        # print(f"{x} / {len(self.data_ls)} = {x / len(self.data_ls)}, max=", n)  # 2%, 501
        import sys
        sys.exit(0)

    def signal_tag(self, line):
        """
        只标注一个标点 tag，非标点的字用 O 标注
        :param line:
        :return:
        """
        chars, tags = [], []
        before = ""

        for i, char in enumerate(line):
            if p := punctuation.get(char):
                if char in tag_before:
                    before = char if not before else before + char
                else:
                    if before:
                        continue
                    if tags:
                        tags[-1] = char if tags[-1] == "O" else tags[-1] + char
            else:
                chars.append(char)
                temp = "O" if not before else before
                tags.append(temp)
                before = ""

        assert len(chars) == (t := len(tags)), f"{len(chars)}, {len(tags)}"

        # 处理标签的合法性
        for i in range(t):
            if not self.judge(tags[i]):
                tags[i] = tags[i][0]
            tags[i] = tags[i].replace("‘‘", "‘")
            tags[i] = tags[i].replace("’’", "’")
            tags[i] = tags[i].replace("““", "“")
            tags[i] = tags[i].replace("””", "”")
            tags[i] = tags[i].replace("《《", "《")
            tags[i] = tags[i].replace("》》", "》")
            tags[i] = tags[i].replace("：：", "：")

        # self.labels |= set(tags)
        # self.chars |= set(chars)
        self.count.update(tags)
        return chars, tags

    def tag2seg(self, tags):
        start = 0
        end = 0
        segs = []

        for tag in tags:
            end += 1
            if tag in self.stop:
                segs.append((start, end, tag))
                start = end

        if start != end:
            segs.append((start, end, tags[-1]))

        # print(len(tags), tags)
        # print(segs)
        # import sys
        # sys.exit(0)
        return segs

    def double_tag(self, line):
        """
        分别标注《》 “ ” ‘’ 与 ，。！？、
        :param line:
        :return:
        """
        chars, tags, tags_stop = [], [], []
        before = ""

        for i, char in enumerate(line):
            if p := punctuation.get(char):
                if char in tag_before:
                    before = char if not before else before + char
                else:
                    if before or char in self.stop:  # 内容为空
                        continue
                    if tags:
                        tags[-1] = char if tags[-1] == "O" else tags[-1] + char
            else:
                chars.append(char)
                temp = "O" if not before else before
                tags.append(temp)
                before = ""

        for char in line:
            if p := punctuation.get(char):
                if char in self.stop and tags_stop:
                    tags_stop[-1] = char if tags_stop[-1] == "O" else tags_stop[-1]
            else:
                tags_stop.append("O")

        assert len(chars) == len(tags) == len(tags_stop), \
            f"{len(chars)}, {len(tags)}, {len(tags_stop)}"

        self.labels |= set(tags)
        self.stop_labels |= set(tags_stop)
        self.chars |= set(chars)
        # return chars, tags, self.tag2seg(tags_stop)  # todo
        return chars, tags, tags_stop

    def judge(self, s):
        c = Counter(s)
        n = 0
        for kk in c:
            if kk in self.stop:
                n += c[kk]
        if "？！" in s and n > 2:
            return False
        elif n > 1:
            return False
        else:
            return True

    def collate_fn(self, batch):
        tokens, labels = zip(*batch)
        labels = [torch.tensor(each) for each in labels]
        batch_input = self.tokenizer(tokens,
                                     is_split_into_words=True,
                                     padding=True,
                                     return_tensors='pt',
                                     truncation=True,
                                     max_length=512
                                     )

        batch_labels = pad_sequence(labels, batch_first=True, padding_value=self.pad_id)
        _, seq_len = batch_input['input_ids'].shape

        lens = batch_input.attention_mask.sum(dim=-1) - 2
        max_length = max(lens) + 1
        mask = np.array([np.pad(np.ones(length), (1, max_length - length), 'constant', constant_values=0)
                         for length in lens])
        mask = torch.tensor(mask, dtype=torch.long)
        assert batch_input.input_ids.shape == mask.shape, f"{batch_input.input_ids.shape} {mask.shape}"
        return batch_input, batch_labels[:, :seq_len - 2], mask

    def transform(self, sequences):
        spans, spans_punc = [], []
        max_len = max([sequence[-1][1] + 1 for sequence in sequences])
        # print(max_len)
        max_len = min(max_len, 510 + 1)
        # print(sequences[0])
        for sequence in sequences:
            span_chart = torch.full((max_len, max_len), 0)
            span_chart_ = torch.full((max_len, max_len), 0)
            for i, j, pos in sequence:
                if j < max_len:
                    span_chart[i, j] = 1
                    span_chart_[i, j] = pos
            spans_punc.append(span_chart_)
            spans.append(span_chart)

        return torch.stack(spans), torch.stack(spans_punc)

    @staticmethod
    def pad(tensors, padding_value=0):
        size = [len(tensors)] + [max(tensor.size(i) for tensor in tensors)
                                 for i in range(len(tensors[0].size()))]
        out_tensor = tensors[0].data.new(*size).fill_(padding_value)
        for i, tensor in enumerate(tensors):
            out_tensor[i][[slice(0, i) for i in tensor.size()]] = tensor
        return out_tensor

    def collate_fn_(self, batch):
        tokens, labels, segs = zip(*batch)
        labels = [torch.tensor(each) for each in labels]
        batch_input = self.tokenizer(tokens,
                                     is_split_into_words=True,
                                     padding=True,
                                     return_tensors='pt',
                                     truncation=True,
                                     max_length=512
                                     )

        batch_labels = pad_sequence(labels, batch_first=True, padding_value=self.pad_id)
        _, seq_len = batch_input['input_ids'].shape

        lens = batch_input.attention_mask.sum(dim=-1) - 2
        max_length = max(lens) + 1
        mask = np.array([np.pad(np.ones(length), (1, max_length - length), 'constant', constant_values=0)
                         for length in lens])
        mask = torch.tensor(mask, dtype=torch.long)
        # print(lens, mask.sum(1))
        assert mask.sum(1).tolist() == lens.tolist()
        assert batch_input.input_ids.shape == mask.shape, f"{batch_input.input_ids.shape} {mask.shape}"
        segs, segs_punc = self.transform(segs)

        return batch_input, batch_labels[:, :seq_len - 2], mask, segs, segs_punc

    def collate_fn_bigram(self, batch):
        tokens, bi_chars, bert_input, attention_mask, mask, non_stop_tags, stop_tags = zip(*batch)
        tokens = self.pad(tokens, padding_value=self.chars2ids[pad])
        bi_chars = self.pad(bi_chars, padding_value=self.bichars2ids[pad])
        bert_input = self.pad(bert_input, padding_value=self.tokenizer.pad_token_id)
        mask = self.pad(mask, padding_value=0).bool()
        attention_mask = self.pad(attention_mask, padding_value=0).bool()

        non_stop_tags = self.pad(non_stop_tags, padding_value=self.pad_id)
        stop_tags = self.pad(stop_tags, padding_value=self.pad_id_stop)

        assert mask.shape == bert_input.shape == tokens.shape == bi_chars.shape, f"{mask.shape}, {bert_input.shape}, {tokens.shape}, {bi_chars.shape}"
        # return ((tokens.to(self.args.device), bi_chars.to(self.args.device), bert_input.to(self.args.device),
        #          attention_mask.to(self.args.device), mask.to(self.args.device)),
        #         non_stop_tags.to(self.args.device), stop_tags.to(self.args.device))
        return ((tokens.to(self.args.device), bi_chars, bert_input.to(self.args.device),
                 attention_mask.to(self.args.device), mask.to(self.args.device)),
                non_stop_tags.to(self.args.device), stop_tags.to(self.args.device))

    def collate_fn_crf2(self, batch):
        tokens, labels, stop = zip(*batch)
        labels = [torch.tensor(each) for each in labels]
        stop = [torch.tensor(each) for each in stop]
        batch_input = self.tokenizer(tokens,
                                     is_split_into_words=True,
                                     padding=True,
                                     return_tensors='pt',
                                     truncation=True,
                                     max_length=512
                                     )

        batch_labels = pad_sequence(labels, batch_first=True, padding_value=self.pad_id)
        stop = pad_sequence(stop, batch_first=True, padding_value=self.pad_id_stop)

        _, seq_len = batch_input['input_ids'].shape

        lens = batch_input.attention_mask.sum(dim=-1) - 2
        max_length = max(lens) + 1
        mask = np.array([np.pad(np.ones(length), (1, max_length - length), 'constant', constant_values=0)
                         for length in lens])
        mask = torch.tensor(mask, dtype=torch.long)
        assert batch_input.input_ids.shape == mask.shape, f"{batch_input.input_ids.shape} {mask.shape}"
        return batch_input, batch_labels[:, :seq_len - 2], stop[:, :seq_len - 2], mask


