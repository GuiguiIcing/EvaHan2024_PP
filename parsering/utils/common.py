# -*- coding: utf-8 -*-

pad = '<pad>'
unk = '<unk>'
bos = '<bos>'
eos = '<eos>'
nul = '<nul>'

# 十个标点符号的拼音
punc = ["D", "J", "Dun", "M", "F", "W", "G", "SY", "DY", "S"]

punctuation_ = {0: 'D', 1: 'J', 2: 'Dun', 3: 'M',
                4: 'F', 5: 'W', 6: 'G', 7: 'SY',
                8: 'DY', 9: 'S'}

punctuation = {'，': 'D',
               '。': 'J',
               '、': 'Dun',
               '：': 'M',
               '；': 'F',
               '？': 'W',
               '!': 'G',
               '“': 'Q_SY',
               '”': 'H_SY',
               '‘': 'Q_DY',
               '’': 'H_DY',
               '《': 'Q_S',
               '》': 'H_S'}

tag_before = {"《", "‘", "“"}
tag_after = {"》", "’", "”"}
