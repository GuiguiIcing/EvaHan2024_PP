# -*- coding: utf-8 -*-
# @Time    : 2024/1/22 20:29
# @Author  : wxb
# @File    : __init__.py


from .train_gram import Train
from .train_single import Train_single

from .predict_gram import Predict
from .predict_single import Predict_single

__all__ = ['Train', 'Train_single',
           'Predict', 'Predict_single']
