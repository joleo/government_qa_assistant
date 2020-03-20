# -*- coding: utf-8 -*-
"""
@Time ： 2020/3/10 20:59
@Auth ： joleo
@File ：bert_model.py
"""
from tqdm import tqdm
from keras.layers import *
from keras.models import Model
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from rank.config import *
#######################################载入预训练结果#######################################
if flag == 1:
    # base bert
    config_path = './chinese_wwm_ex_bert/bert_config.json'
    checkpoint_path = './chinese_wwm_ex_bert/bert_model.ckpt'
    dict_path = './chinese_wwm_ex_bert/vocab.txt'
elif flag == 2:
    # roberta
    config_path = './chinese_roberta/bert_config.json'
    checkpoint_path = './chinese_roberta/bert_model.ckpt'
    dict_path = './chinese_roberta/vocab.txt'

token_dict = {}
with open(dict_path, 'r', encoding='utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)
#######################################函数#######################################
class data_generator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data[0]) // self.batch_size
        if len(self.data[0]) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            X1, X2, y = self.data
            idxs = list(range(len(self.data[0])))
            np.random.shuffle(idxs)
            T, T_, Y = [], [], []
            for c, i in enumerate(idxs):
                achievements = X1[i]
                requirements = X2[i]
                t, t_ = tokenizer.encode(first=achievements, second=requirements, max_len=64)
                T.append(t)
                T_.append(t_)
                Y.append(y[i])
                if len(T) == self.batch_size or i == idxs[-1]:
                    T = np.array(T)
                    T_ = np.array(T_)
                    Y = np.array(Y)
                    yield [T, T_], Y
                    T, T_, Y = [], [], []


def get_model():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    for l in bert_model.layers:
        l.trainable = True

    input1 = Input(shape=(None,))
    input2 = Input(shape=(None,))

    input = bert_model([input1, input2])

    input = Lambda(lambda x: x[:, 0])(input)

    output = Dense(2, activation='softmax')(input)

    model = Model([input1, input2], output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        metrics=['accuracy']
    )
    model.summary()
    return model

