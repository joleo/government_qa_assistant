import os
import re
import jieba
import json
import time
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
import keras.backend.tensorflow_backend as KTF
from tqdm import tqdm
from keras.layers import *
from keras.models import Model
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score, roc_auc_score,log_loss
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
import pickle
import tensorflow as tf
import os
from keras.utils import multi_gpu_model


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

learning_rate = 5e-5
min_learning_rate = 1e-5
MAX_LEN = 512
epoch = 1
n_fold = 5
n = 2
flag = 3
train_flag = 2
################################################################
weight_path = '/data/data01/liyang099/com/weight/chinese'
if flag == 1:
    # base bert
    config_path = weight_path + '/chinese_wwm_ex_bert/bert_config.json'
    checkpoint_path = weight_path + '/chinese_wwm_ex_bert/bert_model.ckpt'
    dict_path = weight_path + '/chinese_wwm_ex_bert/vocab.txt'
elif flag == 2:
    # roberta
    config_path = weight_path + '/chinese_roberta/bert_config.json'
    checkpoint_path = weight_path + '/chinese_roberta/bert_model.ckpt'
    dict_path = weight_path + '/chinese_roberta/vocab.txt'
elif flag == 3:
    config_path = weight_path + '/chinese_wwm_ex_L12/bert_config.json'
    checkpoint_path = weight_path + '/chinese_wwm_ex_L12/bert_model.ckpt'
    dict_path = weight_path + '/chinese_wwm_ex_L12/vocab.txt'
elif flag == 4:
    config_path = weight_path + '/chinese_roberta_wwm_large/bert_config.json'
    checkpoint_path = weight_path + '/chinese_roberta_wwm_large/bert_model.ckpt'
    dict_path = weight_path + '/chinese_roberta_wwm_large/vocab.txt'

token_dict = {}
with open(dict_path, 'r', encoding='utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
tokenizer = Tokenizer(token_dict)



file_path = './data/log/'
# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

# 创建一个handler，
timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
fh = logging.FileHandler(file_path + 'log_' + timestamp + '.txt')
fh.setLevel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)

train = pd.read_csv('data/recall/train_recall.csv')
train = train.drop_duplicates().reset_index(drop=True)
print(train.shape)

labels = train['label'].values
labels_cat = to_categorical(train['label'])
labels_cat = labels_cat.astype(np.int32)
# print(labels.value_counts())

test = pd.read_csv('data/recall/test_recall.csv')
print(test.shape)
test = test.drop_duplicates().reset_index(drop=True)
print(test.shape)


train_achievements = train.question.values
train_requirements = train.text.values
test_achievements = test.question.values
test_requirements = test.text.values

class data_generator:
    def __init__(self, data, batch_size=16):
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
                t, t_ = tokenizer.encode(first=achievements, second=requirements, max_len=512)
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

    T1 = Input(shape=(None,))
    T2 = Input(shape=(None,))

    T = bert_model([T1, T2])

    T = Lambda(lambda x: x[:, 0])(T)

    output = Dense(1, activation='sigmoid')(T)

    model = Model([T1, T2], output)
    if train_flag == 1:
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(1e-5),  # 用足够小的学习率
            metrics=['accuracy']
        )
    else:
        model = multi_gpu_model(model, gpus= 2)  # 使用几张显卡n等于几
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(1e-5),  # 用足够小的学习率
            metrics=['accuracy']
        )
    return model


class Evaluate(Callback):
    def __init__(self, val_data, val_index):
        self.score = []
        self.best = 0.
        self.early_stopping = 0
        self.val_data = val_data
        self.val_index = val_index
        self.predict = []
        self.lr = 0
        self.passed = 0

    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低"""
        if self.passed < self.params['steps']:
            self.lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, self.lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            self.lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            self.lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, self.lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        score, auc, loss = self.evaluate()
        if score > self.best:
            self.best = score
            self.early_stopping = 0
            model.save_weights('./data/model_save/bert{}.w'.format(fold))
        else:
            self.early_stopping += 1
        logger.info('lr: %.6f, epoch: %d, score: %.4f, auc: %.4f, loss: %.4f,best: %.4f\n' % (
        self.lr, epoch, score, auc, loss, self.best))

    def evaluate(self):
        self.predict = []
        prob = []
        val_x1, val_x2, val_y, val_cat = self.val_data
        for i in tqdm(range(len(val_x1))):
            achievements = val_x1[i]
            requirements = val_x2[i]

            t1, t1_ = tokenizer.encode(first=achievements, second=requirements, max_len=512)
            T1, T1_ = np.array([t1]), np.array([t1_])
            _prob = model.predict([T1, T1_])
            oof_train[self.val_index[i]] = _prob[0]
            self.predict.append(_prob[0])
            prob.append(_prob[0])

        # score = mean_absolute_error(val_y, self.predict))
        pred_y = [1 if x >= 0.5 else 0 for x in self.predict]
        score = f1_score(val_y, pred_y, average='macro')
        auc = roc_auc_score(val_y, self.predict)
        loss = log_loss(val_y, self.predict)
        return score, auc, loss


def predict(data):
    prob = []
    val_x1, val_x2 = data
    for i in tqdm(range(len(val_x1))):
        achievements = val_x1[i]
        requirements = val_x2[i]

        t1, t1_ = tokenizer.encode(first=achievements, second=requirements, max_len=512)
        T1, T1_ = np.array([t1]), np.array([t1_])
        _prob = model.predict([T1, T1_])
        prob.append(_prob[0])
    return prob


oof_train = np.zeros((len(train), 1), dtype=np.float32)
oof_test = np.zeros((len(test), 1), dtype=np.float32)
skf = StratifiedKFold(n_splits=1000, shuffle=True, random_state=42)
for fold, (train_index, valid_index) in enumerate(skf.split(train_achievements, labels)):
    logger.info('================     fold {}        ==============='.format(fold))
    if fold in [0,1]:
        x1 = train_achievements[train_index]
        x2 = train_requirements[train_index]
        y = labels[train_index]

        val_x1 = train_achievements[valid_index]
        val_x2 = train_requirements[valid_index]
        val_y = labels[valid_index]
        val_cat = labels_cat[valid_index]

        train_D = data_generator([x1, x2, y])
        evaluator = Evaluate([val_x1, val_x2, val_y, val_cat], valid_index)

        model = get_model()
        model.fit_generator(train_D.__iter__(),
                            steps_per_epoch=len(train_D),
                            epochs=epoch,
                            callbacks=[evaluator]
                            )
        # model.load_weights('./data1/model_save/bert{}.w'.format(fold))
        oof_test += predict([test_achievements, test_requirements])
        K.clear_session()
    else: continue
	
oof_test /= n_fold
cv_score = 1.0 / (1 + mean_absolute_error(labels+1, np.argmax(oof_train, axis=1) + 1))
np.savetxt('./data/model_save/train_bert_prob_{}.txt'.format(cv_score), oof_train)
np.savetxt('./data/model_save/test_bert_prob_{}.txt'.format(cv_score), oof_test)
print(cv_score)	
	
	
import pandas as pd
import numpy as np
# oof_test = np.loadtxt('data/model_save/test_bert_prob_0.9318895465012275.txt')
# test = pd.read_csv('data/recall/test_recall.csv')
# test = test.drop_duplicates().reset_index(drop=True)
test['prob'] = oof_test
test['label'] = test['prob'].map(lambda x:1 if x >0.1 else 0)
test.shape,len(oof_test)	
	
test['rank'] = test.groupby(['id'])['prob'].rank(method='max', ascending=False)#.reset_index()
test['rank'] = test['rank'].astype('int')
test.sort_values(by='rank', inplace=True)
test2 =test.drop_duplicates(subset=['id'])
test2[['id','question','text']].to_csv('data/recall/test_recall_m.csv', index=0)
	
	
	
from load_data import *
test_recall = pd.read_csv('data/recall/test_recall_m.csv')
context = read_context('data/NCPPolicies_context_20200301.csv')	
context['text'] = context['text'].map(lambda x: str(x).strip().replace('\n',''))
test_recall['text'] = test_recall['text'].map(lambda x: str(x).strip())
test_recall2 = test_recall.merge(context, how='left',on='text')	
test_recall2[['question','docid']].to_csv('data/query_docids_v1.csv', sep='\t',index=0)	