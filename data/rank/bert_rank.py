# -*- coding: utf-8 -*-
"""
@Time ： 2020/3/10 20:54
@Auth ： joleo
@File ：bert_rank.py
"""
import pandas as pd
from rank.bert_model import *
from rank.config import *

#######################################加载数据#######################################
train = pd.read_csv('data/train_recall.csv')
test = pd.read_csv('data/test_recall.csv')

question = train['question'].values
text = train['text'].values

labels = train['label'].astype(int).values - 1
labels_cat = to_categorical(labels)
labels_cat = labels_cat.astype(np.int32)

test_question = test['question'].values
test_text = test['text'].values

#######################################评估预测函数#######################################
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
        score, acc, f1 = self.evaluate()
        if score > self.best:
            self.best = score
            self.early_stopping = 0
            model.save_weights('./data/model_save/bert{}.w'.format(fold))
        else:
            self.early_stopping += 1
        logger.info('lr: %.6f, epoch: %d, score: %.4f, acc: %.4f, f1: %.4f,best: %.4f\n' % (
        self.lr, epoch, score, acc, f1, self.best))

    def evaluate(self):
        self.predict = []
        prob = []
        val_x1, val_x2, val_y, val_cat = self.val_data
        for i in tqdm(range(len(val_x1))):
            question = val_x1[i]
            text = val_x2[i]

            t1, t1_ = tokenizer.encode(first=question, second=text)
            T1, T1_ = np.array([t1]), np.array([t1_])
            _prob = model.predict([T1, T1_])
            oof_train[self.val_index[i]] = _prob[0]
            self.predict.append(np.argmax(_prob, axis=1)[0] + 1)
            prob.append(_prob[0])

        score = f1_score(val_y + 1, self.predict , average='macro')
        acc = accuracy_score(val_y + 1, self.predict)
        f1 = f1_score(val_y + 1, self.predict, average='macro')
        return score, acc, f1


def predict(data):
    prob = []
    val_x1, val_x2 = data
    for i in tqdm(range(len(val_x1))):
        question = val_x1[i]
        text = val_x2[i]

        t1, t1_ = tokenizer.encode(first=question, second=text)
        T1, T1_ = np.array([t1]), np.array([t1_])
        _prob = model.predict([T1, T1_])
        prob.append(_prob[0])
    return prob
#######################################开始训练#######################################
oof_train = np.zeros((len(train), 4), dtype=np.float32)
oof_test = np.zeros((len(test), 4), dtype=np.float32)
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
for fold, (train_index, valid_index) in enumerate(skf.split(question, labels)):
    logger.info('================     fold {}        ==============='.format(fold))
    x1 = question[train_index]
    x2 = text[train_index]
    y = labels_cat[train_index]

    val_x1 = question[valid_index]
    val_x2 = text[valid_index]
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
    # model.load_weights('./data/model_save/bert{}.w'.format(fold))
    oof_test += predict([test_question, test_text])
    K.clear_session()

######################################结果评估和提交#######################################
oof_test /= 5
cv_score = 1.0 / (1 + mean_absolute_error(labels + 1, np.argmax(oof_train, axis=1) + 1))
np.savetxt('./data/model_save/train_bert_prob_{}.txt'.format(cv_score), oof_train)
np.savetxt('./data/model_save/test_bert_prob_{}.txt'.format(cv_score), oof_test)
print(cv_score)
test['label'] = np.argmax(oof_test, axis=1) + 1
test[['Guid', 'label']].to_csv('./data/submit/bert_{}.csv'.format(cv_score), index=False)