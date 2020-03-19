# -*- coding: utf-8 -*-
"""
@Time ： 2020/3/7 20:07
@Auth ： joleo
@File ：load_data.py
"""
import pandas as pd

def read_context(input_file):
    context = pd.DataFrame()
    with open(input_file, encoding='utf-8') as f:
        docid, text = [], []
        for line in f:
            value = line.split('\t')  # [1:]
            docid.append(value[0])
            text.append(value[1])
    context['docid'] = docid[1:]
    context['text'] = text[1:]
    context['text'] = context['text'].map(lambda x: x.replace('\n', ''))
    return context

def read_train(input_file):
    train_data = pd.DataFrame()
    # './data/NCPPolicies_train_20200301/NCPPolicies_train_20200301.csv'
    with open(input_file, encoding='utf-8') as f:
        id, docid, question, answer = [], [], [], []
        for line in f:
            value = line.split('\t')  # [1:]
            id.append(value[0])
            docid.append(value[1])
            question.append(value[2])
            answer.append(value[3])
    train_data['id'] = id[1:]
    train_data['docid'] = docid[1:]
    train_data['question'] = question[1:]
    train_data['answer'] = answer[1:]
    train_data['question'] = train_data['question'].map(lambda x: x.replace('\n', ''))
    train_data['answer'] = train_data['answer'].map(lambda x: x.replace('\n', ''))
    return train_data

def read_test(input_file):
    valid_data = pd.DataFrame()
    with open(input_file, encoding='utf-8') as f:
        id = []
        question = []
        for line in f:
            value = line.split('\t')
            id.append(value[0])
            question.append(value[1])
    valid_data['id'] = id[1:]
    valid_data['question'] = question[1:]
    valid_data['question'] = valid_data['question'].map(lambda x: x.replace('\n', ''))
    return valid_data
