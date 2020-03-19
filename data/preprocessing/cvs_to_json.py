# -*- coding: utf-8 -*-
"""
@Time ： 2020/3/7 17:28
@Auth ： joleo
@File ：cvs_to_json.py
"""
import json
import sys
import pandas as pd

def load_context(filename='../data/NCPPolicies_context_20200301/NCPPolicies_context_20200301.csv'):
    # filename：'data/NCPPolicies_context_20200301.csv'
    docid2context = {}
    f = True
    for line in open(filename,encoding='utf-8'):
        if f:
            f = False
            continue
        r = line.strip().split('\t')
        docid2context[r[0]] = r[1]
    return docid2context


def convert_trainset(src_filename, dst_filename):
    """
    :param src_filename: source filename
    :param dst_filename: destination filename
    :return:
    """
    docid2context = load_context()
    first_line = True
    fout = open(dst_filename, 'w', encoding='utf-8')
    for line in open(src_filename, encoding='utf-8'):
        if first_line:
            first_line = False
            continue
        r = line.strip().split('\t')
        rv = {'qid': r[0], 'context': docid2context[r[1]], 'query': r[2], 'answer': {'text': r[3]}}
        fout.write(json.dumps(rv, ensure_ascii=False) + '\n')


# def load_retrieval_docids():
#     q2docid = {}
#     for line in open('../baseline/data/query_docids_v1.csv', encoding='utf-8'):
#         r = line.strip().split('\t')
#         q2docid[r[0]] = r[1].split(',')[0]
#     return q2docid
def load_retrieval_docids():
    q2docid = {}
    for line in open('../data/query_docids_v1.csv', encoding='utf-8'):
        r = line.strip().split('\t')
        q2docid[r[0]] = r[1].split(',')[0]
    return q2docid
# q2docid = load_retrieval_docids()

def convert_testset(src_filename,dst_filename):
    """Convert test set to json format."""
    q2docid = load_retrieval_docids()
    docid2context = load_context()

    first_line = True
    fout = open(dst_filename, 'w', encoding='utf-8')
    for line in open(src_filename, encoding='utf-8'):
        if first_line:
            first_line = False
            continue
        r = line.strip().split('\t')
        if r[1] not in q2docid:
            print(f'cannot find retrieval results: {r[1]}')
        rv = {'qid': r[0], 'context': docid2context[q2docid.get(r[1],'c129c1bc387c312284c3ef61b551c432')], 'query': r[1], 'answer': {'text': ''}}
        fout.write(json.dumps(rv, ensure_ascii=False) + '\n')

def strStr(haystack, needle):
    """
    :type haystack: document
    :type needle: question
    :rtype: int
    """
    if not needle:
        return 0
    l1 = len(haystack)
    l2 = len(needle)

    if l1 < l2:
        return -1

    l = l1 - l2 + 1
    for i in range(l):
        if haystack[i:i + l2] == needle:
            return i
    return -1

def get_position(input_data):
    input_data['start_position'] = list(map(lambda x, y: strStr(x, y), input_data['text'], input_data['answer']))
    input_data['end_position'] = list(
        map(lambda x, y: x + len(y) - 1, input_data['start_position'], input_data['answer']))
    return input_data

def convert_trainset2(src_filename, dst_filename):
    """
    :param src_filename: source filename
    :param dst_filename: destination filename
    :return:
    """
    docid2context = load_context()
    first_line = True
    fout = open(dst_filename, 'w', encoding='utf-8')
    for line in open(src_filename, encoding='utf-8'):
        if first_line:
            first_line = False
            continue
        r = line.strip().split('\t')
        rv = {'qid': r[0], 'context': r[1], 'query': r[2], 'answer': {'text': r[3], 'span': [int(r[4]), int(r[5])]}}
        fout.write(json.dumps(rv, ensure_ascii=False) + '\n')

def convert_testset(src_filename,dst_filename):
    """Convert test set to json format."""
    q2docid = load_retrieval_docids()
    docid2context = load_context()

    first_line = True
    fout = open(dst_filename, 'w', encoding='utf-8')
    for line in open(src_filename, encoding='utf-8'):
        if first_line:
            first_line = False
            continue
        r = line.strip().split('\t')
        if r[1] not in q2docid:
            print(f'cannot find retrieval results: {r[1]}')
        rv = {'qid': r[0], 'context': docid2context[q2docid.get(r[1],'c129c1bc387c312284c3ef61b551c432')], 'query': r[1], 'answer': {'text': ''}}
        fout.write(json.dumps(rv, ensure_ascii=False) + '\n')

def stat_length(filename):
    #
    df = pd.read_csv(filename, sep='\t', error_bad_lines=False)
    df['context_length'] = df['text'].apply(len)
    print(f"context length: {df['context_length'].mean()}")
from preprocessing.load_data import read_train,read_context,read_test
if  __name__ == '__main__':
    # stat_length('data/NCPPolicies_context_20200301.csv')
    # train_filename='../data/NCPPolicies_train_20200301/NCPPolicies_train_20200301.csv'
    test_filename='../data/NCPPolicies_test/NCPPolicies_test.csv'
    # context_filename = '../data/NCPPolicies_context_20200301/NCPPolicies_context_20200301.csv'
    # train = read_train(train_filename)
    # context = read_context(context_filename)
    # train = train.merge(context, how='left', on='docid')[['id','text','question','answer']]
    # train = get_position(train)
    # print(train.columns)
    # train.to_csv('../data/train_pro.csv', index=0, sep='\t')
    # train_filename2= '../data/train_pro.csv'
    # convert_trainset2(src_filename=train_filename2,dst_filename='../data/train_an.json')

    convert_testset(src_filename=test_filename,dst_filename='../data/test.json')


