# -*- coding: utf-8 -*-
"""
@Time ： 2020/3/8 11:06
@Auth ： joleo
@File ：bm_recall.py
"""
# https://blog.csdn.net/byn12345/article/details/81112973
from gensim import corpora, models, similarities
from load_data import read_test,read_context,read_train
from collections import defaultdict
import jieba_fast as jieba
from recall.rank_bm25 import BM25Okapi
import pandas as pd

# jieba.load_userdict(file_name='')
jieba.add_word('复工')
jieba.add_word('稳岗')
jieba.add_word('医保局')
jieba.add_word('暖企')
# jieba.del_word('医保局')

paper_data = read_context('data/NCPPolicies_context_20200301/NCPPolicies_context_20200301.csv')
train_data = read_train('./data/NCPPolicies_train_20200301/NCPPolicies_train_20200301.csv')
valid_data = read_test('./data/NCPPolicies_test/NCPPolicies_test.csv')

train_data['question'] = train_data['question'].map(lambda x: x.replace('\n', ''))
train_data['answer'] = train_data['answer'].map(lambda x: x.replace('\n', ''))
valid_data['question'] = valid_data['question'].map(lambda x: x.replace('\n', ''))
paper_data['text'] = paper_data['text'].map(lambda x: x.replace('\n', ''))

# 分词
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist('data/stopwords.txt')
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr
# list(paper_data['text'].head(1).apply(seg_sentence))
def seg_sentence_char(sentence,filepath):
    sentence_seged = sentence.strip()
    stopwords = stopwordslist(filepath+'stopwords.txt')
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr

paper_data['paper_text'] = paper_data['text'].apply(seg_sentence)#.map(lambda x: x[:-1])
train_data['question_pro'] = train_data['question'].apply(seg_sentence)#.map(lambda x: x[:-1])
valid_data['question_pro'] = valid_data['question'].apply(seg_sentence)#.map(lambda x: x[:-1])
# list( paper_data['text'].head(1).apply(seg_sentence).map(lambda x: x[:-1]))
##########################################################################################
def load_context(df):
    # filename：'data/NCPPolicies_context_20200301.csv'
    docid2context = {}
    for line in df.values:
        docid2context[line[2]] = line[0]
    return docid2context
docid2context = load_context(paper_data)

corpus = list(paper_data['paper_text'].values)
tokenized_corpus = [doc.split(" ") for doc in corpus]
# dictionary = corpora.Dictionary(tokenized_corpus)
# corpus = [dictionary.doc2bow(text) for text in tokenized_corpus]

bm25 = BM25Okapi(tokenized_corpus)

def get_candidate_set2(df, num=10):
    # 构造排序数据
    candidate_set = pd.DataFrame()
    for id,query in zip(df['id'].values,df['question_pro'].values):
        tmp = pd.DataFrame()
        tokenized_query = query.split(" ")
        candidate = bm25.get_top_n(tokenized_query, documents=corpus, n=num)
        cds = [docid2context[i] for i in candidate]
        ids = []
        for i in range(len(cds)):
            ids.append(id)
        tmp['id'] = ids
        tmp['docid'] = cds
        # print(tmp.shape)
        candidate_set = pd.concat([tmp,candidate_set])
    return candidate_set

candidate_set = get_candidate_set2(train_data.head(1))
def get_candidate_set(df, num=10):
    # 构造id:list结构
    candidate_set = pd.DataFrame()
    ids, candidates = [], []
    for id,query in zip(df['id'].values,df['question_pro'].values):
        ids.append(id)
        tokenized_query = query.split(" ")
        candidate = bm25.get_top_n(tokenized_query, documents=corpus, n=num)
        cds = [docid2context[i] for i in candidate]
        candidates.append(cds)
    candidate_set['id'] = ids
    candidate_set['docid'] = candidates
    return candidate_set

def get_candidate_set_res(df,num):
    candidate_set = get_candidate_set(df, num)
    candidate_set['docid'] = candidate_set['docid'].map(lambda x: str(x).replace('[','').replace(']','').replace(' ',''))
    res = df.merge(candidate_set,how='left',on='id')[['question','docid']]
    res.to_csv('data/query_docids_v1.csv', index=0, header=0, sep='\t')
    return res

def get_recall_res(df,paper_data, num=10):
    test_res = get_candidate_set2(df, num)
    test = test_res.merge(df[['id', 'question']],how='left',on='id')[['id','question','docid']]
    # test.rename(columns={'candidate':'docid'}, inplace=True)
    test = test.merge(paper_data,how='left',on='docid')[['id','question','text']]
    # test.to_csv('data/recall/test_recall.csv', index=0)
    return test

if __name__ == '__main__':
    test_recall_res = get_candidate_set2(valid_data, 20)
    test_recall_res.to_csv('data/recall/test_recall.csv', index=0)

    trn_recall_res = get_candidate_set2(train_data, 20)
    label = trn_recall_res.merge(train_data,how='left',on='id')
    label['label'] = list(map(lambda x,y: 1 if x==y else 0, label['docid_x'],label['docid_y']))
    label.rename(columns={'docid_x':'docid'}, inplace=True)
    label = label.merge(paper_data,how='left',on='docid')[['id','question','text','label']]
    tmp = train_data.merge(paper_data,how='left',on='docid')[['id', 'question', 'text']]
    tmp['label'] = 1
    train_recall_res = pd.concat([label, tmp]).drop_duplicates(subset=['id', 'question', 'text','label']).reset_index(drop=True)

    # label[label['docid_x'] == label['docid_y']].shape[0] / train_data['docid'].shape[0]
    train_recall_res.to_csv('data/recall/train_recall.csv', index=0)

#########################################################################