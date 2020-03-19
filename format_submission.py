#!/usr/bin/env python
# coding: utf-8

import json

def load_test():
    qid2query = {}
    f = True
    for line in open('data/NCPPolicies_test.csv'):
        if f:
            f=False
            continue
        r = line.strip().split('\t')
        qid2query[r[0]] = r[1]
    return qid2query


def format_submission():
    qid2query = load_test()
    res = json.load(open('debug_squad_v1/predictions_.json'))
    fout = open('data/submit.csv', 'w')
    fout.write(f'qid\tdocid\tanswer\n')
    for k, v in res.items():
        v = v.replace(' ', '')
        fout.write(f'{k}\t123123\t{v}\n')
    for qid in (qid2query.keys() - res.keys()):
        fout.write(f'{qid}\t12asd\tfake answer\n')


if __name__ == '__main__':
    format_submission()
