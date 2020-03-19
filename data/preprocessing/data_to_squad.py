# -*- coding: utf-8 -*-
"""
@Time ： 2020/3/7 20:07
@Auth ： joleo
@File ：data_to_squad.py
"""
import json

def to_english(string):
    return ' '.join(list(string))

def convert_to_squad(filepath):
    """Convert to squad-like dataset."""
    rv = {'data': []}
    for line in open(filepath, encoding='utf-8'):
        datum = json.loads(line)
        context = to_english(datum['context'])
        query = to_english(datum['query'])
        orig_answer = datum['answer']['text']
        if 'span' not in datum['answer']:
            print('no sapn')
            continue
        span = datum['answer']['span']
        span = [span[0]*2, span[1]*2]
        answer = context[span[0]:span[1]+1]
        ex = {'title': 'fake title','paragraphs':[{'context': context, "qas":[{"answers": [{"answer_start": span[0], "text": answer}], 'question': query, 'id': datum['qid']}]}]}
        rv['data'].append(ex)
    json.dump(rv, open('../data/train_squad.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)


def convert_test_to_squad(filepath):
    rv = {'data': []}
    for line in open(filepath, encoding='utf-8'):
        datum = json.loads(line)
        context = to_english(datum['context'])
        query = to_english(datum['query'])
        ex = {'title': 'fake title','paragraphs':[{'context': context, "qas":[{"answers": [{"answer_start": 0, "text": ''}], 'question': query, 'id': datum['qid']}]}]}
        rv['data'].append(ex)
    json.dump(rv, open('../data/test_squad.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)


if __name__ == '__main__':
    # convert_to_squad('../data/train_answer.json')
    convert_test_to_squad('../data/test.json')
