## *赛题背景*

2020年春节期间，新型冠状病毒感染肺炎疫情迅速向全国蔓延，全国上下共同抗击疫情。新冠疫情不仅对人民生命安全造成了威胁，也对很多企业的生产、发展产生了影响，按照党中央和国务院关于在做好疫情防控前提下，有序做好企业复工复产工作要求，国家各级政府部门、各个行业积极主动应对，相继出台了一系列惠民惠企政策。这些政策内容丰富、涵盖面广，涉及到了稳定就业岗位、减轻企业负担、强化资金补贴、和谐劳动关系等方方面面，给予企业实实在在的支持，切实帮助各类企业（特别是中小微企业）共度疫情难关。

为了更好的帮助各行业企业准确掌握相关政策，疫情政务问答助手旨在通过对惠民惠企政策数据的收集与处理，通过人机对话式问答的方式，对用户提出的政策疑问快速、准确地定位相关政策内容返回给用户。

## *赛题任务*
该赛题旨在评测智能问答算法能力，是问题理解、内容搜索、答案提取等多个环节综合能力的集成。任务将提供以疫情为主的政策数据集、用户问题以及标注好的答案片段，参赛者可自行通过对政策数据的分析、处理和组织，利用训练数据集训练智能问答算法，并在测试数据集上进行评测，评测指标为最终返回答案的准确性。

## 文件执行
### 数据处理
```shell
python preprocessing/cvs_to_json.py
python preprocessing/data_to_squad.py
```
### 召回
```shell
python recall/bm_recall.py
```
## 排序
```shell
python rank/bert_rank.py
```
## 训练和预测
```shell
python src/run_squad.py 
-- model_type bert \
-- model_name_or_path bert-base-chinese\
-- do_train \
-- do_eval \
-- train_file data/train_squad.json \
-- output_dir debug_squad_v1/\
-- predict_file data/test_squad.json \
-- output_dir data/result/
```
## 预测结果格式转换
```shell
python format_submission.py
```
