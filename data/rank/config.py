# -*- coding: utf-8 -*-
"""
@Time ： 2020/3/10 20:59
@Auth ： joleo
@File ：config.py
"""
import os
import time
import logging
import tensorflow as tf
#######################################配置参数#######################################
# 参数
flag = 1
epoch = 5
MAX_LEN = 30
learning_rate = 5e-5
min_learning_rate = 1e-5

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

file_path = './data/log/'
# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setlabel(logging.DEBUG)

# 创建一个handler，
timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
fh = logging.FileHandler(file_path + 'log_' + timestamp + '.txt')
fh.setlabel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setlabel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('[%(asctime)s][%(labelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)