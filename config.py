# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow as tf
"""
Created by gaoyw on 2019/12/1
"""
BERT_VOCAB = 'bert_ckpt/vocab.txt'
BERT_INIT_CHKPNT = 'bert_ckpt/bert_model.ckpt'
BERT_CONFIG = 'bert_ckpt/bert_config.json'
save_model_dir = 'model_dir'

train_data_path = "data/train.csv"
test_data_path = "data/train.csv"
train_tf_record_path = "data/train.tfrecord"
eval_tf_record_path = "data/eval.tfrecord"

MAX_SEQ_LENGTH = 128

ID = 'id'
DATA_COLUMN = 'comment_text'
LABEL_COLUMNS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

TRAIN_VAL_RATIO = 0.9
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 1.0
# Warmup is a period of time where hte learning rate
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 1000
SAVE_SUMMARY_STEPS = 500

train = pd.read_csv(train_data_path)
LEN = train.shape[0]
size_train = int(TRAIN_VAL_RATIO * LEN)
x_train = train[:size_train]
print(len(x_train))
num_train_steps = int(len(x_train) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)