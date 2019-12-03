# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf

"""
Created by gaoyw on 2019/12/1
"""
from config import *
import os
import collections
import pandas as pd
import tensorflow as tf
# import tensorflow_hub as hub
from datetime import datetime
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert import modeling
from config import *
from tqdm import tqdm
from model import *
from data_handler import *

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")


def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """"Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_ids)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        data_set = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples, len(LABEL_COLUMNS)], dtype=tf.int32),
        })

        if is_training:
            data_set = data_set.repeat().shuffle(buffer_size=100)

        data_set = data_set.batch(batch_size=batch_size, drop_remainder=drop_remainder)

        return data_set

    return input_fn


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def _decode_record(record):
        """Decodes a record to a TensorFlow example."""
        name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([6], tf.int64),
            "is_real_example": tf.FixedLenFeature([], tf.int64),
        }
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        d = d.map(_decode_record)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.batch(batch_size, drop_remainder=drop_remainder)

        return d

    return input_fn


def main(args):
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    run_config = tf.estimator.RunConfig(
        model_dir=save_model_dir,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        keep_checkpoint_max=1,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

    bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG)
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(LABEL_COLUMNS),
        init_checkpoint=BERT_INIT_CHKPNT,
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=False,
        use_one_hot_embeddings=False)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"batch_size": BATCH_SIZE})

    train_input_fn = file_based_input_fn_builder(
        input_file=train_tf_record_path,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=True)

    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_tf_record_path,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False)

    if FLAGS.do_train and FLAGS.do_eval:
        print(f'Beginning Training and evaluating!')
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=None)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    elif FLAGS.do_train:
        print(f'Beginning Training!')
        current_time = datetime.now()
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        print("Training took time ", datetime.now() - current_time)
    elif FLAGS.do_eval:
        print(f'Beginning evaluating!')
        current_time = datetime.now()
        result = estimator.evaluate(input_fn=eval_input_fn, steps=None)  # None 表示跑完整个数据集
        print("Training took time ", datetime.now() - current_time)
        output_eval_file = os.path.join("data", "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    if FLAGS.do_predict:
        predict_sample = "If you have a look back at the source, the information I updated was the correct form. I can only guess the source hadn't updated. I shall update the information once again but thank you for your message."
        input_sample = InputExample(guid=0, text_a=predict_sample, labels=[0, 0, 0, 0, 0, 0])
        tokenization.validate_case_matches_checkpoint(True, BERT_INIT_CHKPNT)
        tokenizer = tokenization.FullTokenizer(vocab_file=BERT_VOCAB, do_lower_case=True)
        feature = convert_single_example(input_sample, MAX_SEQ_LENGTH, tokenizer)
        predict_input_fn = input_fn_builder([feature], MAX_SEQ_LENGTH, False, False)
        predictions = estimator.predict(predict_input_fn)
        probabilities = []
        for (i, prediction) in enumerate(predictions):
            preds = prediction["probabilities"]
            probabilities.append(preds)
        print(probabilities[0])


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
