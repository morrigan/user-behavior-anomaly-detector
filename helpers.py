import csv
import pandas as pd
import re
import random
import tensorflow as tf

def read_data(filename):
    for serialized_example in tf.python_io.tf_record_iterator(filename):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        action = example.features.feature['action'].int64_list.value
        action_len = example.features.feature['action_len'].int64_list.value
        label = example.features.feature['label'].int64_list.value[0]
        # do something
        print action, action_len, label


def read_single_example(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'action': tf.FixedLenFeature([200], tf.int64),
            #'action_len': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([1], tf.int64),
        })
    print features['action']
    return features['action'], features['label']

