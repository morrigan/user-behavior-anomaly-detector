#!/usr/bin/python

import tensorflow as tf
import numpy, os, time, math
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import ConfigParser
import os.path
from sklearn.metrics import mean_squared_error
from keras.preprocessing import sequence
import json



def ConfigSectionMap(settings_file, section):
    Config = ConfigParser.ConfigParser()
    Config.read(settings_file)

    dict1 = {}
    options = Config.options(section)
    for option in options:
        try:
            dict1[option] = Config.get(section, option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1

def getConfig(settings_file):
    Config = ConfigParser.ConfigParser()
    if (os.path.isfile(settings_file)):
        Config.read(settings_file)
        return Config
    else:
        print "No settings file found"
        exit()

def read_data(filename):
    actions = []
    lengths = []
    labels = []
    for serialized_example in tf.python_io.tf_record_iterator(filename):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        action = example.features.feature['action'].int64_list.value
        action_len = example.features.feature['action_len'].int64_list.value
        label = example.features.feature['label'].int64_list.value[0]
        actions.append(action)
        labels.append(label)
        lengths.append(action_len)
        # print action, action_len, label
    return actions, lengths, labels

## Operations
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def linear(input_, output_size, layer_scope, stddev=0.02, bias_start=0.0):
    if (input_.count):
        print input_
        shape = input_.get_shape().as_list()
        print shape
    else:
        shape = input_

    with tf.variable_scope(layer_scope):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        return tf.matmul(input_, matrix) + bias


def calculate_rmse(dataset, predictions):
    rmse_totals = []
    for i in range(len(dataset)):
        rmse = math.sqrt(mean_squared_error(dataset[i][0], predictions[i]))
        rmse_totals.append(rmse)

    return rmse_totals

# Convert an array of values into a dataset matrix
def convert_dataset(dataset, look_back=3):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        b = dataset[i + look_back]
        dataX.append(a)
        dataY.append(b)

    return numpy.array(dataX), numpy.array(dataY)

def get_real_predictions(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset) - 1):
        b = dataset[i + 1]
        dataX.append(dataset[i])
        dataY.append(b)

    return numpy.array(dataX), numpy.array(dataY)


def collection_values_to_array(dataset):
    dataset = numpy.array(dataset)
    new_dataset = []
    for row in dataset:
        row_array = numpy.array(eval(row[0]))
        new_dataset.append(row_array)

    return numpy.array(new_dataset)

def scale(dataset):
    scaler = MinMaxScaler(feature_range=(0, 1))

    dataset_scaled = scaler.fit_transform(dataset)

    return scaler, dataset_scaled

def invert_scale(scaler, X):
    return scaler.inverse_transform(X)

def padding(dataset, length):
    # Padding (from left, otherwise results are affected in Keras)
    return sequence.pad_sequences(dataset, maxlen=length, padding='pre')  # shape (n, length)


## File watchers

def watch_file(filename, interval):
    file = open(filename, "r")
    file.seek(0, 2)
    while True:
        line = file.readline()
        if not line:
            time.sleep(interval)
            continue
        yield line


def tail_F(some_file):
    first_call = True
    while True:
        try:
            with open(some_file) as input:
                if first_call:
                    input.seek(0, 2)
                    first_call = False
                latest_data = input.read()
                while True:
                    if '\n' not in latest_data:
                        latest_data += input.read()
                        if '\n' not in latest_data:
                            yield ''
                            if not os.path.isfile(some_file):
                                break
                            continue
                    latest_lines = latest_data.split('\n')
                    if latest_data[-1] != '\n':
                        latest_data = latest_lines[-1]
                    else:
                        latest_data = input.read()
                    for line in latest_lines[:-1]:
                        yield line + '\n'
        except IOError:
            yield ''


def create_iter_generator(filename):
    with open(filename) as file:
        for line in file:
            yield json.loads(line)