#!/usr/bin/python

import numpy, os, time, math
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
import ConfigParser
import os.path
from sklearn.metrics import mean_squared_error
from keras.preprocessing import sequence
import json, time


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

def readScores(config_file):
    settings = getConfig(config_file)
    score_items = settings.items("scores")
    scores = {}
    for pair in score_items:
        scores[pair[0]] = int(pair[1])

    return scores


def getScore(actions_scores, action_name):
    if (action_name in actions_scores):
        score = actions_scores[action_name]
    else:
        score = 0

    return score


"""
    Calculate root mean squared error for each action in the dataset
"""
def calculate_rmse(dataset, predictions):
    rmse_totals = []
    for i in range(len(dataset)):
        rmse = math.sqrt(mean_squared_error(dataset[i], predictions[i]))
        rmse_totals.append(rmse)

    return rmse_totals


"""
    Convert an array of values into a dataset matrix
"""
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

def get_prediction_output(dataset):
    dataY = {}
    for action_name, actions in dataset.iteritems():
        for i in range(len(actions) - 1):
            dataY[action_name] = actions[i+1]

    return dataset, dataY


def collection_values_to_array(dataset):
    dataset = numpy.array(dataset)
    new_dataset = []
    for row in dataset:
        row_array = numpy.array(eval(row[0]))
        new_dataset.append(row_array)

    return numpy.array(new_dataset)

## Transform operations over data ##

def scale(dataset):
    scaler = StandardScaler() #MinMaxScaler(feature_range=(0, 1))

    dataset_scaled = scaler.fit_transform(numpy.float32(dataset))

    return scaler, dataset_scaled


def invert_scale(scaler, X):
    return scaler.inverse_transform(X)


def padding(dataset, length):
    # Padding (from left, otherwise results are affected in Keras)
    return sequence.pad_sequences(dataset, maxlen=length, padding='pre')  # shape (n, length)


def extract_hour(unixtime):
    hour = time.strftime("%H", time.gmtime(float(unixtime)))
    return hour

def normalize(dataset):
    # Normalize X, shape (n_samples, n_features)
    return normalize(dataset)

    # datasetX = datasetX / float(self.settings.getint("Data", "vocabulary_size"))

## File operations ##

def create_iter_generator(filename):
    with open(filename) as file:
        for line in file:
            yield json.loads(line)

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

