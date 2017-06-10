#!/usr/bin/python

import sys, getopt, json

from lstm import LSTM
from prepare_data import action_to_vector, restore_vocabulary, create_and_save_vocabulary, tokenizer_fn
from helpers import tail_F, create_iter_generator, readScores

#-------------------#
LOG_FILE = "/var/log/osquery/osqueryd.results.log"
ALGORITHM = "LSTM"
CONFIG = "settings.ini"
#-------------------#
# Unchangeable parameters
has_new_data = False
#-------------------#

def forecast_lstm(actions):
    lstm = LSTM(CONFIG)
    model = lstm.load_model()

    vocabulary = restore_vocabulary()
    actions_scores = readScores(CONFIG)
    global has_new_data
    actions_vectorized = []
    scores = []
    for action in actions:
        if (action != ''):
            action = json.loads(action)
            action_in_vector = action_to_vector(action, vocabulary)
            actions_vectorized.append(action_in_vector)

            action_name = action['name']#.replace(action_name_prefix, '')
            if (action_name in actions_scores):
                score = actions_scores[action_name]
            else:
                score = 0
            scores.append(score)
            has_new_data = True

        elif (action == '' and has_new_data == True):
            print "Done reading query results."
            predicted, history = lstm.forecast(model, actions_vectorized)

            if (len(predicted) > 0):
                print "Scores for each action (anomaly probability)"
                print lstm.calculate_score(actions_vectorized, predicted, scores)

            has_new_data = False
            # Reset previous query log
            scores = []
            actions_vectorized = []

def train_lstm():
    lstm = LSTM(CONFIG)

    vocabulary = create_and_save_vocabulary(LOG_FILE)

    print "Start preprocessing data"
    iter_generator = create_iter_generator(LOG_FILE)
    actions_vectorized = []
    for i, row in enumerate(iter_generator):
        action_vector = action_to_vector(row, vocabulary)
        actions_vectorized.append(action_vector)
    print "End preprocessing data"

    model = lstm.get_model()
    rmse = lstm.train_on_dataset(actions_vectorized, model)


def call_ocsvm():
    print "OCSVM not yet supported."
    sys.exit(2)

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "lac", ["log=", "algorithm=", "config="])
        if (len(args) == 0):
            print "No arguments given. Using -l {} -a {} -c {}".format(LOG_FILE, ALGORITHM, CONFIG)
    except getopt.GetoptError:
        print 'main.py --log <osquerylogfile> --algorithm <LSTM/OCSVM> --config_file <settings.ini location>'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print 'main.py --log <osquerylogfile> --algorithm <LSTM/OCSVM> --config_file <settings.ini location>'
            print '--algorithm can be LSTM or OCSVM.'
            sys.exit()
        elif opt in ("-l", "--log"):
            LOG_FILE = arg
        elif opt in ("-a", "--algorithm"):
            ALGORITHM = arg
        elif opt in ("-c", "--settings"):
            CONFIG = arg

    ## First, training on current log files
    #train_lstm()

    ## Watch for new logs
    logfiles = tail_F(LOG_FILE)
    if ALGORITHM.lower() == 'lstm':
        forecast_lstm(logfiles)
    elif ALGORITHM.lower() == 'ocsvm':
        call_ocsvm()