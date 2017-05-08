#!/usr/bin/python
import sys, getopt

from lstm import load_model, predict
from prepare_data import action_to_vector
from helpers import tail_F

LOG_FILE = "/var/log/osquery/osqueryd.results.log"
ALGORITHM = "LSTM"
INTERVAL = 3   # in seconds
#-------------------#
has_new_data = False
#-------------------#

def tokenizer_fn(iterator):
  return (x.split(" ") for x in iterator)

def call_lstm(actions):
    model = load_model()

    global has_new_data
    actions_vectorized = []
    for action in actions:
        if (action != ''):
            print action
            action_in_vector = action_to_vector(action)
            actions_vectorized.append(action_in_vector)
            has_new_data = True

        elif (action == '' and has_new_data == True):
            print "Done reading query results."
            predicted = predict(model, actions_vectorized)
            print predicted
            has_new_data = False

def call_ocsvm():
    print "OCSVM not yet supported."

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "la", ["log=", "algorithm="])
        if (len(args) == 0):
            print "No arguments given. Using -l {} -a {}".format(LOG_FILE, ALGORITHM)
    except getopt.GetoptError:
        print 'main.py -l <osquerylogfile> -a <algorithm>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'main.py -l <inputfile> -a <algorithm>'
            print '--algorithm can be LSTM or OCSVM.'
            sys.exit()
        elif opt in ("-l", "--log"):
            LOG_FILE = arg
        elif opt in ("-a", "--algorithm"):
            LOG_FILE = arg

    #logfiles = watch_file(LOG_FILE, INTERVAL)
    logfiles = tail_F(LOG_FILE)
    if ALGORITHM.lower() == 'lstm':
        call_lstm(logfiles)
    elif ALGORITHM.lower() == 'ocsvm':
        call_ocsvm()