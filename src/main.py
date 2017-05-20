#!/usr/bin/python

import sys, getopt, json

from lstm import LSTM
from prepare_data import action_to_vector, tokenizer_fn
from helpers import tail_F

#-------------------#
LOG_FILE = "/var/log/osquery/osqueryd.results.log"
ALGORITHM = "LSTM"
CONFIG = "settings.ini"

actions_scores = {
    'usb_devices': 2,
    'listening_ports': 3,
    'arp_cache': 1,
    'suid_bin': 2,
    'shell_history': 2,
    'logged_in_users': 3,
    'ramdisk': 2,
    'open_files': 2,
    'open_sockets': 1,
    'last': 3,
    'etc_hosts': 3,
    'iptables': 2,
    'deb_packages': 2,
    'kernel_modules': 3,
    'firefox_addons': 1,
    'chrome_extensions': 1,
    'syslog': 3
}
action_name_prefix = 'pack_external_pack_'
#-------------------#
# Unchangeable parameters
has_new_data = False
#-------------------#

def call_lstm(actions):
    lstm = LSTM(CONFIG)
    model = lstm.load_model()

    global has_new_data
    actions_vectorized = []
    scores = []
    for action in actions:
        if (action != ''):
            action = json.loads(action)
            action_in_vector = action_to_vector(action, CONFIG)
            actions_vectorized.append(action_in_vector)

            action_name = action['name'].replace(action_name_prefix, '')
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

    logfiles = tail_F(LOG_FILE)
    if ALGORITHM.lower() == 'lstm':
        call_lstm(logfiles)
    elif ALGORITHM.lower() == 'ocsvm':
        call_ocsvm()