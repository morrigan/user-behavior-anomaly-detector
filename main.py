#!/usr/bin/python
import sys, getopt

def call_lstm():
    print "LSTM run"

def call_ocsvm():
    print "OCSVM run"


if __name__ == "__main__":
    LOG_FILE = "/var/log/osquery/osqueryd.results.log"
    ALGORITHM = "LSTM"

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

    if ALGORITHM.lower() == 'lstm':
        call_lstm()
    elif ALGORITHM.lower() == 'ocsvm':
        call_ocsvm()