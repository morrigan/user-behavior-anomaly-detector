#!/usr/bin/python

import os, csv
import tensorflow as tf
import numpy as np
import pandas as pd
import helpers

# fix random seed for reproducibility
np.random.seed(7)


#-------------------------- Constants --------------------------#
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string(
  "input_dir", os.path.abspath("../data/real_logs"),
  "Input directory containing original JSON data files (default = '../data')"
)
tf.flags.DEFINE_string(
  "output_dir", os.path.abspath("../data"),
  "Output directory for TFrEcord files (default = '../data')")

tf.flags.DEFINE_integer("max_vector_len", 30, "Maximum vector length")

#----------------------------------------------------------------#
TRAIN_PATH = os.path.join(FLAGS.input_dir, "test_new")
TEST_PATH = os.path.join(FLAGS.input_dir, "20170609_Belma.log")

CURRENT_PATH = TRAIN_PATH
OUTPUT_FILE = "user1_train_new.csv"
#----------------------------------------------------------------#

### START VOCABULARY FUNCTIONS ###
def tokenizer_fn(iterator):
    return (x for x in iterator if x != "" and x != "0")


def create_vocabulary(train_path, test_path):
    print("Creating vocabulary...")

    iter_generator = helpers.create_iter_generator(train_path)
    input_iter = []
    for x in iter_generator:
        input = get_features(x)
        input_iter.append(input)

    if (test_path):
        iter_generator = helpers.create_iter_generator(test_path)
        for x in iter_generator:
            input = get_features(x)
            for x in input:
                input_iter.append(x)

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        FLAGS.max_vector_len,
        tokenizer_fn=tokenizer_fn)
    vocab_processor.fit(input_iter)

    print("Done creating vocabulary.")
    return vocab_processor


def write_vocabulary(vocabulary_processor, outfile):
  with open(outfile, "w") as vocabfile:
    for id in range(len(vocabulary_processor.vocabulary_)):
      word =  vocabulary_processor.vocabulary_._reverse_mapping[id]
      vocabfile.write(word + "\n")
  print("Saved vocabulary to {}".format(outfile))


def create_and_save_vocabulary(train, test="", vocabularyfile="vocabulary.txt", processorfile="vocab_processor.bin"):
    vocabulary = create_vocabulary(train, test)

    # Create vocabulary.txt file
    write_vocabulary(vocabulary, os.path.join(FLAGS.output_dir, vocabularyfile))
    # Save vocab processor
    vocabulary.save(os.path.join(tf.flags.FLAGS.output_dir, processorfile))

    return vocabulary


def restore_vocabulary(filename = os.path.join(tf.flags.FLAGS.output_dir, "vocab_processor.bin")):
    return tf.contrib.learn.preprocessing.VocabularyProcessor.restore(filename)
### END VOCABULARY FUNCTIONS ###


def transform_sentence(sequence, vocab_processor):
    # Maps a single vector input into the integer vocabulary.
    if (type(sequence) is not list):
        sequence = [sequence]


    print sequence
    vector = next(vocab_processor.transform(sequence)).tolist()
    print "Vector:"
    print vector
    print "====="
    vector_len = len(next(vocab_processor._tokenizer(sequence)))
    vector = vector[:vector_len]

    return vector


def get_features(line):
    structure = ["added_or_removed", "hour", "usb_devices", "kernel_modules", "open_sockets", "open_sockets",
                 "open_sockets", "open_sockets", "open_files", "logged_in_users", "logged_in_users", "shell_history",
                 "listening_ports", "arp_cache", "arp_cache", "syslog", "syslog"]

    # First feature
    added_or_removed = "2"  # added
    if (line["action"] == "removed"):
        added_or_removed = "1"
    # Second feature
    time = helpers.extract_hour(line["unixTime"])

    # Other osquery features
    columns = line["columns"].values()

    # Compatibility with old shell_history query
    #if (line["name"] == "pack_external_pack_shell_history"):
        #columns = str(helpers._parse_shell_history(columns))

    initial_vector = [added_or_removed, time] + ["0"] * (len(structure) - 2)
    # Put action columns in the right place of vector according to structure
    index = structure.index(line["name"].replace('pack_external_pack_', ''))

    for i in range(len(columns)):
        initial_vector[index + i] = columns[i]

    return initial_vector


"""
    Takes logline in json format and vocabulary object.
    Prepare to extract features from logline.
    Return dictionary containing features vector in key name action
"""
def action_to_vector(line, vocabulary):
    features_vector = get_features(line)

    action = transform_sentence(features_vector, vocabulary)

    return action


def create_csv_file(input_filename, output_filename, vocabulary):
    print("Creating CSV file at {}...".format(output_filename))

    actions = []

    for i, row in enumerate(helpers.create_iter_generator(input_filename)):
        action_transformed = action_to_vector(row, vocabulary)
        actions.append(action_transformed)

    output = pd.DataFrame(data={'action': actions})
    output.to_csv(output_filename, index=False, sep=";", quoting=csv.QUOTE_NONE, quotechar='')

    print("Wrote to {}".format(output_filename))


if __name__ == "__main__":
    #vocabulary = create_and_save_vocabulary(TRAIN_PATH, TEST_PATH)
    vocabulary = create_and_save_vocabulary(TRAIN_PATH)
    #vocabulary = restore_vocabulary()

    create_csv_file(
        input_filename=CURRENT_PATH,
        output_filename=os.path.join(tf.flags.FLAGS.output_dir, OUTPUT_FILE),
        vocabulary=vocabulary)
