#!/usr/bin/python

import os, csv
import functools
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

tf.flags.DEFINE_integer("min_word_frequency", 1, "Minimum frequency of occurrences in the vocabulary")
tf.flags.DEFINE_integer("max_vector_len", 30, "Maximum vector length")

#----------------------------------------------------------------#
TRAIN_PATH = os.path.join(FLAGS.input_dir, "train.json")
TEST_PATH = os.path.join(FLAGS.input_dir, "test.json")

CURRENT_PATH = TEST_PATH
OUTPUT_FILE = "test.csv"
#----------------------------------------------------------------#

def tokenizer_fn(iterator):
  return (x.split(" ") for x in iterator)

def json_dict_to_string(dictionary):
    return ''.join('{} {} '.format(key, val) for key, val in dictionary["columns"].items() if key != "time")


def create_vocabulary(train_path, test_path):
    print("Creating vocabulary...")

    iter_generator = helpers.create_iter_generator(train_path)
    input_iter = []
    for x in iter_generator:
        column = json_dict_to_string(x)
        input = x["action"] + " " + column
        input_iter.append(input)

    if (test_path):
        iter_generator = helpers.create_iter_generator(test_path)
        for x in iter_generator:
            column = json_dict_to_string(x)
            input = x["action"] + " " + column
            input_iter.append(input)

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        FLAGS.max_vector_len,
        min_frequency=FLAGS.min_word_frequency,
        tokenizer_fn=tokenizer_fn)
    vocab_processor.fit(input_iter)

    print("Done creating vocabulary.")
    return vocab_processor


def write_vocabulary(vocabulary_processor, outfile):
  """
  Writes the vocabulary to a file, one word per line.
  """
  with open(outfile, "w") as vocabfile:
    for id in range(len(vocabulary_processor.vocabulary_)):
      word =  vocabulary_processor.vocabulary_._reverse_mapping[id]
      vocabfile.write(word + "\n")
  print("Saved vocabulary to {}".format(outfile))


def create_csv_file(input_filename, output_filename, convert_fn):
    print("Creating CSV file at {}...".format(output_filename))

    actions = []

    for i, row in enumerate(helpers.create_iter_generator(input_filename)):
        columns = json_dict_to_string(row)

        output_row = {
            'action': row["action"] + " " + columns,    # action is of type added/removed
        }

        action_transformed, action_len = convert_fn(output_row)
        actions.append(action_transformed[:action_len])     # Remove padding :)

    output = pd.DataFrame(data={'action': actions})
    output.to_csv(output_filename, index=False, sep=";", quoting=csv.QUOTE_NONE, quotechar='')

    print("Wrote to {}".format(output_filename))


def transform_sentence(sequence, vocab_processor):
  """
  Maps a single sentence into the integer vocabulary.
  Returns an array.
  """
  return next(vocab_processor.transform([sequence])).tolist()


def extract_action(row, vocab):
    action = row['action'].strip()
    action_transformed = transform_sentence(action, vocab)
    action_len = len(next(vocab._tokenizer([action])))

    return action_transformed, action_len


def create_and_save_vocabulary(train, test = "", vocabularyfile = "vocabulary.txt", processorfile = "vocab_processor.bin"):
    vocabulary = create_vocabulary(train, test)
    # Create vocabulary.txt file
    write_vocabulary(vocabulary, os.path.join(FLAGS.output_dir, vocabularyfile))
    # Save vocab processor
    vocabulary.save(os.path.join(tf.flags.FLAGS.output_dir, processorfile))

    return vocabulary


def restore_vocabulary(filename = "", config_file = "settings.ini"):
    if (filename == ""):
        settings = helpers.getConfig(config_file)
        filename = settings.get('Data', 'vocabulary_processor')

    return tf.contrib.learn.preprocessing.VocabularyProcessor.restore(filename)


def action_to_vector(action, vocabulary):
    columns = json_dict_to_string(action)
    output_row = {
        'action': action["action"] + " " + columns
    }

    action, action_len = extract_action(output_row, vocabulary)
    action = action[:action_len]  # Remove padding :)
    return action


if __name__ == "__main__":
    #vocabulary = create_and_save_vocabulary(TRAIN_PATH, TEST_PATH)
    vocabulary = restore_vocabulary(os.path.join(tf.flags.FLAGS.output_dir, "vocab_processor.bin"))

    create_csv_file(
        input_filename=CURRENT_PATH,
        output_filename=os.path.join(tf.flags.FLAGS.output_dir, OUTPUT_FILE),
        convert_fn=functools.partial(extract_action, vocab=vocabulary))
