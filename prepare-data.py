import os
import csv
import itertools
import functools
import tensorflow as tf
import numpy as np
import array
import json

# Constants
tf.flags.DEFINE_string(
  "input_dir", os.path.abspath("./data"),
  "Input directory containing original JSON data files (default = './data')"
)

TRAIN_PATH = os.path.join(tf.flags.FLAGS.input_dir, "deset.json")
#
def create_iter_generator(filename):
    with open(filename) as file:
        reader = json.load(file)
        for row in reader:
            yield row

def create_vocabulary():
    iter_generator = create_iter_generator(TRAIN_PATH)
    for x in iter_generator:
        print x['columns']
        exit()
    #input_iteration = (x[0] + " " + x[1] for x in iter_generator)

    print("Creating vocabulary...")
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        FLAGS.max_sentence_len,
        min_frequency=min_frequency,
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

def create_tfrecords_file(input_filename, output_filename, example_fn):
  """
  Creates a TFRecords file for the given input data and
  example transformation function
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  print("Creating TFRecords file at {}...".format(output_filename))
  for i, row in enumerate(create_iter_generator(input_filename)):
      x = example_fn(row)
      writer.write(x.SerializeToString())

  print("Wrote to {}".format(output_filename))
  writer.close()

def transform_sentence(sequence, vocab_processor):
  """
  Maps a single sentence into the integer vocabulary.
  Returns an array.
  """
  return next(vocab_processor.transform([sequence])).tolist()

def create_example_train(row, vocab):
  """
  Returns the a tensorflow Example Protocol Buffer object.
  """
  context, utterance, label = row
  context_transformed = transform_sentence(context, vocab)
  utterance_transformed = transform_sentence(utterance, vocab)
  context_len = len(next(vocab._tokenizer([context])))
  utterance_len = len(next(vocab._tokenizer([utterance])))
  label = int(float(label))

  # New Example
  example = tf.train.Example()
  example.features.feature["context"].int64_list.value.extend(context_transformed)
  example.features.feature["utterance"].int64_list.value.extend(utterance_transformed)
  example.features.feature["context_len"].int64_list.value.extend([context_len])
  example.features.feature["utterance_len"].int64_list.value.extend([utterance_len])
  example.features.feature["label"].int64_list.value.extend([label])
  return example


if __name__ == "__main__":
    vocabulary = create_vocabulary()

    # Create vocabulary.txt file
    #write_vocabulary(vocabulary, os.path.join(FLAGS.output_dir, "vocabulary.txt"))

    # Save vocab processor
    #vocabulary.save(os.path.join(tf.flags.FLAGS.output_dir, "vocab_processor.bin"))

    # Create train.tfrecords
    """create_tfrecords_file(
        input_filename=TRAIN_PATH,
        output_filename=os.path.join(tf.flags.FLAGS.output_dir, "train.tfrecords"),
        example_fn=functools.partial(create_example_train, vocab=vocabulary))"""